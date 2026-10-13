//! Generic OpenAI-compatible `/chat/completions` and `/models` axum handlers.
//!
//! The host supplies a [`ChatBackend`] that turns a request's `model` string
//! into a ready [`ChatModel`] (routing, credentials, privacy and egress policy
//! all live there) and lists the models it advertises. This module owns only
//! the wire handling: request mapping, unary and SSE responses, and error
//! envelopes. Authentication is the host's: wrap the returned router in its own
//! middleware. Nest it, e.g. `.nest("/v1", router(backend))`.

use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{self, StreamExt};
use serde_json::json;
use tracing::{debug, warn};

use super::server::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta,
    ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage,
    ModelObject, ModelsResponse,
};
use crate::message::Message;
use crate::model::{ChatModel, ModelRequest, ModelStreamItem};

const LOG_PREFIX: &str = "[inference::http]";

/// How a [`ChatBackend`] failure maps onto an HTTP status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFailureKind {
    /// The request named something the backend cannot serve (HTTP 400).
    InvalidRequest,
    /// The backend failed for its own reasons (HTTP 500).
    Internal,
}

/// A backend failure carried to the caller as an OpenAI-style error body.
#[derive(Debug, Clone)]
pub struct BackendFailure {
    /// Status mapping.
    pub kind: BackendFailureKind,
    /// Caller-visible message.
    pub message: String,
}

impl BackendFailure {
    /// A 400-class failure.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            kind: BackendFailureKind::InvalidRequest,
            message: message.into(),
        }
    }

    /// A 500-class failure.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: BackendFailureKind::Internal,
            message: message.into(),
        }
    }

    fn into_response(self) -> Response {
        let (status, kind) = match self.kind {
            BackendFailureKind::InvalidRequest => {
                (StatusCode::BAD_REQUEST, "invalid_request_error")
            }
            BackendFailureKind::Internal => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        };
        (
            status,
            Json(json!({ "error": { "message": self.message, "type": kind }})),
        )
            .into_response()
    }
}

/// A chat model chosen for one request.
#[allow(missing_debug_implementations)]
pub struct ResolvedChat {
    /// The model to invoke.
    pub chat_model: Arc<dyn ChatModel<()>>,
    /// Provider-side model id sent on the request.
    pub model_id: String,
    /// Temperature to send.
    pub temperature: f64,
}

/// One entry of the `/models` listing.
#[derive(Debug, Clone)]
pub struct ListedModel {
    /// Model id as callers should send it.
    pub id: String,
    /// Owner or provider label.
    pub owned_by: String,
}

impl ListedModel {
    /// Builds an entry.
    pub fn new(id: impl Into<String>, owned_by: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            owned_by: owned_by.into(),
        }
    }
}

/// The host side of the OpenAI-compatible endpoint.
#[async_trait]
pub trait ChatBackend: Send + Sync {
    /// Resolves the request's `model` (and optional caller temperature) to a
    /// ready chat model.
    async fn resolve(
        &self,
        model: &str,
        temperature: Option<f64>,
    ) -> Result<ResolvedChat, BackendFailure>;

    /// Lists the models this backend advertises.
    async fn list_models(&self) -> Result<Vec<ListedModel>, BackendFailure>;
}

/// Builds the `/chat/completions` + `/models` router over `backend`.
pub fn router<S>(backend: Arc<dyn ChatBackend>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/models", get(models))
        .with_state(backend)
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn error_response(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message, "type": "internal_error" }})),
    )
        .into_response()
}

fn map_messages(req: &ChatCompletionRequest) -> Vec<Message> {
    req.messages
        .iter()
        .map(|m| match m.role.as_str() {
            "system" => Message::system(&m.content),
            "assistant" => Message::assistant(&m.content),
            "tool" => Message::tool("external-tool-call", &m.content),
            _ => Message::user(&m.content),
        })
        .collect()
}

async fn chat_completions(
    State(backend): State<Arc<dyn ChatBackend>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    debug!(
        model = %req.model,
        stream = req.stream,
        message_count = req.messages.len(),
        "{LOG_PREFIX} chat_completions: start"
    );
    let resolved = match backend.resolve(&req.model, req.temperature).await {
        Ok(r) => r,
        Err(e) => {
            warn!(kind = ?e.kind, "{LOG_PREFIX} chat_completions: backend resolve failed");
            return e.into_response();
        }
    };
    let ResolvedChat {
        chat_model,
        model_id,
        temperature,
    } = resolved;

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = now_unix();
    let model_name = req.model.clone();
    let mut model_request = ModelRequest::new(map_messages(&req))
        .with_model(model_id.clone())
        .with_temperature(temperature);
    if let Some(tokens) = req.max_completion_tokens.or(req.max_tokens) {
        model_request = model_request.with_max_tokens(tokens);
    }

    if req.stream {
        let model_stream = match chat_model.stream(&(), model_request).await {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, model = %model_id, "{LOG_PREFIX} chat_completions: stream start failed");
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("inference error: {e}"),
                );
            }
        };
        let first_delta = Arc::new(AtomicBool::new(true));
        let event_stream = model_stream
            .filter_map(move |item| {
                let cid = completion_id.clone();
                let model = model_name.clone();
                let first_delta = Arc::clone(&first_delta);
                async move {
                    let (content, finish_reason) = match item {
                        ModelStreamItem::MessageDelta(delta) if !delta.text.is_empty() => {
                            (Some(delta.text), None)
                        }
                        ModelStreamItem::Completed(_) => (None, Some("stop")),
                        ModelStreamItem::Failed(message) => {
                            return Some(stream_error_event(&message));
                        }
                        ModelStreamItem::ProviderFailed(error) => {
                            return Some(stream_error_event(&error.message));
                        }
                        _ => return None,
                    };
                    let role = first_delta
                        .swap(false, Ordering::Relaxed)
                        .then(|| "assistant".to_string());
                    let chunk = ChatCompletionChunk {
                        id: cid,
                        object: "chat.completion.chunk",
                        created,
                        model,
                        choices: vec![ChatCompletionChunkChoice {
                            index: 0,
                            delta: ChatCompletionDelta { role, content },
                            finish_reason,
                        }],
                    };
                    let data = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
                    Some(Ok::<Event, Infallible>(Event::default().data(data)))
                }
            })
            .chain(stream::once(async {
                Ok::<Event, Infallible>(Event::default().data("[DONE]"))
            }));
        debug!("{LOG_PREFIX} chat_completions: streaming response started");
        return Sse::new(event_stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    match chat_model.invoke(&(), model_request).await {
        Ok(model_response) => {
            debug!("{LOG_PREFIX} chat_completions: non-streaming ok");
            let usage = model_response.usage.unwrap_or_default();
            let response = ChatCompletionResponse {
                id: completion_id,
                object: "chat.completion",
                created,
                model: model_name,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: ChatCompletionMessage {
                        role: "assistant".to_string(),
                        content: model_response.text(),
                    },
                    finish_reason: "stop",
                }],
                usage: ChatCompletionUsage {
                    prompt_tokens: usage.input_tokens,
                    completion_tokens: usage.output_tokens,
                    total_tokens: usage.total_tokens,
                },
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            warn!("{LOG_PREFIX} chat_completions: inference failed: {e}");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("inference error: {e}"),
            )
        }
    }
}

fn stream_error_event(message: &str) -> Result<Event, Infallible> {
    let body = json!({"error": {"message": message, "type": "stream_error"}});
    Ok(Event::default().data(body.to_string()))
}

async fn models(State(backend): State<Arc<dyn ChatBackend>>) -> Response {
    debug!("{LOG_PREFIX} models: start");
    let listed = match backend.list_models().await {
        Ok(l) => l,
        Err(e) => {
            warn!("{LOG_PREFIX} models: backend failed");
            return e.into_response();
        }
    };
    let created = now_unix();
    let mut seen = HashSet::new();
    let data: Vec<ModelObject> = listed
        .into_iter()
        .filter(|m| seen.insert(m.id.clone()))
        .map(|m| ModelObject {
            id: m.id,
            object: "model",
            created,
            owned_by: m.owned_by,
        })
        .collect();
    debug!(model_count = data.len(), "{LOG_PREFIX} models: ok");
    (
        StatusCode::OK,
        Json(ModelsResponse {
            object: "list",
            data,
        }),
    )
        .into_response()
}

/// Strips a trailing numeric `@<temperature>` suffix from a model id.
pub fn strip_temperature_suffix(model: &str) -> &str {
    let trimmed = model.trim();
    let Some((head, tail)) = trimmed.rsplit_once('@') else {
        return trimmed;
    };
    if tail.parse::<f64>().is_ok() {
        head.trim()
    } else {
        trimmed
    }
}

#[cfg(test)]
#[path = "server_http_tests.rs"]
mod tests;
