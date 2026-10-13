//! Wire shapes for hosts serving text-only OpenAI-compatible HTTP endpoints.
//!
//! These types describe incoming chat requests and outgoing unary responses,
//! SSE chunks and model listings. They carry no routing, authentication,
//! inference execution or HTTP framework policy. Hosts construct responses and
//! apply their own request limits. The richer client-side wire types in the
//! parent module remain separate: accepting a text-only server request must not
//! silently broaden its message or tool contract.

use serde::{Deserialize, Serialize};

// ── Chat Completions ──────────────────────────────────────────────────────────

/// Request payload for OpenAI-compatible chat completions (`POST /v1/chat/completions`).
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Identifier of the model to query.
    pub model: String,
    /// List of input messages in conversation order.
    pub messages: Vec<ChatCompletionMessage>,
    /// Whether to stream back partial progress via SSE chunks.
    #[serde(default)]
    pub stream: bool,
    /// Sampling temperature supplied by the caller; hosts validate its range.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Legacy maximum tokens limit (superseded by `max_completion_tokens`).
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Maximum number of output tokens for newer OpenAI / reasoning models.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    /// Optional opaque tool definitions; hosts decide whether to accept them.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
}

/// A text-only message accepted or returned by a compatible server.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionMessage {
    /// Message role supplied by the host or caller.
    pub role: String,
    /// Visible message text.
    pub content: String,
}

/// A unary chat completion returned by a compatible server.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    /// Host-assigned response or model identifier.
    pub id: String,
    /// Wire object discriminator chosen by the host.
    pub object: &'static str,
    /// Host-supplied creation time in Unix seconds.
    pub created: i64,
    /// Model identifier selected by the host.
    pub model: String,
    /// Completion candidates in wire order.
    pub choices: Vec<ChatCompletionChoice>,
    /// Token usage for the completed request.
    pub usage: ChatCompletionUsage,
}

/// One candidate in a unary chat completion.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    /// Zero-based candidate index.
    pub index: u32,
    /// The completed assistant message.
    pub message: ChatCompletionMessage,
    /// Reason generation stopped, supplied when known.
    pub finish_reason: String,
}

/// Token counts for a completed request.
#[derive(Debug, Serialize)]
pub struct ChatCompletionUsage {
    /// Input tokens processed for this request.
    pub prompt_tokens: u64,
    /// Output tokens generated for this request.
    pub completion_tokens: u64,
    /// Total input and output token count.
    pub total_tokens: u64,
}

// ── Streaming (SSE) ───────────────────────────────────────────────────────────

/// An incremental chat response carried in one SSE data event.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    /// Host-assigned response or model identifier.
    pub id: String,
    /// Wire object discriminator chosen by the host.
    pub object: &'static str,
    /// Host-supplied creation time in Unix seconds.
    pub created: i64,
    /// Model identifier selected by the host.
    pub model: String,
    /// Completion candidates in wire order.
    pub choices: Vec<ChatCompletionChunkChoice>,
}

/// One candidate delta carried by an SSE chunk.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    /// Zero-based candidate index.
    pub index: u32,
    /// Incremental message fields.
    pub delta: ChatCompletionDelta,
    /// Reason generation stopped, supplied when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Incremental role and visible text fields in a streamed choice.
#[derive(Debug, Serialize)]
pub struct ChatCompletionDelta {
    /// Message role supplied by the host or caller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Visible message text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── Models ────────────────────────────────────────────────────────────────────

/// The top-level response for an OpenAI-compatible models listing.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    /// Wire object discriminator chosen by the host.
    pub object: &'static str,
    /// Models advertised by this server.
    pub data: Vec<ModelObject>,
}

/// One model advertised by a compatible server.
#[derive(Debug, Serialize)]
pub struct ModelObject {
    /// Host-assigned response or model identifier.
    pub id: String,
    /// Wire object discriminator chosen by the host.
    pub object: &'static str,
    /// Host-supplied creation time in Unix seconds.
    pub created: i64,
    /// Model owner or provider label.
    pub owned_by: String,
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod tests;
