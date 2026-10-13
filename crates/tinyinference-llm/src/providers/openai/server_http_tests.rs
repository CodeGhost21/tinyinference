use super::*;
use crate::MockModel;
use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;

struct Fake {
    fail: Option<BackendFailure>,
}

#[async_trait]
impl ChatBackend for Fake {
    async fn resolve(
        &self,
        model: &str,
        temperature: Option<f64>,
    ) -> Result<ResolvedChat, BackendFailure> {
        if let Some(f) = &self.fail {
            return Err(f.clone());
        }
        Ok(ResolvedChat {
            chat_model: Arc::new(MockModel::constant(format!("hi from {model}"))),
            model_id: model.to_string(),
            temperature: temperature.unwrap_or(0.5),
        })
    }

    async fn list_models(&self) -> Result<Vec<ListedModel>, BackendFailure> {
        Ok(vec![
            ListedModel::new("a", "x"),
            ListedModel::new("a", "dup"),
            ListedModel::new("b", "y"),
        ])
    }
}

fn app(fail: Option<BackendFailure>) -> Router {
    router::<()>(Arc::new(Fake { fail }))
}

async fn call(app: Router, req: Request<Body>) -> (StatusCode, String) {
    let res = app.oneshot(req).await.unwrap();
    let status = res.status();
    let body = res.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

fn post_json(v: serde_json::Value) -> Request<Body> {
    Request::post("/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(v.to_string()))
        .unwrap()
}

#[test]
fn strip_temperature_suffix_only_removes_numeric_suffixes() {
    assert_eq!(strip_temperature_suffix("gpt-4o@0.7"), "gpt-4o");
    assert_eq!(strip_temperature_suffix("llama3.1:8b@1"), "llama3.1:8b");
    assert_eq!(strip_temperature_suffix("gpt@beta"), "gpt@beta");
}

#[tokio::test]
async fn unary_completion_returns_openai_shape() {
    let (status, body) = call(
        app(None),
        post_json(json!({"model":"m1","messages":[{"role":"user","content":"yo"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "m1");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "hi from m1");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
}

#[tokio::test]
async fn streaming_completion_ends_with_done_and_role_on_first_chunk_only() {
    let (status, body) = call(
        app(None),
        post_json(json!({"model":"m1","stream":true,"messages":[{"role":"user","content":"yo"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
    assert!(body.contains("chat.completion.chunk"));
    assert_eq!(body.matches("\"role\":\"assistant\"").count(), 1, "{body}");
}

#[tokio::test]
async fn backend_failures_map_to_status_and_error_envelope() {
    for (failure, status, kind) in [
        (
            BackendFailure::invalid_request("nope"),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
        ),
        (
            BackendFailure::internal("boom"),
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
        ),
    ] {
        let (s, body) = call(
            app(Some(failure.clone())),
            post_json(json!({"model":"m","messages":[]})),
        )
        .await;
        assert_eq!(s, status);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], kind);
        assert_eq!(v["error"]["message"], failure.message);
    }
}

#[tokio::test]
async fn models_listing_dedupes_by_id() {
    let (status, body) = call(
        app(None),
        Request::get("/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    let ids: Vec<_> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["a", "b"]);
    assert_eq!(v["data"][0]["owned_by"], "x");
}

#[tokio::test]
async fn max_completion_tokens_takes_precedence_over_max_tokens() {
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"m","messages":[],"max_tokens":100,"max_completion_tokens":200
    }))
    .unwrap();
    assert_eq!(req.max_completion_tokens.or(req.max_tokens), Some(200));
}
