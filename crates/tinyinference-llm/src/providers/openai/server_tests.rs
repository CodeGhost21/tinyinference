use super::*;
use serde_json::{Value, json};

#[test]
fn chat_request_preserves_both_token_limits_and_optional_defaults() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "text-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 128,
        "max_completion_tokens": 256,
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();
    assert_eq!(request.model, "text-model");
    assert_eq!(request.messages[0].role, "user");
    assert_eq!(request.messages[0].content, "hello");
    assert!(!request.stream);
    assert_eq!(request.temperature, None);
    assert_eq!(request.max_tokens, Some(128));
    assert_eq!(request.max_completion_tokens, Some(256));
    assert_eq!(request.tools.unwrap()[0]["function"]["name"], "lookup");
    let minimal: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "text-model", "messages": []
    }))
    .unwrap();
    assert_eq!(
        (
            minimal.max_tokens,
            minimal.max_completion_tokens,
            minimal.tools
        ),
        (None, None, None)
    );
}

#[test]
fn text_only_requests_reject_missing_model_and_non_text_content() {
    for input in [
        json!({"messages": []}),
        json!({"model": "text-model", "messages": [{"role": "user", "content": null}]}),
        json!({"model": "text-model", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}),
    ] {
        assert!(serde_json::from_value::<ChatCompletionRequest>(input).is_err());
    }
}

#[test]
fn unary_completion_serializes_the_complete_server_envelope() {
    let response = ChatCompletionResponse {
        id: "completion-1".into(),
        object: "chat.completion",
        created: 123,
        model: "text-model".into(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".into(),
                content: "hello".into(),
            },
            finish_reason: "stop".into(),
        }],
        usage: ChatCompletionUsage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
        },
    };
    assert_eq!(
        serde_json::to_value(response).unwrap(),
        json!({
            "id": "completion-1", "object": "chat.completion", "created": 123, "model": "text-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })
    );
}

#[test]
fn streamed_chunks_omit_absent_fields_and_keep_the_terminal_reason() {
    for (delta, finish_reason, expected) in [
        (
            ChatCompletionDelta {
                role: Some("assistant".into()),
                content: None,
            },
            None,
            json!({"index": 0, "delta": {"role": "assistant"}}),
        ),
        (
            ChatCompletionDelta {
                role: None,
                content: Some("hello".into()),
            },
            None,
            json!({"index": 0, "delta": {"content": "hello"}}),
        ),
        (
            ChatCompletionDelta {
                role: None,
                content: None,
            },
            Some("stop".to_string()),
            json!({"index": 0, "delta": {}, "finish_reason": "stop"}),
        ),
    ] {
        let chunk = ChatCompletionChunk {
            id: "completion-1".into(),
            object: "chat.completion.chunk",
            created: 123,
            model: "text-model".into(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
        };
        let actual: Value = serde_json::to_value(chunk).unwrap();
        assert_eq!(
            actual,
            json!({"id": "completion-1", "object": "chat.completion.chunk", "created": 123,
            "model": "text-model", "choices": [expected]})
        );
    }
}

#[test]
fn model_listing_preserves_the_server_discriminators_and_owner() {
    let listing = ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: "text-model".into(),
            object: "model",
            created: 123,
            owned_by: "provider".into(),
        }],
    };
    assert_eq!(
        serde_json::to_value(listing).unwrap(),
        json!({"object": "list", "data": [
            {"id": "text-model", "object": "model", "created": 123, "owned_by": "provider"}
        ]})
    );
}
