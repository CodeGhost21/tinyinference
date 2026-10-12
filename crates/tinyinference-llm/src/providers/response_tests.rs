use super::*;

#[derive(Debug, Clone)]
struct BillingExtension {
    usd: f64,
    reported: bool,
}
#[derive(Debug, Clone)]
struct RawCall {
    id: String,
    name: String,
    arguments: String,
    extra_content: serde_json::Value,
}

#[test]
fn empty_response_does_not_require_default_on_billing_or_raw_calls() {
    let response = ChatResponse::<BillingExtension, RawCall>::default();
    assert!(response.text.is_none());
    assert!(response.tool_calls.is_empty());
    assert!(response.usage.is_none());
    assert!(response.reasoning_content.is_none());
    let portable: ChatResponse = ChatResponse::default();
    assert!(portable.usage.is_none());
}

#[test]
fn raw_call_identity_arguments_signature_and_exact_billing_survive_a_clone() {
    let raw = "{ \"subject\": \"雪\", \"limit\":1.00 }\n";
    let signature = serde_json::json!({"google":{"thought_signature":"opaque-signature"}});
    let response = ChatResponse {
        text: None,
        tool_calls: vec![RawCall {
            id: "provider-call-α".into(),
            name: "lookup".into(),
            arguments: raw.into(),
            extra_content: signature.clone(),
        }],
        usage: Some(BillingExtension {
            usd: 0.000000123456789,
            reported: true,
        }),
        reasoning_content: Some(" reason\nverbatim ".into()),
    }
    .clone();
    assert_eq!(response.tool_calls[0].id, "provider-call-α");
    assert_eq!(response.tool_calls[0].name, "lookup");
    assert_eq!(response.tool_calls[0].arguments.as_bytes(), raw.as_bytes());
    assert_eq!(response.tool_calls[0].extra_content, signature);
    let usage = response.usage.unwrap();
    assert_eq!(usage.usd.to_bits(), 0.000000123456789_f64.to_bits());
    assert!(usage.reported);
    assert!(response.text.is_none());
    assert_eq!(
        response.reasoning_content.as_deref(),
        Some(" reason\nverbatim ")
    );
}

#[test]
fn empty_text_and_reasoning_are_not_replaced_with_absence() {
    let response = ChatResponse::<crate::usage::Usage, crate::tool::ToolCall> {
        text: Some(String::new()),
        reasoning_content: Some(String::new()),
        usage: Some(crate::usage::Usage::new(0, 0)),
        ..Default::default()
    };
    assert_eq!(response.text.as_deref(), Some(""));
    assert_eq!(response.reasoning_content.as_deref(), Some(""));
    assert!(response.usage.is_some());
}

#[test]
fn provider_delta_preserves_start_identity_and_unparsed_argument_fragments() {
    let events = [
        ProviderDelta::ToolCallStart {
            call_id: "same-id".into(),
            tool_name: "lookup".into(),
        },
        ProviderDelta::ToolCallArgsDelta {
            call_id: "same-id".into(),
            delta: "{ \"x\": ".into(),
        },
        ProviderDelta::ThinkingDelta {
            delta: "reason\n".into(),
        },
        ProviderDelta::TextDelta {
            delta: "reply 雪".into(),
        },
        ProviderDelta::ToolCallArgsDelta {
            call_id: "same-id".into(),
            delta: "\"雪\" }".into(),
        },
    ];
    let mut args = String::new();
    for event in events.iter().cloned() {
        match event {
            ProviderDelta::ToolCallStart { call_id, tool_name } => {
                assert_eq!(call_id, "same-id");
                assert_eq!(tool_name, "lookup");
            }
            ProviderDelta::ToolCallArgsDelta { call_id, delta } => {
                assert_eq!(call_id, "same-id");
                args.push_str(&delta);
            }
            ProviderDelta::ThinkingDelta { delta } => assert_eq!(delta, "reason\n"),
            ProviderDelta::TextDelta { delta } => assert_eq!(delta, "reply 雪"),
        }
    }
    assert_eq!(args.as_bytes(), "{ \"x\": \"雪\" }".as_bytes());
}
