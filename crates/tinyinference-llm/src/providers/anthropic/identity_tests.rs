use super::*;
use futures::StreamExt;

#[test]
fn resolved_provider_identity_reaches_origins_and_unary_errors() {
    let model = AnthropicModel::new("key").with_provider_name("my-anthropic");
    let request = ModelRequest::new(vec![]);
    assert_eq!(model.origin_for(&request).provider, "my-anthropic");
    assert_eq!(
        model
            .provider_error(&request, "rejected".into(), Some(400), None, None, None)
            .provider,
        "my-anthropic"
    );
}

#[tokio::test]
async fn resolved_provider_identity_reaches_stream_completions_and_failures() {
    for (bytes, succeeds) in [
        ("data: {\"type\":\"message_stop\"}\n\n", true),
        (
            "data: {\"type\":\"error\",\"error\":{\"message\":\"rejected\",\"type\":\"invalid_request_error\"}}\n\n",
            false,
        ),
    ] {
        let mut stream = stream::stream_from_bytes_with_provider(
            vec![bytes.as_bytes().to_vec()],
            "claude-test",
            "my-anthropic",
        );
        let mut terminal = false;
        while let Some(item) = stream.next().await {
            match item {
                ModelStreamItem::Completed(response) => {
                    assert!(succeeds);
                    assert_eq!(response.message.origin.unwrap().provider, "my-anthropic");
                    terminal = true;
                }
                ModelStreamItem::ProviderFailed(error) => {
                    assert!(!succeeds);
                    assert_eq!(error.provider, "my-anthropic");
                    terminal = true;
                }
                _ => {}
            }
        }
        assert!(terminal);
    }
}
