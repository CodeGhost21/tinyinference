use tinyinference_llm::providers::openai::codex::OpenAiCodexRouting;
use tinyinference_llm::providers::registry::{
    CredentialResolver, ProviderBuildOptions, ProviderRegistry,
};
use tinyinference_llm::providers::{ProviderKind, ProviderSpec};

fn registry() -> ProviderRegistry<impl CredentialResolver> {
    ProviderRegistry::new(|_: &ProviderSpec| Ok("\n test-key \r\n".to_string()))
}

#[tokio::test]
async fn registry_routes_send_the_configured_wire_request() {
    use tinyinference_llm::message::Message;
    use tinyinference_llm::model::ModelRequest;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    for variant in ["local", "cloud", "codex", "openrouter"] {
        let oauth = variant == "codex";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let capture = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let read = stream.read(&mut buffer).await.unwrap();
                assert!(read > 0, "connection closed before request");
                bytes.extend_from_slice(&buffer[..read]);
                if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&bytes[..end]);
                    let length: usize = header
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|value| value.trim().parse().unwrap())
                        })
                        .unwrap();
                    if bytes.len() >= end + 4 + length {
                        let body =
                            serde_json::from_slice::<serde_json::Value>(&bytes[end + 4..]).unwrap();
                        stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").await.unwrap();
                        return (header.to_string(), body);
                    }
                }
            }
        });
        let kind = match variant {
            "local" => ProviderKind::Ollama,
            "openrouter" => ProviderKind::OpenRouter,
            _ => ProviderKind::OpenAi,
        };
        let mut spec = ProviderSpec::for_kind(kind);
        let host = match variant {
            "openrouter" => "openrouter.ai",
            _ => "127.0.0.1",
        };
        spec.base_url = format!("http://{host}:{}/v1", address.port());
        let route = OpenAiCodexRouting {
            endpoint: spec.base_url.clone(),
            using_oauth: oauth,
            account_id: None,
        };
        let headers = vec![("X-Host".into(), "test-host".into())];
        let query = vec![("client_version".into(), "test-version".into())];
        let model = registry()
            .build_with(
                &spec,
                ProviderBuildOptions {
                    temperature_override: Some(0.25),
                    num_ctx: Some(8192),
                    codex: oauth.then_some(&route),
                    extra_headers: &headers,
                    extra_query_params: &query,
                    http: Some(
                        reqwest::Client::builder()
                            .no_proxy()
                            .resolve(host, address)
                            .build()
                            .unwrap(),
                    ),
                    ..Default::default()
                },
            )
            .unwrap();
        let result = model
            .invoke(
                &(),
                ModelRequest::new(vec![Message::user("hello")]).with_max_tokens(123),
            )
            .await;
        assert!(result.is_err());
        let (header, body) = capture.await.unwrap();
        assert!(header.contains("x-host: test-host"));
        assert!(header.contains("client_version=test-version"));
        assert_eq!(body["model"], spec.model);
        assert_eq!(body["temperature"], 0.25);
        if oauth {
            assert!(header.starts_with("POST /v1/responses?"));
            assert!(header.contains("authorization: Bearer test-key"));
            assert!(body.get("max_output_tokens").is_none());
        } else {
            assert!(header.starts_with("POST /v1/chat/completions?"));
            if variant == "local" {
                assert!(!header.contains("authorization:"));
                assert_eq!(body["options"]["num_ctx"], 8192);
            } else {
                assert!(header.contains("authorization: Bearer test-key"));
            }
            if variant == "openrouter" {
                assert_eq!(body["provider"]["sort"], "price");
            }
            assert_eq!(body["max_tokens"], 123);
        }
    }
}
