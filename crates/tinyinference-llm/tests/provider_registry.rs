//! Loopback integration coverage of stateless provider construction.
use tinyinference_llm::providers::openai::codex::OpenAiCodexRouting;
use tinyinference_llm::providers::registry::{ProviderBuildOptions, ProviderRegistry};
use tinyinference_llm::providers::{ProviderKind, ProviderSpec};

#[tokio::test]
async fn registry_routes_send_the_configured_wire_request() {
    use tinyinference_llm::message::Message;
    use tinyinference_llm::model::ModelRequest;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    for variant in [
        "local_reject",
        "optional",
        "local",
        "cloud",
        "codex",
        "openrouter",
        "local_override",
        "openrouter_override",
    ] {
        let oauth = variant == "codex";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let native_ctx = match variant {
            "local" | "local_reject" => Some(8192),
            "local_override" => Some(4096),
            _ => None,
        };
        let capture = tokio::spawn(async move {
            for call in 0..if native_ctx.is_some() { 2 } else { 1 } {
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
                                serde_json::from_slice::<serde_json::Value>(&bytes[end + 4..])
                                    .unwrap();
                            if let Some(native_ctx) = native_ctx.filter(|_| call == 0) {
                                assert!(
                                    header.starts_with("POST /api/chat ")
                                        || header.starts_with("POST /api/chat?"),
                                    "unexpected native route: {}",
                                    header.lines().next().unwrap()
                                );
                                assert_eq!(body["options"]["num_ctx"], native_ctx);
                                if variant == "local_override" {
                                    assert_eq!(body["options"]["num_predict"], 32);
                                }
                                if variant != "local_reject" {
                                    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").await.unwrap();
                                    break;
                                }
                            }
                            stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").await.unwrap();
                            return (header.to_string(), body);
                        }
                    }
                }
            }
            unreachable!("the final captured request always returns")
        });
        let kind = match variant {
            "local" | "local_override" | "local_reject" => ProviderKind::Ollama,
            "openrouter" | "openrouter_override" => ProviderKind::OpenRouter,
            _ => ProviderKind::OpenAi,
        };
        let mut spec = ProviderSpec::for_kind(kind);
        let host = match variant {
            "openrouter" | "openrouter_override" => "openrouter.ai",
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
        spec.requires_api_key = variant != "optional" && spec.requires_api_key;
        let registry = ProviderRegistry::new(move |_: &ProviderSpec| {
            Ok(if variant == "optional" {
                String::new()
            } else {
                "test-key".into()
            })
        });
        let supplied_options = match variant {
            "local_override" => {
                serde_json::json!({"options":{"num_ctx":4096,"num_predict":32},"top_k":7})
            }
            "openrouter_override" => {
                serde_json::json!({"provider":{"sort":"latency","only":["DeepInfra"]},"top_k":7})
            }
            _ => serde_json::json!({"top_k":7}),
        };
        let model = registry
            .build_with(
                &spec,
                ProviderBuildOptions {
                    temperature_override: Some(0.25),
                    num_ctx: Some(8192),
                    default_provider_options: Some(supplied_options),
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
        if let Some(num_ctx) = native_ctx {
            assert_eq!(
                model.profile().unwrap().max_input_tokens,
                Some(num_ctx as u64)
            );
        }
        let result = model
            .invoke(
                &(),
                ModelRequest::new(vec![Message::user("hello")]).with_max_tokens(123),
            )
            .await;
        assert!(result.is_err());
        let (header, body) = capture.await.unwrap();
        if variant == "local_reject" {
            assert!(
                matches!(&result, Err(tinyinference_llm::Error::Model(message)) if message.contains("400")),
                "native setup rejection must stop inference: {result:?}"
            );
            continue;
        }
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
            if variant == "local" || variant == "local_override" || variant == "optional" {
                assert!(!header.contains("authorization:"));
                if variant != "optional" {
                    assert_eq!(
                        body["options"]["num_ctx"],
                        if variant == "local_override" {
                            4096
                        } else {
                            8192
                        }
                    );
                }
            } else {
                assert!(header.contains("authorization: Bearer test-key"));
            }
            if variant == "openrouter" || variant == "openrouter_override" {
                assert_eq!(
                    body["provider"]["sort"],
                    if variant == "openrouter_override" {
                        "latency"
                    } else {
                        "price"
                    }
                );
                if variant == "openrouter_override" {
                    assert_eq!(body["provider"]["only"], serde_json::json!(["DeepInfra"]));
                }
            }
            assert_eq!(body["top_k"], 7);
            assert_eq!(body["max_tokens"], 123);
        }
    }
}

async fn capture_compatible_request(
    build: impl FnOnce(&str) -> std::sync::Arc<dyn tinyinference_llm::model::ChatModel<()>>,
) -> (String, serde_json::Value) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let model = build(&endpoint);
    let captured = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        loop {
            let mut buffer = [0; 4096];
            let read = stream.read(&mut buffer).await.unwrap();
            assert!(read > 0);
            bytes.extend_from_slice(&buffer[..read]);
            if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&bytes[..end]);
                let length: usize = header
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse().unwrap())
                    })
                    .unwrap();
                if bytes.len() >= end + 4 + length {
                    let body = serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
                    let header = header.into_owned();
                    stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").await.unwrap();
                    return (header, body);
                }
            }
        }
    });
    let request = tinyinference_llm::model::ModelRequest::builder()
        .messages(vec![tinyinference_llm::message::Message::user("hello")])
        .build();
    let _ = model.invoke(&(), request).await;
    captured.await.unwrap()
}

#[tokio::test]
async fn explicit_local_modes_preserve_legacy_compatible_wire_without_native_warmup() {
    use tinyinference_llm::providers::openai::{AuthStyle, build_local_runtime_chat_model};
    use tinyinference_llm::providers::registry::{LocalCompatibleKind, OllamaContextMode};
    for (slug, kind, key) in [
        ("mlx", Some(LocalCompatibleKind::Mlx), ""),
        ("omlx", Some(LocalCompatibleKind::Omlx), "local-key"),
        (
            "local-openai",
            Some(LocalCompatibleKind::OpenAi),
            "local-key",
        ),
        ("ollama", None, ""),
    ] {
        for model_id in ["CaseSensitive:latest", "claude-3-5-sonnet-latest"] {
            let auth = if key.is_empty() {
                AuthStyle::None
            } else {
                AuthStyle::Bearer
            };
            let num_ctx = (slug == "ollama").then_some(8192);
            let (old_header, old_body) = capture_compatible_request(|endpoint| {
                build_local_runtime_chat_model(
                    slug,
                    endpoint,
                    key,
                    auth.clone(),
                    model_id,
                    &[],
                    Some(0.25),
                    num_ctx,
                )
            })
            .await;
            let (new_header, new_body) = capture_compatible_request(|endpoint| {
                let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok(key.into()));
                let mut spec = ProviderSpec::for_kind(if slug == "ollama" {
                    ProviderKind::Ollama
                } else {
                    ProviderKind::Compatible
                })
                .with_model(model_id);
                spec.provider = slug.into();
                spec.base_url = endpoint.into();
                spec.requires_api_key = false;
                registry
                    .build_with(
                        &spec,
                        ProviderBuildOptions {
                            local_compatible: kind,
                            ollama_context_mode: OllamaContextMode::CompatibleOnly,
                            num_ctx,
                            temperature_override: Some(0.25),
                            auth_style: Some(auth.clone()),
                            ..Default::default()
                        },
                    )
                    .unwrap()
            })
            .await;
            assert!(old_header.starts_with("POST /v1/chat/completions "));
            assert!(
                new_header.starts_with("POST /v1/chat/completions "),
                "{new_header}"
            );
            assert_eq!(new_body, old_body, "{slug}:{model_id}");
            let authorization = |header: &str| {
                header
                    .lines()
                    .find(|line| line.to_ascii_lowercase().starts_with("authorization:"))
                    .map(str::to_owned)
            };
            assert_eq!(authorization(&new_header), authorization(&old_header));
        }
    }
}
