use super::*;

fn registry() -> ProviderRegistry<impl CredentialResolver> {
    ProviderRegistry::new(|_: &ProviderSpec| Ok("test-key".to_string()))
}

#[test]
fn builds_cloud_and_local_profiles_from_resolved_specs() {
    for kind in [
        ProviderKind::OpenAi,
        ProviderKind::OpenRouter,
        ProviderKind::Anthropic,
        ProviderKind::Ollama,
    ] {
        let spec = ProviderSpec::for_kind(kind.clone());
        let model = registry().build(&spec).unwrap();
        let profile = model.profile().unwrap();
        assert_eq!(profile.model.as_deref(), Some(spec.model.as_str()));
        assert_eq!(profile.tool_calling, kind != ProviderKind::Ollama);
        assert!(
            model
                .cache_identity()
                .is_none_or(|identity| !identity.contains("test-key"))
        );
    }
}

#[test]
fn explicit_auth_and_text_tools_keep_anthropic_on_compatible_adapter() {
    let spec = ProviderSpec::for_kind(ProviderKind::Anthropic);
    let model = registry()
        .build_with(
            &spec,
            ProviderBuildOptions {
                native_tool_calling: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!model.profile().unwrap().tool_calling);
    let spec = ProviderSpec {
        provider: "private".into(),
        ..spec
    };
    let model = registry()
        .build_with(
            &spec,
            ProviderBuildOptions {
                auth_style: Some(AuthStyle::Bearer),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        model.profile().unwrap().provider.as_deref(),
        Some("private")
    );
}

#[test]
fn discovery_uses_same_endpoint_model_auth_and_pinned_routes() {
    let spec = ProviderSpec::for_kind(ProviderKind::OpenRouter);
    let options = ProviderBuildOptions {
        default_provider_options: Some(serde_json::json!({"provider":{"only":["DeepInfra"]}})),
        ..Default::default()
    };
    let request = registry()
        .discovery_request(&spec, &options)
        .unwrap()
        .unwrap();
    assert_eq!(request.endpoint, spec.base_url);
    assert_eq!(request.model, spec.model);
    assert_eq!(
        request.headers,
        vec![("Authorization".into(), "Bearer test-key".into())]
    );
    assert_eq!(request.pinned_providers, vec!["DeepInfra"]);
    let spec = ProviderSpec::for_kind(ProviderKind::Anthropic);
    let request = registry()
        .discovery_request(&spec, &Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        request.headers,
        vec![
            ("x-api-key".into(), "test-key".into()),
            ("anthropic-version".into(), "2023-06-01".into()),
        ]
    );
}

#[test]
fn local_and_oauth_discovery_do_not_read_credentials() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| -> Result<String> {
        panic!("discovery must not consult credentials when no listing exists")
    });
    let spec = ProviderSpec::for_kind(ProviderKind::Ollama);
    assert!(
        registry
            .discovery_request(&spec, &Default::default())
            .unwrap()
            .is_none()
    );
    let spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    let route = OpenAiCodexRouting {
        endpoint: "https://chatgpt.com/backend-api/codex".into(),
        using_oauth: true,
        account_id: Some("account".into()),
    };
    assert!(
        registry
            .discovery_request(
                &spec,
                &ProviderBuildOptions {
                    codex: Some(&route),
                    ..Default::default()
                }
            )
            .unwrap()
            .is_none()
    );
}

#[test]
fn invalid_specs_and_missing_required_keys_fail_before_construction() {
    let registry = ProviderRegistry::new(|_: &ProviderSpec| Ok(String::new()));
    let mut spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    assert!(matches!(registry.build(&spec), Err(Error::Validation(_))));
    spec.requires_api_key = false;
    assert!(registry.build(&spec).is_ok());
    spec.model.clear();
    assert!(matches!(registry.build(&spec), Err(Error::Validation(_))));
    spec.model = "gpt-test".into();
    spec.base_url.clear();
    assert!(matches!(registry.build(&spec), Err(Error::Validation(_))));
}

#[test]
fn credential_errors_are_returned_without_rewriting() {
    let registry =
        ProviderRegistry::new(|_: &ProviderSpec| Err(Error::Unsupported("keyring offline".into())));
    let spec = ProviderSpec::for_kind(ProviderKind::OpenAi);
    assert!(
        matches!(registry.build(&spec), Err(Error::Unsupported(message)) if message == "keyring offline")
    );
}

#[test]
fn price_ordering_is_only_supplied_to_openrouter() {
    assert_eq!(
        openrouter_default_provider_options("https://openrouter.ai/api/v1"),
        Some(serde_json::json!({"provider":{"sort":"price"}}))
    );
    assert_eq!(
        openrouter_default_provider_options("https://api.openai.com/v1"),
        None
    );
}

#[tokio::test]
async fn registry_routes_send_the_configured_wire_request() {
    use crate::message::Message;
    use crate::model::ModelRequest;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    for variant in ["local", "cloud", "codex", "anthropic", "openrouter"] {
        let oauth = variant == "codex";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let capture = std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            started.elapsed() < std::time::Duration::from_secs(5),
                            "adapter never connected"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(error) => panic!("accept failed: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
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
                        stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
                        return (header.to_string(), body);
                    }
                }
            }
        });
        let kind = match variant {
            "local" => ProviderKind::Ollama,
            "anthropic" => ProviderKind::Anthropic,
            "openrouter" => ProviderKind::OpenRouter,
            _ => ProviderKind::OpenAi,
        };
        let mut spec = ProviderSpec::for_kind(kind);
        let host = match variant {
            "anthropic" => "api.anthropic.com",
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
        let result = model.invoke(&(), ModelRequest::new(vec![Message::user("hello")]).with_max_tokens(123)).await;
        eprintln!("variant={variant} result={result:?}");
        assert!(result.is_err());
        let (header, body) = capture.join().unwrap();
        assert!(header.contains("x-host: test-host"));
        if variant != "anthropic" {
            assert!(header.contains("client_version=test-version"));
        }
        assert_eq!(body["model"], spec.model);
        assert_eq!(body["temperature"], 0.25);
        if oauth {
            assert!(header.starts_with("POST /v1/responses?"));
            assert!(header.contains("authorization: Bearer test-key"));
            assert!(body.get("max_output_tokens").is_none());
        } else if variant == "anthropic" {
            assert!(header.starts_with("POST /v1/messages "));
            assert!(header.contains("x-api-key: test-key"));
            assert_eq!(body["max_tokens"], 123);
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

#[test]
fn debugging_registry_options_never_discloses_transport_secrets() {
    let headers = vec![("Authorization".into(), "secret-header".into())];
    let query = vec![("token".into(), "secret-query".into())];
    let route = OpenAiCodexRouting {
        endpoint: "https://user:secret-password@example.com?token=secret-url".into(),
        using_oauth: true,
        account_id: Some("secret-account".into()),
    };
    let debug = format!(
        "{:?}",
        ProviderBuildOptions {
            extra_headers: &headers,
            extra_query_params: &query,
            codex: Some(&route),
            user_agent: Some("secret-user-agent"),
            ..Default::default()
        }
    );
    assert!(!debug.contains("secret"));
    assert!(debug.contains("Authorization"));
}
