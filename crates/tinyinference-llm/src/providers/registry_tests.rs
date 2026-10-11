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
        assert!(!model.cache_identity().unwrap().contains("test-key"));
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
