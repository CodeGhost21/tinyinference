use super::*;

fn no_env(_: &str) -> Option<String> {
    None
}

fn inputs<'a>(base: Option<&'a str>, key: Option<&'a str>) -> LocalRuntimeInputs<'a> {
    LocalRuntimeInputs {
        base_url: base,
        api_key: key,
        num_ctx: Some(8192),
        env: no_env,
    }
}

fn ok(provider: &str, i: &LocalRuntimeInputs<'_>) -> ResolvedLocalRuntime {
    resolve_local_runtime(provider, i).unwrap().unwrap()
}

#[test]
fn non_local_strings_do_not_resolve() {
    let i = inputs(None, None);
    for p in [
        "openai:gpt-4o",
        "anthropic:claude",
        "openhuman:x",
        "",
        "gpt-4o",
    ] {
        assert!(resolve_local_runtime(p, &i).is_none(), "{p}");
    }
}

#[test]
fn ollama_appends_v1_and_carries_num_ctx_and_temp() {
    let r = ok("ollama:Llama3.1:8b@0.2", &inputs(Some("http://h:1/"), None));
    assert_eq!(r.endpoint, "http://h:1/v1");
    assert_eq!(r.model, "Llama3.1:8b");
    assert_eq!(r.temperature, Some(0.2));
    assert_eq!(r.num_ctx, Some(8192));
    assert_eq!(r.auth, LocalRuntimeAuth::None);
}

#[test]
fn ollama_does_not_double_v1() {
    let r = ok("ollama:m", &inputs(Some("http://h:1/v1"), None));
    assert_eq!(r.endpoint, "http://h:1/v1");
}

#[test]
fn mlx_defaults_to_8080_and_ignores_key() {
    let r = ok("mlx:m", &inputs(None, Some("k")));
    assert_eq!(r.endpoint, "http://127.0.0.1:8080/v1");
    assert_eq!(r.auth, LocalRuntimeAuth::None);
    assert!(r.api_key.is_empty());
}

#[test]
fn omlx_defaults_to_8000_and_uses_bearer_with_key() {
    let r = ok("omlx:m", &inputs(None, Some("k")));
    assert_eq!(r.endpoint, "http://127.0.0.1:8000/v1");
    assert_eq!(r.auth, LocalRuntimeAuth::Bearer);
    assert_eq!(r.api_key, "k");
    assert_eq!(
        ok("omlx:m", &inputs(None, Some("  "))).auth,
        LocalRuntimeAuth::None
    );
}

#[test]
fn local_openai_config_url_beats_default() {
    let r = ok("local-openai:m", &inputs(Some("http://x:9/v1"), None));
    assert_eq!(r.endpoint, "http://x:9/v1");
}

#[test]
fn env_overrides_beat_config_url() {
    fn env(k: &str) -> Option<String> {
        match k {
            "MLX_SERVER_URL" => Some("http://env:1/v1".into()),
            _ => None,
        }
    }
    let i = LocalRuntimeInputs {
        env,
        ..inputs(Some("http://cfg/v1"), None)
    };
    assert_eq!(ok("mlx:m", &i).endpoint, "http://env:1/v1");
    fn shared(k: &str) -> Option<String> {
        (k == "OPENHUMAN_LOCAL_INFERENCE_URL").then(|| "http://shared/v1".into())
    }
    let i = LocalRuntimeInputs { env: shared, ..i };
    assert_eq!(ok("local-openai:m", &i).endpoint, "http://shared/v1");
}

#[test]
fn empty_model_reports_expected_form() {
    let e = resolve_local_runtime("lmstudio:", &inputs(None, None))
        .unwrap()
        .unwrap_err();
    assert!(e.to_string().contains("use 'lmstudio:<model-id>'"), "{e}");
}

#[test]
fn prefix_alias_is_canonicalised_but_model_case_kept() {
    let r = ok("LM-Studio:Qwen/Q3-GGUF", &inputs(None, None));
    assert_eq!(r.kind, LocalProviderKind::LmStudio);
    assert_eq!(r.model, "Qwen/Q3-GGUF");
}

fn fake_env(name: &str) -> Option<String> {
    match name {
        "OPENHUMAN_OLLAMA_BASE_URL" => Some("http://injected-ollama:1".into()),
        "OPENHUMAN_LM_STUDIO_BASE_URL" => Some("http://injected-lms:2/v1".into()),
        _ => None,
    }
}

#[test]
fn ollama_and_lm_studio_honor_injected_env_not_ambient() {
    let mut i = inputs(None, None);
    i.env = fake_env;
    assert_eq!(ok("ollama:m", &i).endpoint, "http://injected-ollama:1/v1");
    assert_eq!(ok("lmstudio:m", &i).endpoint, "http://injected-lms:2/v1");
}

#[test]
fn debug_output_redacts_api_key() {
    let i = inputs(None, Some("sk-secret-key"));
    assert!(!format!("{i:?}").contains("sk-secret-key"));
    let r = ok("omlx:m", &i);
    assert_eq!(r.api_key, "sk-secret-key");
    assert!(!format!("{r:?}").contains("sk-secret-key"));
}

#[test]
fn debug_output_strips_url_credentials_query_and_fragment() {
    let url = "http://user:pw@host:1/v1?token=abc#frag";
    let i = inputs(Some(url), None);
    let shown = format!("{i:?}");
    for secret in ["user", "pw@", "token=abc", "frag"] {
        assert!(!shown.contains(secret), "{shown}");
    }
    assert!(shown.contains("host:1"), "{shown}");
    let r = ok("omlx:m", &inputs(Some(url), None));
    let shown = format!("{r:?}");
    for secret in ["user", "pw@", "token=abc", "frag"] {
        assert!(!shown.contains(secret), "{shown}");
    }
}
