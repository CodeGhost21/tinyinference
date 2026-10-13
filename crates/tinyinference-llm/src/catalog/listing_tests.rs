use super::*;

#[test]
fn claude_code_and_cli_endpoints_have_no_models_listing() {
    assert!(!endpoint_hosts_models_listing(
        "claude-code",
        "https://example.com/v1"
    ));
    assert!(!endpoint_hosts_models_listing("x", "cli://claude-code"));
}

#[test]
fn http_and_unparseable_endpoints_are_probed() {
    assert!(endpoint_hosts_models_listing(
        "openai",
        "https://api.openai.com/v1"
    ));
    assert!(endpoint_hosts_models_listing("x", "ftp://host/v1"));
    assert!(endpoint_hosts_models_listing("x", "not a url"));
}

#[test]
fn append_query_param_encodes_and_falls_back() {
    assert_eq!(
        append_query_param("https://h/models", "k", "a b"),
        "https://h/models?k=a+b"
    );
    assert_eq!(append_query_param("rel?x=1", "k", "v"), "rel?x=1&k=v");
    assert_eq!(append_query_param("rel", "k", "v"), "rel?k=v");
}

#[test]
fn openrouter_detected_by_slug_or_host() {
    assert!(is_openrouter_provider("OpenRouter", "http://localhost"));
    assert!(is_openrouter_provider("x", "https://openrouter.ai/api/v1"));
    assert!(is_openrouter_provider("x", "https://eu.openrouter.ai/v1"));
    assert!(!is_openrouter_provider("x", "https://notopenrouter.ai/v1"));
}
