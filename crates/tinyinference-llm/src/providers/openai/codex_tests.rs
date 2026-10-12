use super::*;
use tempfile::tempdir;

#[test]
fn oauth_routing_selects_subscription_endpoint_without_an_account_header() {
    for account in [None, Some(""), Some(" \t\n ")] {
        let routing = OpenAiCodexRouting::oauth(account);
        assert_eq!(routing.endpoint, "https://chatgpt.com/backend-api/codex");
        assert!(routing.using_oauth);
        assert_eq!(routing.account_id, None);
    }
}

#[test]
fn oauth_routing_preserves_the_trimmed_account_header() {
    let routing = OpenAiCodexRouting::oauth(Some("  account-example \n"));
    assert_eq!(routing.endpoint, "https://chatgpt.com/backend-api/codex");
    assert!(routing.using_oauth);
    assert_eq!(routing.account_id.as_deref(), Some("account-example"));
}

#[test]
fn standard_routing_preserves_host_endpoint_and_omits_oauth_headers() {
    let routing = OpenAiCodexRouting::standard("https://example.invalid/v1///");
    assert_eq!(routing.endpoint, "https://example.invalid/v1");
    assert!(!routing.using_oauth);
    assert_eq!(routing.account_id, None);
}

#[test]
fn client_version_prefers_explicit_override() {
    let (version, source) =
        resolve_openai_codex_client_version_from(Some("  0.200.0  ".into()), None);
    assert_eq!(version, "0.200.0");
    assert_eq!(source, "env");
}

#[test]
fn client_version_reads_codex_models_cache() {
    let tmp = tempdir().unwrap();
    std::fs::write(
        tmp.path().join("models_cache.json"),
        serde_json::json!({ "client_version": "0.137.0", "models": [] }).to_string(),
    )
    .unwrap();
    let (version, source) =
        resolve_openai_codex_client_version_from(None, Some(tmp.path().to_path_buf()));
    assert_eq!(version, "0.137.0");
    assert_eq!(source, "models_cache");
}

#[test]
fn client_version_models_cache_precedes_version_file() {
    let tmp = tempdir().unwrap();
    std::fs::write(
        tmp.path().join("models_cache.json"),
        serde_json::json!({ "client_version": "0.137.0" }).to_string(),
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("version.json"),
        serde_json::json!({ "latest_version": "0.140.0" }).to_string(),
    )
    .unwrap();
    let (version, source) =
        resolve_openai_codex_client_version_from(None, Some(tmp.path().to_path_buf()));
    assert_eq!(version, "0.137.0");
    assert_eq!(source, "models_cache");
}

#[test]
fn client_version_falls_back_to_version_file() {
    let tmp = tempdir().unwrap();
    std::fs::write(
        tmp.path().join("version.json"),
        serde_json::json!({ "latest_version": "0.140.0" }).to_string(),
    )
    .unwrap();
    let (version, source) =
        resolve_openai_codex_client_version_from(None, Some(tmp.path().to_path_buf()));
    assert_eq!(version, "0.140.0");
    assert_eq!(source, "version_json");
}

#[test]
fn client_version_uses_default_when_files_are_missing() {
    let tmp = tempdir().unwrap();
    let (version, source) =
        resolve_openai_codex_client_version_from(None, Some(tmp.path().to_path_buf()));
    assert_eq!(version, OPENAI_CODEX_DEFAULT_CLIENT_VERSION);
    assert_eq!(source, "default");
}

#[test]
fn user_agent_names_embedding_host() {
    assert_eq!(
        openai_codex_user_agent("OpenHuman", "1.2.3"),
        "codex_cli_rs/0.0.0 (OpenHuman 1.2.3)"
    );
}
