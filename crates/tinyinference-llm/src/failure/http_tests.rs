use super::*;

#[test]
fn status_gates_pin_budget_local_policy_credits_and_moderation_facts() {
    type Case<'a> = (&'a str, u16, fn(ProviderHttpFailureFacts) -> bool);
    let cases: &[Case<'_>] = &[
        ("Insufficient budget", 400, |facts| facts.budget_exhausted),
        (
            "No models loaded. Please load a model in the developer page or via lms load.",
            400,
            |facts| facts.no_model_loaded,
        ),
        ("currently only available for Coding Agents", 403, |facts| {
            facts.access_policy_denied
        }),
        (
            "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 49732.",
            402,
            |facts| facts.insufficient_credits,
        ),
        (
            "{\"error\":\"Message rejected by Ombudsman\",\"score\":80}",
            400,
            |facts| facts.moderation_rejected,
        ),
    ];
    for (body, expected_status, select) in cases {
        for status in [400, 401, 402, 403, 404, 422, 429, 500, 503] {
            assert_eq!(
                select(classify_http_failure("configured-provider", status, body)),
                status == *expected_status,
                "body={body} status={status}"
            );
        }
    }
}

#[test]
fn api_key_rejection_remains_distinct_from_oauth_expiry_and_unknown_users() {
    let oauth = r#"{"error":{"message":"Provided authentication token is expired. Please try signing in again.","type":null,"code":"token_expired","param":null}}"#;
    let revoked = r#"{"error":{"message":"User not found.","code":401}}"#;
    for status in [400, 401, 403, 500] {
        let facts = classify_http_failure("openai", status, oauth);
        assert_eq!(facts.oauth_session_expired, status == 401);
        assert!(!facts.api_key_rejected);
        let facts = classify_http_failure("openai", status, "invalid_api_key");
        assert_eq!(facts.api_key_rejected, matches!(status, 401 | 403));
        assert!(!facts.oauth_session_expired);
        assert_eq!(
            classify_http_failure("openrouter", status, revoked).api_key_rejected,
            matches!(status, 401 | 403)
        );
        assert!(!classify_http_failure("other-provider", status, revoked).api_key_rejected);
    }
}

#[test]
fn hosted_ollama_error_is_provider_and_status_scoped() {
    let body = r#"{"error":"Internal Server Error (ref: df512dcb-d915-493b-8f2d-e8d3dfa640c1)"}"#;
    assert!(classify_http_failure("ollama", 500, body).ollama_cloud_internal_error);
    assert!(!classify_http_failure("other-provider", 500, body).ollama_cloud_internal_error);
    for status in [400, 503] {
        assert!(!classify_http_failure("ollama", status, body).ollama_cloud_internal_error);
    }
    assert!(
        !classify_http_failure(
            "ollama",
            500,
            r#"{"error":"llama runner process has terminated: exit status 0xc0000409"}"#
        )
        .ollama_cloud_internal_error
    );
}

#[test]
fn custom_upstream_envelope_requires_both_provider_and_status() {
    let body = r#"{"error":{"message":"Bad request to upstream provider","type":"upstream_error","status":400}}"#;
    assert!(classify_http_failure("custom_openai", 400, body).custom_upstream_bad_request);
    assert!(!classify_http_failure("other-provider", 400, body).custom_upstream_bad_request);
    assert!(!classify_http_failure("custom_openai", 500, body).custom_upstream_bad_request);
}

#[test]
fn context_and_wrapped_quota_are_status_agnostic() {
    let quota = "kiro API error (500 Internal Server Error): {\"error\":{\"message\":\"HTTP 402 from Kiro IDE: {\\\"message\\\":\\\"You have reached the limit.\\\",\\\"reason\\\":\\\"MONTHLY_REQUEST_COUNT\\\"}\",\"type\":\"server_error\"}}";
    for status in [400, 402, 429, 500] {
        assert!(classify_http_failure("kiro", status, quota).quota_exhausted);
        assert!(
            classify_http_failure("proxy", status, "maximum context length is 8192 tokens")
                .context_window_exceeded
        );
    }
    assert!(!classify_http_failure("kiro", 500, quota).insufficient_credits);
}

#[test]
fn unknown_model_fact_is_separate_from_other_configuration_rejections() {
    for provider in ["managed-provider", "configured-provider"] {
        for status in [400, 403, 404, 422] {
            let facts = classify_http_failure(
                provider,
                status,
                "Model 'example' is not available. Use GET /openai/v1/models to list available models.",
            );
            assert!(facts.configuration_rejected);
            assert!(facts.unknown_model);
            let facts = classify_http_failure(provider, status, "invalid temperature");
            assert!(facts.configuration_rejected);
            assert!(!facts.unknown_model);
        }
        assert!(
            !classify_http_failure(provider, 500, "invalid temperature").configuration_rejected
        );
    }
}

#[test]
fn facts_can_overlap_without_choosing_host_precedence() {
    let facts = classify_http_failure("openai", 401, "invalid_api_key MONTHLY_REQUEST_COUNT");
    assert!(facts.api_key_rejected);
    assert!(facts.quota_exhausted);
}

#[test]
fn reporter_observes_once_and_never_receives_the_original_body() {
    let reports = std::sync::Mutex::new(Vec::new());
    let reporter = |failure: ProviderHttpFailure<'_>| {
        reports.lock().unwrap().push(format!("{failure:?}"));
        assert_eq!(failure.provider, "openai");
        assert_eq!(failure.operation, "completion");
        assert_eq!(failure.model, Some("test-model"));
        assert_eq!(failure.status, 401);
        assert!(failure.facts.api_key_rejected);
    };
    let facts = classify_and_report_http_failure(
        "openai",
        "completion",
        Some("test-model"),
        401,
        "invalid_api_key secret-credential",
        &reporter,
    );
    assert!(facts.api_key_rejected);
    let reports = reports.lock().unwrap();
    assert_eq!(reports.len(), 1);
    assert!(!reports[0].contains("secret-credential"));
}
