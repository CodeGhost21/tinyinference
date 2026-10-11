//! HTTP status/provider gates around shared provider-body classifiers.
//!
//! Facts can overlap. Hosts retain their classification precedence, managed
//! backend exclusions, retry policy, product messages and telemetry decisions.
//! No host provider slug is reserved and no reporting backend is linked here.

/// Independently matched facts about a provider HTTP error.
///
/// A managed host can apply its session/error-code handling before consulting
/// these facts. In particular, auth and configuration facts alone do not
/// establish whether a host should emit a user notification or demote telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderHttpFailureFacts {
    /// HTTP 400 with a deterministic provider budget rejection.
    pub budget_exhausted: bool,
    /// HTTP 400 with a local server's no-model-loaded body.
    pub no_model_loaded: bool,
    /// A custom_openai HTTP 400 carrying the known upstream error envelope.
    pub custom_upstream_bad_request: bool,
    /// HTTP 403 with an access-policy rejection.
    pub access_policy_denied: bool,
    /// HTTP 400/403/404/422 with a known model/parameter rejection.
    pub configuration_rejected: bool,
    /// The configuration rejection is the OpenAI-compatible unknown-model
    /// shape, which hosts may surface even for their managed gateway.
    pub unknown_model: bool,
    /// Context overflow, regardless of reported HTTP status.
    pub context_window_exceeded: bool,
    /// Plan quota exhausted, regardless of reported HTTP status.
    pub quota_exhausted: bool,
    /// HTTP 401/403 with API-key rejection markers. OpenRouter's
    /// provider-specific "user not found" envelope is included.
    pub api_key_rejected: bool,
    /// HTTP 401 with OpenAI OAuth token-expiry markers.
    pub oauth_session_expired: bool,
    /// HTTP 402 with a credit/payment rejection.
    pub insufficient_credits: bool,
    /// An ollama HTTP 500 with the hosted internal-error reference envelope.
    pub ollama_cloud_internal_error: bool,
    /// HTTP 400 with an external moderation-verdict body.
    pub moderation_rejected: bool,
}

/// Classifies generic provider errors without making host reporting decisions.
pub fn classify_http_failure(provider: &str, status: u16, body: &str) -> ProviderHttpFailureFacts {
    let config_status = matches!(status, 400 | 403 | 404 | 422);
    ProviderHttpFailureFacts {
        budget_exhausted: status == 400 && tinyinference_providers::is_budget_exhausted_message(body),
        no_model_loaded: status == 400 && super::body_indicates_no_model_loaded(body),
        custom_upstream_bad_request: provider == "custom_openai" && status == 400
            && super::body_indicates_custom_openai_upstream_bad_request(body),
        access_policy_denied: status == 403 && super::body_indicates_provider_access_policy_denied(body),
        configuration_rejected: config_status && tinyinference_providers::is_provider_config_rejection_message(body),
        unknown_model: config_status && tinyinference_providers::is_openai_compatible_unknown_model_message(body),
        context_window_exceeded: super::is_context_window_exceeded_message(body),
        quota_exhausted: super::body_indicates_quota_exhausted(body),
        api_key_rejected: matches!(status, 401 | 403) && (
            super::body_indicates_auth_key_error(body)
            || (provider == "openrouter" && body.to_ascii_lowercase().contains("user not found"))
        ),
        oauth_session_expired: status == 401 && tinyinference_providers::is_openai_oauth_session_expired_message(body),
        insufficient_credits: status == 402 && super::body_indicates_insufficient_credits(body),
        ollama_cloud_internal_error: provider == "ollama" && status == 500
            && super::body_indicates_ollama_cloud_internal_error(body),
        moderation_rejected: status == 400 && super::body_indicates_moderation_rejection(body),
    }
}

/// Body-free classification metadata supplied to a host reporting callback.
///
/// The host must sanitize its original error body separately if it chooses
/// to report it; this metadata intentionally carries no credential or payload.
#[derive(Debug, Clone, Copy)]
pub struct ProviderHttpFailure<'a> {
    /// Provider slug.
    pub provider: &'a str,
    /// Operation label supplied by the caller.
    pub operation: &'a str,
    /// Optional model identifier supplied by the caller.
    pub model: Option<&'a str>,
    /// Reported HTTP status.
    pub status: u16,
    /// Independently matched classifier facts.
    pub facts: ProviderHttpFailureFacts,
}

/// Host callback for telemetry/demotion decisions.
///
/// Implementations choose severity, exclusions and notification delivery.
pub trait ErrorReporter: Send + Sync {
    /// Observes a classified HTTP failure.
    fn report(&self, failure: ProviderHttpFailure<'_>);
}

impl<F> ErrorReporter for F
where
    F: Fn(ProviderHttpFailure<'_>) + Send + Sync,
{
    fn report(&self, failure: ProviderHttpFailure<'_>) {
        self(failure);
    }
}

/// Classifies an error, calls the host reporter once, and returns the facts.
///
/// Classification does not change propagation or retries.
pub fn classify_and_report_http_failure(
    provider: &str,
    operation: &str,
    model: Option<&str>,
    status: u16,
    body: &str,
    reporter: &dyn ErrorReporter,
) -> ProviderHttpFailureFacts {
    let facts = classify_http_failure(provider, status, body);
    reporter.report(ProviderHttpFailure { provider, operation, model, status, facts });
    facts
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
