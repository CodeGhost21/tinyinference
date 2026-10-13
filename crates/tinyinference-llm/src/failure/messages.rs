//! Host-free user guidance and status policy for classified provider failures.

/// HTTP statuses that are transient upstream/caller-side throttling signals.
/// Retry and fallback layers handle them, so a per-attempt telemetry report is
/// noise.
pub const TRANSIENT_PROVIDER_HTTP_STATUSES: &[u16] = &[408, 429, 502, 503, 504, 520];

/// Whether a non-2xx provider status is worth a per-attempt telemetry report.
pub fn should_report_provider_http_status(status: u16) -> bool {
    !TRANSIENT_PROVIDER_HTTP_STATUSES.contains(&status)
}

/// Stable anchor shared by [`ollama_cloud_internal_500_user_message`] and
/// [`is_ollama_cloud_internal_500_message`] so they cannot drift.
const OLLAMA_CLOUD_INTERNAL_500_USER_PREFIX: &str = "Ollama cloud is temporarily unavailable";

/// Actionable guidance for a local inference server with no model loaded.
pub fn local_provider_no_model_loaded_user_message() -> String {
    "Your local inference server (e.g. LM Studio) is running but has no model loaded. \
     Load a model — in LM Studio use the developer page or the `lms load` command — \
     then try again."
        .to_string()
}

/// Message-level half of the Ollama Cloud hosted-inference 500 classifier:
/// matches the user message built by [`ollama_cloud_internal_500_user_message`]
/// when it is re-raised at a higher layer.
pub fn is_ollama_cloud_internal_500_message(message: &str) -> bool {
    message
        .to_ascii_lowercase()
        .contains(&OLLAMA_CLOUD_INTERNAL_500_USER_PREFIX.to_ascii_lowercase())
}

/// Actionable replacement for Ollama Cloud's opaque `Internal Server Error
/// (ref: <uuid>)` body. `model` is included when known.
pub fn ollama_cloud_internal_500_user_message(model: Option<&str>, status: u16) -> String {
    match model {
        Some(model) => format!(
            "{OLLAMA_CLOUD_INTERNAL_500_USER_PREFIX} for model `{model}` (Ollama returned HTTP \
             {status}); the hosted model failed on Ollama's side — retry shortly or switch models."
        ),
        None => format!(
            "{OLLAMA_CLOUD_INTERNAL_500_USER_PREFIX} (Ollama returned HTTP {status}); the hosted \
             model failed on Ollama's side — retry shortly or switch models."
        ),
    }
}

#[cfg(test)]
#[path = "messages_tests.rs"]
mod tests;
