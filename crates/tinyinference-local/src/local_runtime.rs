//! Pure resolution of a `<runtime>:<model>[@<temp>]` provider string into the
//! endpoint, auth style and model a local runtime (Ollama, LM Studio, MLX,
//! OMLX, local OpenAI-compatible) is called with.
//!
//! No I/O happens here: environment lookups go through the injected
//! [`LocalRuntimeInputs::env`] so hosts and tests control them. Privacy and
//! session gates stay in the host.

use crate::lm_studio::lm_studio_base_url_with_env;
use crate::ollama::ollama_base_url_from_override_with_env;
use crate::profile::{LocalProviderKind, kind_from_provider_string, profile_for_kind};
use tinyinference_providers::route::split_model_and_temperature;

/// How the runtime authenticates requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalRuntimeAuth {
    /// No `Authorization` header.
    None,
    /// `Authorization: Bearer <api_key>`.
    Bearer,
}

/// Host-supplied configuration for resolving a local runtime.
#[derive(Clone, Copy)]
pub struct LocalRuntimeInputs<'a> {
    /// Configured base URL override (`local_ai.base_url`).
    pub base_url: Option<&'a str>,
    /// Configured API key (`local_ai.api_key`).
    pub api_key: Option<&'a str>,
    /// Configured Ollama `num_ctx`.
    pub num_ctx: Option<u32>,
    /// Environment lookup used for the `*_URL` overrides.
    pub env: fn(&str) -> Option<String>,
}

impl std::fmt::Debug for LocalRuntimeInputs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalRuntimeInputs")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.map(|_| "<redacted>"))
            .field("num_ctx", &self.num_ctx)
            .finish_non_exhaustive()
    }
}

/// A fully resolved local runtime call target.
#[derive(Clone, PartialEq)]
pub struct ResolvedLocalRuntime {
    /// Runtime kind.
    pub kind: LocalProviderKind,
    /// OpenAI-compatible endpoint, including `/v1`.
    pub endpoint: String,
    /// API key (empty when none applies).
    pub api_key: String,
    /// Auth style.
    pub auth: LocalRuntimeAuth,
    /// Model id, case preserved, without the `@<temp>` suffix.
    pub model: String,
    /// Fixed temperature override from the `@<temp>` suffix.
    pub temperature: Option<f64>,
    /// `num_ctx` to send (Ollama only).
    pub num_ctx: Option<u32>,
}

impl std::fmt::Debug for ResolvedLocalRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedLocalRuntime")
            .field("kind", &self.kind)
            .field("endpoint", &self.endpoint)
            .field(
                "api_key",
                &if self.api_key.is_empty() {
                    ""
                } else {
                    "<redacted>"
                },
            )
            .field("auth", &self.auth)
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("num_ctx", &self.num_ctx)
            .finish()
    }
}

/// Why a provider string could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalRuntimeError {
    /// The model part is empty; carries the expected `<slug>:<model-id>` form.
    EmptyModel {
        /// Canonical provider string that was given.
        provider: String,
        /// Expected form.
        form: String,
    },
}

impl std::fmt::Display for LocalRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModel { provider, form } => write!(
                f,
                "[chat-factory] provider string '{provider}' has an empty model — use '{form}'"
            ),
        }
    }
}

impl std::error::Error for LocalRuntimeError {}

fn non_empty(v: Option<String>) -> Option<String> {
    v.filter(|s| !s.trim().is_empty())
}

/// Resolve a provider string to a local runtime target.
///
/// Returns `None` when the string does not name a local runtime. Only the
/// provider prefix is canonicalised; the model id keeps its case.
pub fn resolve_local_runtime(
    provider: &str,
    inputs: &LocalRuntimeInputs<'_>,
) -> Option<Result<ResolvedLocalRuntime, LocalRuntimeError>> {
    let kind = kind_from_provider_string(provider)?;
    let raw_model = provider
        .trim()
        .split_once(':')
        .map_or("", |(_, model)| model);
    let slug = kind.as_str();
    let canonical = format!("{slug}:{raw_model}");
    let (model, temperature) = split_model_and_temperature(raw_model);
    if model.is_empty() {
        tracing::debug!(slug, "[local-runtime] empty model in provider string");
        return Some(Err(LocalRuntimeError::EmptyModel {
            provider: canonical,
            form: format!("{slug}:<model-id>"),
        }));
    }

    let keyed = |inputs: &LocalRuntimeInputs<'_>| {
        let key = inputs.api_key.unwrap_or("").to_string();
        let auth = if key.trim().is_empty() {
            LocalRuntimeAuth::None
        } else {
            LocalRuntimeAuth::Bearer
        };
        (key, auth)
    };
    let env_or_config = |env_name: &str, default: &str| {
        non_empty((inputs.env)("OPENHUMAN_LOCAL_INFERENCE_URL"))
            .or_else(|| non_empty((inputs.env)(env_name)))
            .or_else(|| inputs.base_url.map(str::to_owned))
            .unwrap_or_else(|| default.to_string())
    };
    let profile = profile_for_kind(kind);

    let (endpoint, api_key, auth, num_ctx) = match kind {
        LocalProviderKind::Ollama => {
            let base = ollama_base_url_from_override_with_env(inputs.base_url, inputs.env);
            let normalized = base.trim_end_matches('/').trim_end_matches("/v1");
            (
                format!("{normalized}/v1"),
                String::new(),
                LocalRuntimeAuth::None,
                inputs.num_ctx,
            )
        }
        LocalProviderKind::LmStudio => {
            let (k, a) = keyed(inputs);
            (
                lm_studio_base_url_with_env(inputs.base_url, inputs.env),
                k,
                a,
                None,
            )
        }
        LocalProviderKind::Mlx => (
            env_or_config(profile.base_url_env, profile.default_base_url),
            String::new(),
            LocalRuntimeAuth::None,
            None,
        ),
        LocalProviderKind::Omlx | LocalProviderKind::LocalOpenai => {
            let (k, a) = keyed(inputs);
            (
                env_or_config(profile.base_url_env, profile.default_base_url),
                k,
                a,
                None,
            )
        }
    };
    tracing::debug!(slug, "[local-runtime] resolved local runtime endpoint");
    Some(Ok(ResolvedLocalRuntime {
        kind,
        endpoint,
        api_key,
        auth,
        model,
        temperature,
        num_ctx,
    }))
}

#[cfg(test)]
#[path = "local_runtime_tests.rs"]
mod tests;
