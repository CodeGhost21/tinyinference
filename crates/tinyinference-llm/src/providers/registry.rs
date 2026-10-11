//! Uniform construction and listing discovery for resolved provider specs.
//!
//! Hosts own saved configuration, routing policy and access gates. This factory
//! consumes their resolved spec and obtains credentials through a callback;
//! it neither loads host configuration nor implements managed gateway policy.

use std::sync::Arc;

use crate::model::ChatModel;
use crate::model::discover::{DiscoveryRequest, pinned_openrouter_providers};
use crate::{Error, Result};

use super::anthropic::{
    AnthropicConfig, build_anthropic_model_with_http, endpoint_is_anthropic_messages,
};
use super::openai::codex::OpenAiCodexRouting;
use super::openai::{AuthStyle, OpenAiConfig, endpoint_is_openrouter};
use super::{ProviderKind, ProviderSpec};

/// Resolves a credential for an already selected provider.
///
/// Implementations can consult a keyring, OAuth profile, or environment. Errors
/// are returned unchanged. Empty credentials are permitted only when the spec
/// does not require an API key.
pub trait CredentialResolver: Send + Sync {
    /// Reads the credential without making an inference request.
    fn resolve(&self, spec: &ProviderSpec) -> Result<String>;
}

impl<F> CredentialResolver for F
where
    F: Fn(&ProviderSpec) -> Result<String> + Send + Sync,
{
    fn resolve(&self, spec: &ProviderSpec) -> Result<String> {
        self(spec)
    }
}

/// Optional adapter settings applied after host policy and route resolution.
///
/// Header/query values can contain secrets; this type deliberately has no
/// Debug implementation. The registry never adds host identity or telemetry.
#[derive(Clone, Default)]
pub struct ProviderBuildOptions<'a> {
    /// Authentication override. Defaults to Anthropic for that family, no
    /// authentication for local families, and Bearer otherwise.
    pub auth_style: Option<AuthStyle>,
    /// Fixed per-route temperature.
    pub temperature_override: Option<f64>,
    /// Model glob patterns that reject temperature.
    pub temperature_unsupported_models: &'a [String],
    /// Native tool capability override; local adapters default to false.
    pub native_tool_calling: Option<bool>,
    /// Vision capability override; local adapters default to false.
    pub vision: Option<bool>,
    /// Whether to fold system messages into the first user message.
    pub merge_system_into_user: bool,
    /// Additional request headers.
    pub extra_headers: &'a [(String, String)],
    /// Additional request query parameters.
    pub extra_query_params: &'a [(String, String)],
    /// Host client identity, when required by an endpoint.
    pub user_agent: Option<&'a str>,
    /// Options merged beneath request-specific options.
    pub default_provider_options: Option<serde_json::Value>,
    /// Optional Ollama context setting.
    pub num_ctx: Option<u32>,
    /// Resolved Codex routing. OAuth endpoints have no discoverable listing.
    /// The host supplies any account/originator headers and client identity.
    pub codex: Option<&'a OpenAiCodexRouting>,
    /// HTTP client carrying host TLS/proxy policy.
    pub http: Option<reqwest::Client>,
}

/// Constructs third-party and local adapters from a host-resolved spec.
///
/// ProviderKind::TinyHumans is treated as an ordinary compatible endpoint:
/// managed credentials, billing and endpoint guards belong to the host's
/// managed gateway extension, which should bypass this factory.
pub struct ProviderRegistry<R> {
    credentials: R,
}

impl<R: CredentialResolver> ProviderRegistry<R> {
    /// Creates a factory using the supplied credential source.
    pub fn new(credentials: R) -> Self {
        Self { credentials }
    }

    /// Builds a model using default adapter settings.
    pub fn build(&self, spec: &ProviderSpec) -> Result<Arc<dyn ChatModel<()>>> {
        self.build_with(spec, ProviderBuildOptions::default())
    }

    /// Builds a model with host-resolved adapter settings.
    ///
    /// First-party Anthropic endpoints use Messages when native tools are not
    /// explicitly disabled. Other endpoints use Chat Completions; Codex OAuth
    /// routing selects Responses and omits max_output_tokens.
    pub fn build_with(
        &self,
        spec: &ProviderSpec,
        options: ProviderBuildOptions<'_>,
    ) -> Result<Arc<dyn ChatModel<()>>> {
        validate(spec)?;
        let key = self.resolve_key(spec)?;
        let endpoint = effective_endpoint(spec, &options);
        let auth = effective_auth(spec, &options);
        if matches!(auth, AuthStyle::Anthropic)
            && options.native_tool_calling != Some(false)
            && endpoint_is_anthropic_messages(endpoint)
        {
            return Ok(build_anthropic_model_with_http(
                AnthropicConfig {
                    endpoint,
                    api_key: &key,
                    model: &spec.model,
                    temperature_override: options.temperature_override,
                    temperature_unsupported_models: options.temperature_unsupported_models,
                    extra_headers: options.extra_headers,
                },
                options.http.clone(),
            ));
        }
        let local = is_local(&spec.kind);
        let oauth = options.codex.is_some_and(|routing| routing.using_oauth);
        let provider_options = options.default_provider_options.clone().or_else(|| {
            if local {
                options
                    .num_ctx
                    .map(|num_ctx| serde_json::json!({"options": {"num_ctx": num_ctx}}))
            } else {
                openrouter_default_provider_options(endpoint)
            }
        });
        Ok(super::openai::config::build_openai_model_with(
            OpenAiConfig {
                provider_name: &spec.provider,
                endpoint,
                api_key: &key,
                auth_style: auth,
                model: &spec.model,
                temperature_unsupported_models: options.temperature_unsupported_models,
                temperature_override: options.temperature_override,
                merge_system_into_user: options.merge_system_into_user,
                extra_headers: options.extra_headers,
                native_tool_calling: options.native_tool_calling.or(local.then_some(false)),
                vision: options.vision.or(local.then_some(false)),
                default_provider_options: provider_options,
                responses_api_primary: oauth,
                responses_omit_max_output_tokens: oauth,
                extra_query_params: options.extra_query_params,
                user_agent: options.user_agent,
                explicit_cache_control: !local && endpoint_is_openrouter(endpoint),
            },
            !local,
            options.http.clone(),
        ))
    }

    /// Builds a listing request with the same endpoint, model and credentials
    /// as construction. Local families and Codex OAuth return None because
    /// their model limits are supplied by local profiles or Codex hints.
    ///
    /// Extra headers are retained; query parameters and user-agent settings
    /// describe inference transport and are not copied into listing requests.
    pub fn discovery_request(
        &self,
        spec: &ProviderSpec,
        options: &ProviderBuildOptions<'_>,
    ) -> Result<Option<DiscoveryRequest>> {
        validate(spec)?;
        if is_local(&spec.kind) || options.codex.is_some_and(|route| route.using_oauth) {
            return Ok(None);
        }
        let key = self.resolve_key(spec)?;
        let endpoint = effective_endpoint(spec, options);
        let mut request = DiscoveryRequest::new(endpoint, &spec.model);
        if !key.trim().is_empty() {
            request = match effective_auth(spec, options) {
                AuthStyle::None => request,
                AuthStyle::Bearer => {
                    request.with_header("Authorization", format!("Bearer {}", key.trim()))
                }
                AuthStyle::XApiKey => request.with_header("x-api-key", key.trim()),
                AuthStyle::Anthropic => request
                    .with_header("x-api-key", key.trim())
                    .with_header("anthropic-version", "2023-06-01"),
                AuthStyle::Custom(name) => request.with_header(name, key.trim()),
            };
        }
        for (name, value) in options.extra_headers {
            request = request.with_header(name, value);
        }
        if let Some(provider_options) = options
            .default_provider_options
            .clone()
            .or_else(|| openrouter_default_provider_options(endpoint))
        {
            request = request.with_pinned_providers(pinned_openrouter_providers(&provider_options));
        }
        Ok(Some(request))
    }

    fn resolve_key(&self, spec: &ProviderSpec) -> Result<String> {
        let key = self.credentials.resolve(spec)?;
        if spec.requires_api_key && key.trim().is_empty() {
            return Err(Error::Validation(format!(
                "provider '{}' requires an API key",
                spec.provider
            )));
        }
        Ok(key)
    }
}

/// Stable price ordering for direct OpenRouter routes; request-specific
/// options can override it. Other endpoints receive no default routing option.
pub fn openrouter_default_provider_options(endpoint: &str) -> Option<serde_json::Value> {
    endpoint_is_openrouter(endpoint).then(|| serde_json::json!({"provider": {"sort": "price"}}))
}

fn validate(spec: &ProviderSpec) -> Result<()> {
    if spec.model.trim().is_empty() {
        return Err(Error::Validation("provider model must not be empty".into()));
    }
    if spec.base_url.trim().is_empty() {
        return Err(Error::Validation(
            "provider endpoint must not be empty".into(),
        ));
    }
    Ok(())
}

fn is_local(kind: &ProviderKind) -> bool {
    matches!(
        kind,
        ProviderKind::Ollama | ProviderKind::LmStudio | ProviderKind::LlamaCpp | ProviderKind::Vllm
    )
}

fn effective_endpoint<'a>(
    spec: &'a ProviderSpec,
    options: &'a ProviderBuildOptions<'_>,
) -> &'a str {
    options
        .codex
        .map_or(&spec.base_url, |route| &route.endpoint)
}

fn effective_auth(spec: &ProviderSpec, options: &ProviderBuildOptions<'_>) -> AuthStyle {
    options.auth_style.clone().unwrap_or_else(|| {
        if is_local(&spec.kind) {
            AuthStyle::None
        } else if spec.kind == ProviderKind::Anthropic {
            AuthStyle::Anthropic
        } else {
            AuthStyle::Bearer
        }
    })
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
