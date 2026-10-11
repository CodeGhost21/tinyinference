//! Context-window precedence and remembered provider-sourced windows.
//!
//! Hosts supply configuration overrides, a resolved discovery request, local
//! profile fallback and static hints. This module neither selects providers nor
//! reads configuration, credentials or platform pricing catalogs.

use std::collections::HashMap;
use std::sync::Mutex;

use super::discover::{
    DiscoveryRequest, LimitSource, ModelLimitsCache, ModelListingFetcher,
    discover_model_limits_with,
};

/// Where a resolved context window came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSource {
    /// An explicit host configuration override.
    ConfigOverride,
    /// Provider listing or pinned endpoint metadata.
    ProviderReported,
    /// A smaller window learned from a provider overflow response.
    LearnedFromOverflow,
    /// Local runtime profile fallback.
    LocalProfile,
    /// A host-supplied static model hint.
    StaticGuess,
    /// No usable window was known.
    Unknown,
}

impl WindowSource {
    /// Stable label for host diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::ConfigOverride => "config_override",
            Self::ProviderReported => "provider_reported",
            Self::LearnedFromOverflow => "learned_from_overflow",
            Self::LocalProfile => "local_profile",
            Self::StaticGuess => "static_guess",
            Self::Unknown => "unknown",
        }
    }
}

/// A context window and its provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedWindow {
    /// Token window, or None when unknown.
    pub window: Option<u64>,
    /// Source selected by precedence.
    pub source: WindowSource,
    /// Positive provider-reported output ceiling, including partial discovery.
    pub max_output_tokens: Option<u64>,
}

/// Inputs already resolved by the host's routing/configuration adapter.
#[derive(Clone, Debug, Default)]
pub struct ContextWindowInput<'a> {
    /// A non-zero explicit override takes precedence without discovery.
    pub config_override: Option<u64>,
    /// Listing request for the endpoint/model actually used by inference.
    pub discovery: Option<&'a DiscoveryRequest>,
    /// Static model estimate, including any host-specific tier/catalog hints.
    pub static_window: Option<u64>,
    /// Whether a local runtime profile was selected. A local profile can
    /// remain unknown, distinct from an unknown external model.
    pub has_local_profile: bool,
    /// Local fallback used only when no static estimate is known.
    pub local_window: Option<u64>,
}

/// Instance-owned route/model window state and precedence resolution.
///
/// Only provider-sourced and learned windows are remembered. Explicit host
/// overrides and fallback guesses are evaluated anew on every call. Keys are
/// trimmed and ASCII-folded, matching the existing model-limit cache and the
/// host migration's synchronous lookup semantics.
#[derive(Debug, Default)]
pub struct ContextWindowResolver {
    remembered: Mutex<HashMap<(String, String), ResolvedWindow>>,
}

impl ContextWindowResolver {
    /// Creates empty state. The host owns the resolver's runtime lifetime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the last provider-sourced window resolved for a route/model.
    pub fn remembered_window(&self, provider: &str, model: &str) -> Option<u64> {
        self.remembered
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&key(provider, model))
            .and_then(|resolved| resolved.window)
    }

    /// Clears remembered windows when the host replaces runtime identity.
    pub fn clear(&self) {
        self.remembered
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }

    /// Resolves override > discovered/learned > remembered > static/local > unknown.
    ///
    /// Discovery failures retain remembered provenance and output ceilings.
    /// Partial discovery can supply an output ceiling independently of the window.
    /// Discovery bounds/caching come from the request and `ModelLimitsCache`. No diagnostics
    /// or telemetry are emitted: hosts use the returned source label.
    pub async fn resolve(
        &self,
        provider: &str,
        model: &str,
        input: ContextWindowInput<'_>,
        fetcher: &dyn ModelListingFetcher,
        cache: &ModelLimitsCache,
    ) -> ResolvedWindow {
        if let Some(window) = input.config_override.filter(|window| *window > 0) {
            return ResolvedWindow {
                window: Some(window),
                source: WindowSource::ConfigOverride,
                max_output_tokens: None,
            };
        }
        let limits = if let Some(request) = input.discovery {
            discover_model_limits_with(fetcher, cache, request).await
        } else {
            None
        };
        let max_output_tokens = limits
            .as_ref()
            .and_then(|limits| limits.max_output_tokens)
            .filter(|tokens| *tokens > 0);
        if let Some(limits) = limits
            && let Some(window) = limits.context_window.filter(|window| *window > 0)
        {
            let source = match limits.source {
                LimitSource::LearnedFromOverflow => WindowSource::LearnedFromOverflow,
                _ => WindowSource::ProviderReported,
            };
            let mut remembered = self
                .remembered
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let route_key = key(provider, model);
            let remembered_cap = remembered
                .get(&route_key)
                .and_then(|resolved| resolved.max_output_tokens);
            let resolved = ResolvedWindow {
                window: Some(window),
                source,
                max_output_tokens: max_output_tokens.or(remembered_cap),
            };
            remembered.insert(route_key, resolved);
            return resolved;
        }
        {
            let mut remembered = self
                .remembered
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(resolved) = remembered.get_mut(&key(provider, model)) {
                resolved.max_output_tokens = max_output_tokens.or(resolved.max_output_tokens);
                return *resolved;
            }
        }
        let (window, source) =
            if let Some(window) = input.static_window.filter(|window| *window > 0) {
                (Some(window), WindowSource::StaticGuess)
            } else if input.has_local_profile {
                (
                    input.local_window.filter(|window| *window > 0),
                    WindowSource::LocalProfile,
                )
            } else {
                (None, WindowSource::Unknown)
            };
        ResolvedWindow {
            window,
            source,
            max_output_tokens,
        }
    }
}

fn key(provider: &str, model: &str) -> (String, String) {
    (
        provider.trim().to_ascii_lowercase(),
        model.trim().to_ascii_lowercase(),
    )
}

#[cfg(test)]
#[path = "context_window_tests.rs"]
mod tests;
