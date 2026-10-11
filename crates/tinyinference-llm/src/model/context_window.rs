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
    /// Provider-reported output ceiling when discovery supplied the window.
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
    remembered: Mutex<HashMap<(String, String), u64>>,
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
            .copied()
    }

    /// Clears remembered windows when the host replaces runtime identity.
    pub fn clear(&self) {
        self.remembered
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }

    /// Resolves override > discovered/learned > static/local > unknown.
    ///
    /// Discovery failures fall through to hints; discovery bounds/caching are
    /// supplied by the existing request and ModelLimitsCache. No diagnostics
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
        if let Some(request) = input.discovery
            && let Some(limits) = discover_model_limits_with(fetcher, cache, request).await
            && let Some(window) = limits.context_window
        {
            let source = match limits.source {
                LimitSource::LearnedFromOverflow => WindowSource::LearnedFromOverflow,
                _ => WindowSource::ProviderReported,
            };
            self.remembered
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .insert(key(provider, model), window);
            return ResolvedWindow {
                window: Some(window),
                source,
                max_output_tokens: limits.max_output_tokens,
            };
        }
        let (window, source) = if let Some(window) = input.static_window {
            (Some(window), WindowSource::StaticGuess)
        } else if input.has_local_profile {
            (input.local_window, WindowSource::LocalProfile)
        } else {
            (None, WindowSource::Unknown)
        };
        ResolvedWindow {
            window,
            source,
            max_output_tokens: None,
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
