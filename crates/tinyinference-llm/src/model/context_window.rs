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
/// Provider-sourced output caps are remembered independently of context windows.
/// Only provider-sourced and learned windows are remembered. Explicit host
/// overrides and fallback guesses are evaluated anew on every call. Keys are
/// trimmed and ASCII-folded, matching the existing model-limit cache and the
/// host migration's synchronous lookup semantics.
#[derive(Debug, Default)]
pub struct ContextWindowResolver {
    state: Mutex<ResolverState>,
}

#[derive(Debug, Default)]
struct ResolverState {
    generation: u64,
    remembered: HashMap<(String, String), RememberedLimits>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RememberedLimits {
    window: Option<(u64, WindowSource)>,
    max_output_tokens: Option<u64>,
}

impl ContextWindowResolver {
    /// Creates empty state. The host owns the resolver's runtime lifetime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the last provider-sourced window resolved for a route/model.
    pub fn remembered_window(&self, provider: &str, model: &str) -> Option<u64> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remembered
            .get(&key(provider, model))
            .and_then(|resolved| resolved.window.map(|(window, _)| window))
    }

    /// Clears remembered limits when the host replaces runtime identity.
    /// Pending discovery may finish, but cannot repopulate the newer identity.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.generation = state.generation.wrapping_add(1);
        state.remembered.clear();
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
        let route_key = key(provider, model);
        let (generation, remembered_cap) = {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            (
                state.generation,
                state
                    .remembered
                    .get(&route_key)
                    .and_then(|limits| limits.max_output_tokens),
            )
        };
        if let Some(window) = input.config_override.filter(|window| *window > 0) {
            return ResolvedWindow {
                window: Some(window),
                source: WindowSource::ConfigOverride,
                max_output_tokens: remembered_cap,
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
        let discovered_window = limits.as_ref().and_then(|limits| {
            limits
                .context_window
                .filter(|window| *window > 0)
                .map(|window| {
                    let source = match limits.source {
                        LimitSource::LearnedFromOverflow => WindowSource::LearnedFromOverflow,
                        _ => WindowSource::ProviderReported,
                    };
                    (window, source)
                })
        });
        let remembered = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.generation == generation {
                if discovered_window.is_some() || max_output_tokens.is_some() {
                    let entry = state.remembered.entry(route_key.clone()).or_default();
                    entry.window = discovered_window.or(entry.window);
                    entry.max_output_tokens = max_output_tokens.or(entry.max_output_tokens);
                }
                state.remembered.get(&route_key).copied()
            } else {
                None
            }
        };
        let max_output_tokens =
            max_output_tokens.or(remembered.and_then(|limits| limits.max_output_tokens));
        if let Some((window, source)) =
            discovered_window.or(remembered.and_then(|limits| limits.window))
        {
            return ResolvedWindow {
                window: Some(window),
                source,
                max_output_tokens,
            };
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

#[cfg(test)]
#[path = "context_window_review_tests.rs"]
mod review_tests;
