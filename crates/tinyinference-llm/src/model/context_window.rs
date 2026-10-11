//! Context-window precedence and remembered provider-sourced windows.
//!
//! Hosts supply configuration overrides, a resolved discovery request, local
//! profile fallback and static hints. This module neither selects providers nor
//! reads configuration, credentials or platform pricing catalogs.

use std::collections::HashMap;
use std::sync::Mutex;

use super::discover::{
    DiscoveryRequest, LimitSource, ModelLimitsCache, ModelListingFetcher,
    discover_model_limits_with_guard,
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

/// Instance-owned route/model and discovery-identity precedence resolution.
///
/// Provider-sourced output caps are remembered independently of context windows.
/// Only provider-sourced and learned windows are remembered. Explicit host
/// overrides and fallback guesses are evaluated anew on every call. Keys are
/// trimmed and ASCII-folded. A discovery identity additionally includes its
/// endpoint, target model and cache variant (listing/probe/provider/credential
/// scope), preventing fallback across different resolved requests. Debug output
/// reports counts only, omitting discovery identities that may contain secrets.
#[derive(Default)]
pub struct ContextWindowResolver {
    state: Mutex<ResolverState>,
}

#[derive(Debug, Default)]
struct ResolverState {
    generation: u64,
    remembered: HashMap<RememberedKey, RememberedLimits>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RememberedKey {
    route: (String, String),
    discovery: Option<(String, String, String)>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RememberedLimits {
    window: Option<(u64, WindowSource)>,
    max_output_tokens: Option<u64>,
}

impl std::fmt::Debug for ContextWindowResolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        formatter
            .debug_struct("ContextWindowResolver")
            .field("generation", &state.generation)
            .field("remembered_count", &state.remembered.len())
            .finish_non_exhaustive()
    }
}

impl ContextWindowResolver {
    /// Creates empty state. The host owns the resolver's runtime lifetime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the smallest remembered provider-sourced window for a route/model.
    ///
    /// This backward-compatible identity-free projection is conservative across
    /// observed discovery variants. It does not select an endpoint or credential;
    /// [`Self::resolve`] with discovery uses only that exact request identity.
    pub fn remembered_window(&self, provider: &str, model: &str) -> Option<u64> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .projection(&key(provider, model))
            .window
            .map(|(window, _)| window)
    }

    /// Clears remembered limits when the host replaces runtime identity.
    /// Pending discovery may finish, but cannot repopulate the newer identity
    /// or publish stale results into the caller's shared discovery cache.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.generation = state.generation.wrapping_add(1);
        state.remembered.clear();
    }

    /// Resolves override > discovered/learned > remembered > static/local > unknown.
    ///
    /// Discovery failures retain provenance and output ceilings only for the
    /// matching endpoint/model/cache variant. Without a discovery request, the
    /// route/model lookup projects the smallest observed window and output cap;
    /// callers requiring exact routed facts must supply the resolved request.
    /// Partial discovery can supply an output ceiling independently of the window.
    /// A positive override performs no network discovery, but preserves a positive
    /// output ceiling already in the matching discovery-cache slot or remembered
    /// state. It never remembers the explicit context override.
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
        let route_key = remembered_key(provider, model, input.discovery);
        let (generation, remembered_cap) = {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            let cached_cap = if input.config_override.is_some_and(|window| window > 0) {
                // Cache-only lookup under the same generation guard as discovery:
                // no fetch and no remembered-window write for explicit overrides.
                input.discovery.and_then(|request| {
                    cache
                        .get_variant(&request.endpoint, &request.model, &request.cache_variant())
                        .effective()
                        .and_then(|limits| limits.max_output_tokens)
                        .filter(|tokens| *tokens > 0)
                })
            } else {
                None
            };
            (
                state.generation,
                cached_cap.or(state.lookup(&route_key).max_output_tokens),
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
            discover_model_limits_with_guard(fetcher, cache, request, &|operation| {
                let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
                if state.generation != generation {
                    return false;
                }
                operation();
                true
            })
            .await
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
                Some(state.lookup(&route_key))
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

impl ResolverState {
    fn lookup(&self, key: &RememberedKey) -> RememberedLimits {
        if key.discovery.is_some() {
            self.remembered.get(key).copied().unwrap_or_default()
        } else {
            self.projection(&key.route)
        }
    }

    fn projection(&self, route: &(String, String)) -> RememberedLimits {
        let mut result = RememberedLimits::default();
        for limits in self
            .remembered
            .iter()
            .filter_map(|(key, limits)| (key.route == *route).then_some(limits))
        {
            if let Some(window) = limits.window
                && result.window.is_none_or(|current| window.0 < current.0)
            {
                result.window = Some(window);
            }
            if let Some(cap) = limits.max_output_tokens {
                result.max_output_tokens =
                    Some(result.max_output_tokens.map_or(cap, |old| old.min(cap)));
            }
        }
        result
    }
}

fn remembered_key(
    provider: &str,
    model: &str,
    discovery: Option<&DiscoveryRequest>,
) -> RememberedKey {
    RememberedKey {
        route: key(provider, model),
        discovery: discovery.map(|request| {
            (
                request
                    .endpoint
                    .trim()
                    .trim_end_matches('/')
                    .to_ascii_lowercase(),
                request
                    .model
                    .trim()
                    .trim_end_matches('/')
                    .to_ascii_lowercase(),
                request.cache_variant(),
            )
        }),
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
