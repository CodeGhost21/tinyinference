use super::*;
use crate::{Error, Result};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Listing {
    body: Option<serde_json::Value>,
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl ModelListingFetcher for Listing {
    async fn get_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
    ) -> Result<serde_json::Value> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.body
            .clone()
            .ok_or_else(|| Error::Catalog("offline".into()))
    }
}

fn listing(window: Option<u64>) -> Listing {
    Listing {
        body: window.map(|window| {
            serde_json::json!({
                "data": [{"id": "model", "context_length": window, "max_output_tokens": 4096}]
            })
        }),
        calls: AtomicUsize::new(0),
    }
}

#[tokio::test]
async fn explicit_override_precedes_listing_and_is_not_remembered() {
    let resolver = ContextWindowResolver::new();
    let fetcher = listing(Some(128_000));
    let cache = ModelLimitsCache::default();
    let request = DiscoveryRequest::new("https://provider.example/v1", "model");
    let resolved = resolver
        .resolve(
            "route",
            "model",
            ContextWindowInput {
                config_override: Some(200_000),
                discovery: Some(&request),
                static_window: Some(64_000),
                ..Default::default()
            },
            &fetcher,
            &cache,
        )
        .await;
    assert_eq!(resolved.window, Some(200_000));
    assert_eq!(resolved.source, WindowSource::ConfigOverride);
    assert_eq!(fetcher.calls.load(Ordering::SeqCst), 0);
    assert_eq!(resolver.remembered_window("route", "model"), None);
}

#[tokio::test]
async fn discovered_window_precedes_hints_and_is_remembered_for_its_route() {
    let resolver = ContextWindowResolver::new();
    let fetcher = listing(Some(128_000));
    let cache = ModelLimitsCache::default();
    let request = DiscoveryRequest::new("https://provider.example/v1", "model");
    let resolved = resolver
        .resolve(
            " Route ",
            " MODEL ",
            ContextWindowInput {
                config_override: Some(0),
                discovery: Some(&request),
                static_window: Some(64_000),
                has_local_profile: true,
                local_window: Some(8192),
            },
            &fetcher,
            &cache,
        )
        .await;
    assert_eq!(resolved.window, Some(128_000));
    assert_eq!(resolved.source, WindowSource::ProviderReported);
    assert_eq!(resolved.max_output_tokens, Some(4096));
    assert_eq!(resolver.remembered_window("route", "model"), Some(128_000));
    assert_eq!(resolver.remembered_window("other-route", "model"), None);
    assert_eq!(resolver.remembered_window("route", "other-model"), None);
}

#[tokio::test]
async fn learned_overflow_lowers_listing_and_updates_remembered_window() {
    let resolver = ContextWindowResolver::new();
    let fetcher = listing(Some(128_000));
    let cache = ModelLimitsCache::default();
    let request = DiscoveryRequest::new("https://provider.example/v1", "model");
    super::super::discover::record_overflow_error_in(
        &cache,
        &request.endpoint,
        &request.model,
        "This model's maximum context length is 64000 tokens",
    );
    let resolved = resolver
        .resolve(
            "route",
            "model",
            ContextWindowInput {
                discovery: Some(&request),
                ..Default::default()
            },
            &fetcher,
            &cache,
        )
        .await;
    assert_eq!(resolved.window, Some(64_000));
    assert_eq!(resolved.source, WindowSource::LearnedFromOverflow);
    assert_eq!(resolver.remembered_window("route", "model"), Some(64_000));
}

#[tokio::test]
async fn failed_discovery_falls_through_to_static_or_local_profiles_without_remembering() {
    let resolver = ContextWindowResolver::new();
    let fetcher = listing(None);
    let cache = ModelLimitsCache::default();
    let request = DiscoveryRequest::new("https://provider.example/v1", "model")
        .with_single_model_probe(false)
        .with_ollama_native(false);
    for (static_window, expected, source) in [
        (Some(64_000), Some(64_000), WindowSource::StaticGuess),
        (None, Some(8192), WindowSource::LocalProfile),
    ] {
        let resolved = resolver
            .resolve(
                "route",
                "model",
                ContextWindowInput {
                    discovery: Some(&request),
                    static_window,
                    has_local_profile: true,
                    local_window: Some(8192),
                    ..Default::default()
                },
                &fetcher,
                &cache,
            )
            .await;
        assert_eq!(resolved.window, expected);
        assert_eq!(resolved.source, source);
        assert_eq!(resolved.max_output_tokens, None);
        assert_eq!(resolver.remembered_window("route", "model"), None);
    }
    assert_eq!(fetcher.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn unknown_models_and_unknown_local_profiles_preserve_distinct_sources() {
    let resolver = ContextWindowResolver::new();
    let fetcher = listing(None);
    let cache = ModelLimitsCache::default();
    for local in [false, true] {
        let resolved = resolver
            .resolve(
                "route",
                "model",
                ContextWindowInput {
                    has_local_profile: local,
                    ..Default::default()
                },
                &fetcher,
                &cache,
            )
            .await;
        assert_eq!(resolved.window, None);
        assert_eq!(
            resolved.source,
            if local {
                WindowSource::LocalProfile
            } else {
                WindowSource::Unknown
            }
        );
    }
    assert_eq!(fetcher.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn resolver_instances_and_clear_keep_runtime_scopes_independent() {
    let resolver = ContextWindowResolver::new();
    let other = ContextWindowResolver::new();
    let fetcher = listing(Some(128_000));
    let cache = ModelLimitsCache::default();
    let request = DiscoveryRequest::new("https://provider.example/v1", "model");
    resolver
        .resolve(
            "route",
            "model",
            ContextWindowInput {
                discovery: Some(&request),
                ..Default::default()
            },
            &fetcher,
            &cache,
        )
        .await;
    assert_eq!(other.remembered_window("route", "model"), None);
    resolver.clear();
    assert_eq!(resolver.remembered_window("route", "model"), None);
}

#[tokio::test]
async fn partial_discovery_keeps_output_caps_with_static_local_or_unknown_context() {
    let fetcher = Listing {
        body: Some(serde_json::json!({"data":[{"id":"model","max_output_tokens":2048}]})),
        calls: AtomicUsize::new(0),
    };
    let request = DiscoveryRequest::new("https://provider.example/v1", "model")
        .with_single_model_probe(false)
        .with_ollama_native(false);
    for (static_window, local_window, local, window, source) in [
        (
            Some(64000),
            None,
            false,
            Some(64000),
            WindowSource::StaticGuess,
        ),
        (
            None,
            Some(8192),
            true,
            Some(8192),
            WindowSource::LocalProfile,
        ),
        (None, None, false, None, WindowSource::Unknown),
    ] {
        let resolved = ContextWindowResolver::new()
            .resolve(
                "route",
                "model",
                ContextWindowInput {
                    discovery: Some(&request),
                    static_window,
                    local_window,
                    has_local_profile: local,
                    ..Default::default()
                },
                &fetcher,
                &ModelLimitsCache::default(),
            )
            .await;
        assert_eq!(resolved.window, window);
        assert_eq!(resolved.source, source);
        assert_eq!(resolved.max_output_tokens, Some(2048));
    }
}

#[tokio::test]
async fn discovery_failure_retains_remembered_window_provenance_and_output_cap() {
    for learned in [false, true] {
        let resolver = ContextWindowResolver::new();
        let cache = ModelLimitsCache::default();
        let request = DiscoveryRequest::new("https://provider.example/v1", "model")
            .with_single_model_probe(false)
            .with_ollama_native(false);
        if learned {
            cache.insert_learned(&request.endpoint, &request.model, 64000);
        }
        let first = resolver
            .resolve(
                " Route ",
                " MODEL ",
                ContextWindowInput {
                    discovery: Some(&request),
                    ..Default::default()
                },
                &listing(Some(128000)),
                &cache,
            )
            .await;
        cache.clear();
        let next = resolver
            .resolve(
                "route",
                "model",
                ContextWindowInput {
                    discovery: Some(&request),
                    static_window: Some(256000),
                    ..Default::default()
                },
                &listing(None),
                &cache,
            )
            .await;
        assert_eq!(next, first);
    }
}

#[tokio::test]
async fn zero_host_hints_are_unknown_and_do_not_hide_valid_local_context() {
    for (local_window, expected) in [(Some(8192), Some(8192)), (Some(0), None)] {
        let resolved = ContextWindowResolver::new()
            .resolve(
                "route",
                "model",
                ContextWindowInput {
                    config_override: Some(0),
                    static_window: Some(0),
                    local_window,
                    has_local_profile: true,
                    ..Default::default()
                },
                &listing(None),
                &ModelLimitsCache::default(),
            )
            .await;
        assert_eq!(resolved.window, expected);
        assert_eq!(resolved.source, WindowSource::LocalProfile);
    }
}

#[tokio::test]
async fn zero_cached_discovery_is_not_remembered_and_keeps_valid_output_cap() {
    let resolver = ContextWindowResolver::new();
    let cache = ModelLimitsCache::default();
    let request = DiscoveryRequest::new("https://provider.example/v1", "model");
    cache.insert_discovered_variant(
        &request.endpoint,
        &request.model,
        &request.cache_variant(),
        Some(super::super::discover::ModelLimits {
            context_window: Some(0),
            max_output_tokens: Some(2048),
            input_modalities: None,
            source: LimitSource::ProviderListing,
        }),
    );
    let resolved = resolver
        .resolve(
            "route",
            "model",
            ContextWindowInput {
                discovery: Some(&request),
                static_window: Some(64000),
                ..Default::default()
            },
            &listing(None),
            &cache,
        )
        .await;
    assert_eq!(resolved.window, Some(64000));
    assert_eq!(resolved.max_output_tokens, Some(2048));
    assert_eq!(resolver.remembered_window("route", "model"), None);
}
