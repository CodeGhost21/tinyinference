use super::*;

use crate::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Notify;

struct Listing {
    body: Option<serde_json::Value>,
    calls: AtomicUsize,
}
#[async_trait::async_trait]
impl ModelListingFetcher for Listing {
    async fn get_json(&self, _: &str, _: &[(String, String)]) -> Result<serde_json::Value> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.body
            .clone()
            .ok_or_else(|| Error::Catalog("offline".into()))
    }
}
fn listing(body: Option<serde_json::Value>) -> Listing {
    Listing {
        body,
        calls: AtomicUsize::new(0),
    }
}
fn request() -> DiscoveryRequest {
    DiscoveryRequest::new("https://provider.example/v1", "model")
        .with_single_model_probe(false)
        .with_ollama_native(false)
}

#[tokio::test]
async fn context_override_retains_provider_output_cap_without_discovery() {
    let resolver = ContextWindowResolver::new();
    let cache = ModelLimitsCache::default();
    let fetcher = listing(Some(
        serde_json::json!({"data":[{"id":"model","context_length":128000,"max_output_tokens":4096}]}),
    ));
    let request = request();
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
    let resolved = resolver
        .resolve(
            " ROUTE ",
            "MODEL",
            ContextWindowInput {
                config_override: Some(200000),
                discovery: Some(&request),
                ..Default::default()
            },
            &fetcher,
            &cache,
        )
        .await;
    assert_eq!(resolved.window, Some(200000));
    assert_eq!(resolved.source, WindowSource::ConfigOverride);
    assert_eq!(resolved.max_output_tokens, Some(4096));
    assert_eq!(fetcher.calls.load(Ordering::SeqCst), 1);
    assert_eq!(resolver.remembered_window("route", "model"), Some(128000));
    let other = resolver
        .resolve(
            "other",
            "model",
            ContextWindowInput {
                config_override: Some(200000),
                ..Default::default()
            },
            &fetcher,
            &cache,
        )
        .await;
    assert_eq!(other.max_output_tokens, None);
}

#[tokio::test]
async fn cap_only_discovery_survives_failure_without_freezing_fallback_context() {
    for (local, first_static, next_static, expected, source) in [
        (
            false,
            Some(64000),
            Some(256000),
            Some(256000),
            WindowSource::StaticGuess,
        ),
        (true, None, None, Some(16384), WindowSource::LocalProfile),
        (false, None, None, None, WindowSource::Unknown),
    ] {
        let resolver = ContextWindowResolver::new();
        let cache = ModelLimitsCache::default();
        let request = request();
        let first = resolver
            .resolve(
                "route",
                "model",
                ContextWindowInput {
                    discovery: Some(&request),
                    static_window: first_static,
                    has_local_profile: local,
                    local_window: Some(8192),
                    ..Default::default()
                },
                &listing(Some(
                    serde_json::json!({"data":[{"id":"model","max_output_tokens":2048}]}),
                )),
                &cache,
            )
            .await;
        assert_eq!(first.max_output_tokens, Some(2048));
        assert_eq!(resolver.remembered_window("route", "model"), None);
        cache.clear();
        let next = resolver
            .resolve(
                "route",
                "model",
                ContextWindowInput {
                    discovery: Some(&request),
                    static_window: next_static,
                    has_local_profile: local,
                    local_window: Some(16384),
                    ..Default::default()
                },
                &listing(None),
                &cache,
            )
            .await;
        assert_eq!(next.max_output_tokens, Some(2048));
        assert_eq!(next.window, expected);
        assert_eq!(next.source, source);
        assert_eq!(resolver.remembered_window("route", "model"), None);
    }
}

struct HeldListing {
    started: Notify,
    release: Notify,
}
#[async_trait::async_trait]
impl ModelListingFetcher for HeldListing {
    async fn get_json(&self, _: &str, _: &[(String, String)]) -> Result<serde_json::Value> {
        self.started.notify_one();
        self.release.notified().await;
        Ok(
            serde_json::json!({"data":[{"id":"model","context_length":128000,"max_output_tokens":4096},{"id":"sibling","context_length":96000}]}),
        )
    }
}

#[tokio::test]
async fn clear_prevents_pending_discovery_from_repopulating_previous_identity() {
    for replace in [false, true] {
        let resolver = Arc::new(ContextWindowResolver::new());
        let fetcher = Arc::new(HeldListing {
            started: Notify::new(),
            release: Notify::new(),
        });
        let task = {
            let resolver = resolver.clone();
            let fetcher = fetcher.clone();
            tokio::spawn(async move {
                let request = request();
                resolver
                    .resolve(
                        "route",
                        "model",
                        ContextWindowInput {
                            discovery: Some(&request),
                            ..Default::default()
                        },
                        fetcher.as_ref(),
                        &ModelLimitsCache::default(),
                    )
                    .await
            })
        };
        fetcher.started.notified().await;
        resolver.clear();
        if replace {
            let request = request();
            resolver.resolve("route", "model", ContextWindowInput { discovery: Some(&request), ..Default::default() },
            &listing(Some(serde_json::json!({"data":[{"id":"model","context_length":64000,"max_output_tokens":2048}]}))),
            &ModelLimitsCache::default()).await;
        }
        fetcher.release.notify_one();
        assert_eq!(task.await.unwrap().window, Some(128000));
        assert_eq!(
            resolver.remembered_window("route", "model"),
            replace.then_some(64000)
        );
        let next = resolver
            .resolve(
                "route",
                "model",
                ContextWindowInput::default(),
                &listing(None),
                &ModelLimitsCache::default(),
            )
            .await;
        assert_eq!(next.window, replace.then_some(64000));
        assert_eq!(next.max_output_tokens, replace.then_some(2048));
    }
}

#[tokio::test]
async fn clear_prevents_pending_discovery_from_poisoning_reused_cache() {
    for replace_before_completion in [false, true] {
        let resolver = Arc::new(ContextWindowResolver::new());
        let cache = Arc::new(ModelLimitsCache::default());
        let old = Arc::new(HeldListing {
            started: Notify::new(),
            release: Notify::new(),
        });
        let task = {
            let resolver = resolver.clone();
            let cache = cache.clone();
            let old = old.clone();
            tokio::spawn(async move {
                let request = request();
                resolver
                    .resolve(
                        "route",
                        "model",
                        ContextWindowInput {
                            discovery: Some(&request),
                            ..Default::default()
                        },
                        old.as_ref(),
                        cache.as_ref(),
                    )
                    .await
            })
        };
        old.started.notified().await;
        resolver.clear();
        let fresh = listing(Some(
            serde_json::json!({"data":[{"id":"model","context_length":64000,"max_output_tokens":2048}]}),
        ));
        let request = request();
        if replace_before_completion {
            assert_eq!(
                resolver
                    .resolve(
                        "route",
                        "model",
                        ContextWindowInput {
                            discovery: Some(&request),
                            ..Default::default()
                        },
                        &fresh,
                        cache.as_ref()
                    )
                    .await
                    .window,
                Some(64000)
            );
        }
        old.release.notify_one();
        let _ = task.await.unwrap();
        assert!(
            cache
                .get_variant(&request.endpoint, "sibling", &request.cache_variant())
                .discovered
                .is_none(),
            "old listing must not prime another model after identity clearing"
        );
        let result = resolver
            .resolve(
                "route",
                "model",
                ContextWindowInput {
                    discovery: Some(&request),
                    ..Default::default()
                },
                &fresh,
                cache.as_ref(),
            )
            .await;
        assert_eq!(result.window, Some(64000));
        assert_eq!(result.max_output_tokens, Some(2048));
        assert_eq!(fresh.calls.load(Ordering::SeqCst), 1);
        assert_eq!(resolver.remembered_window("route", "model"), Some(64000));
    }
}

#[tokio::test]
async fn remembered_limits_do_not_cross_discovery_identity() {
    let original = request().with_header("Authorization", "Bearer tenant-a");
    let variants = [
        request().with_header("Authorization", "Bearer tenant-b"),
        original
            .clone()
            .with_pinned_providers(vec!["provider-b".into()]),
        DiscoveryRequest::new("https://other.example/v1", "model")
            .with_single_model_probe(false)
            .with_ollama_native(false)
            .with_header("Authorization", "Bearer tenant-a"),
    ];
    for variant in variants {
        let resolver = ContextWindowResolver::new();
        let cache = ModelLimitsCache::default();
        let first = resolver.resolve("route", "model", ContextWindowInput {
            discovery: Some(&original), ..Default::default()
        }, &listing(Some(serde_json::json!({"data":[{"id":"model","context_length":128000,"max_output_tokens":4096}]}))), &cache).await;
        assert_eq!(first.window, Some(128000));
        let offline = listing(None);
        for config_override in [None, Some(32000)] {
            let different = resolver
                .resolve(
                    "route",
                    "model",
                    ContextWindowInput {
                        discovery: Some(&variant),
                        config_override,
                        static_window: Some(16000),
                        ..Default::default()
                    },
                    &offline,
                    &cache,
                )
                .await;
            assert_eq!(different.window, config_override.or(Some(16000)));
            assert_eq!(different.max_output_tokens, None);
        }
        cache.clear();
        let same = resolver
            .resolve(
                " ROUTE ",
                "MODEL",
                ContextWindowInput {
                    discovery: Some(&original),
                    ..Default::default()
                },
                &offline,
                &cache,
            )
            .await;
        assert_eq!(same.window, Some(128000));
        assert_eq!(same.max_output_tokens, Some(4096));
    }
}

#[tokio::test]
async fn first_override_reuses_matching_sibling_cache_cap_without_fetching() {
    let cache = ModelLimitsCache::default();
    let target = request().with_header("Authorization", "Bearer tenant-a");
    let mut sibling = target.clone();
    sibling.model = "sibling".into();
    let priming = listing(Some(serde_json::json!({"data":[
        {"id":"sibling","context_length":96000},
        {"id":"model","context_length":128000,"max_output_tokens":4096}
    ]})));
    super::super::discover::discover_model_limits_with(&priming, &cache, &sibling).await;
    let resolver = ContextWindowResolver::new();
    let offline = listing(None);
    let result = resolver
        .resolve(
            "new-route",
            "model",
            ContextWindowInput {
                config_override: Some(200000),
                discovery: Some(&target),
                ..Default::default()
            },
            &offline,
            &cache,
        )
        .await;
    assert_eq!(result.window, Some(200000));
    assert_eq!(result.source, WindowSource::ConfigOverride);
    assert_eq!(result.max_output_tokens, Some(4096));
    assert_eq!(offline.calls.load(Ordering::SeqCst), 0);
    assert_eq!(resolver.remembered_window("new-route", "model"), None);
    let other = target.with_header("x-tenant", "other");
    let other = resolver
        .resolve(
            "new-route",
            "model",
            ContextWindowInput {
                config_override: Some(200000),
                discovery: Some(&other),
                ..Default::default()
            },
            &offline,
            &cache,
        )
        .await;
    assert_eq!(other.max_output_tokens, None);
    assert_eq!(offline.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn identity_free_lookup_projects_smallest_observed_limits_without_changing_exact_routes() {
    let resolver = ContextWindowResolver::new();
    let cache = ModelLimitsCache::default();
    let first = request();
    let second = request().with_header("x-tenant", "second");
    for (request, window, cap) in [(&first, 128000, 4096), (&second, 64000, 2048)] {
        resolver.resolve("route", "model", ContextWindowInput {
            discovery: Some(request), ..Default::default()
        }, &listing(Some(serde_json::json!({"data":[{"id":"model","context_length":window,"max_output_tokens":cap}]}))), &cache).await;
    }
    assert_eq!(resolver.remembered_window(" ROUTE ", "MODEL"), Some(64000));
    let offline = listing(None);
    let projected = resolver
        .resolve(
            "route",
            "model",
            ContextWindowInput::default(),
            &offline,
            &cache,
        )
        .await;
    assert_eq!(projected.window, Some(64000));
    assert_eq!(projected.max_output_tokens, Some(2048));
    cache.clear();
    let exact = resolver
        .resolve(
            "route",
            "model",
            ContextWindowInput {
                discovery: Some(&first),
                ..Default::default()
            },
            &offline,
            &cache,
        )
        .await;
    assert_eq!(exact.window, Some(128000));
    assert_eq!(exact.max_output_tokens, Some(4096));
    resolver.clear();
    assert_eq!(resolver.remembered_window("route", "model"), None);
}

#[tokio::test]
async fn resolver_debug_omits_discovery_credentials_and_urls() {
    let resolver = ContextWindowResolver::new();
    let request = DiscoveryRequest::new("https://user:password@example.test/v1", "model")
        .with_listing_url("https://listing.test/models?api_key=synthetic-secret")
        .with_single_model_probe(false)
        .with_ollama_native(false);
    resolver.resolve("route", "model", ContextWindowInput {
        discovery: Some(&request), ..Default::default()
    }, &listing(Some(serde_json::json!({"data":[{"id":"model","context_length":128000}]}))), &ModelLimitsCache::default()).await;
    let debug = format!("{resolver:?}");
    for secret in ["password", "synthetic-secret", "example.test", "listing.test"] {
        assert!(!debug.contains(secret), "resolver debug disclosed discovery identity");
    }
}

struct HeldProbe {
    started: Notify,
    release: Notify,
}
#[async_trait::async_trait]
impl ModelListingFetcher for HeldProbe {
    async fn get_json(&self, url: &str, _: &[(String, String)]) -> Result<serde_json::Value> {
        if url.ends_with("/models/model") {
            self.started.notify_one();
            self.release.notified().await;
            Ok(serde_json::json!({"id":"model","context_length":128000,"max_output_tokens":4096}))
        } else {
            Ok(serde_json::json!({"data":[
                {"id":"model","max_output_tokens":4096},
                {"id":"sibling","context_length":96000,"max_output_tokens":4096}
            ]}))
        }
    }
}

#[tokio::test]
async fn clear_during_target_probe_does_not_publish_sibling_limits() {
    let resolver = Arc::new(ContextWindowResolver::new());
    let cache = Arc::new(ModelLimitsCache::default());
    let fetcher = Arc::new(HeldProbe { started: Notify::new(), release: Notify::new() });
    let request = request().with_single_model_probe(true);
    let task = {
        let resolver = resolver.clone();
        let cache = cache.clone();
        let fetcher = fetcher.clone();
        let request = request.clone();
        tokio::spawn(async move {
            resolver.resolve("route", "model", ContextWindowInput {
                discovery: Some(&request), ..Default::default()
            }, fetcher.as_ref(), cache.as_ref()).await
        })
    };
    fetcher.started.notified().await;
    let premature = cache.get_variant(&request.endpoint, "sibling", &request.cache_variant()).discovered.is_some();
    resolver.clear();
    fetcher.release.notify_one();
    task.await.unwrap();
    assert!(!premature, "sibling limits must wait for the final discovery commit");
    assert!(cache.get_variant(&request.endpoint, "sibling", &request.cache_variant()).discovered.is_none());
    let mut sibling = request;
    sibling.model = "sibling".into();
    let fresh = listing(Some(serde_json::json!({"data":[{"id":"sibling","context_length":64000,"max_output_tokens":2048}]})));
    let resolved = resolver.resolve("route", "sibling", ContextWindowInput {
        discovery: Some(&sibling), ..Default::default()
    }, &fresh, cache.as_ref()).await;
    assert_eq!(resolved.window, Some(64000));
    assert_eq!(fresh.calls.load(Ordering::SeqCst), 1);
}
