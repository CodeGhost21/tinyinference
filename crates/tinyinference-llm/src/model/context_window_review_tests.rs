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
            serde_json::json!({"data":[{"id":"model","context_length":128000,"max_output_tokens":4096}]}),
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
