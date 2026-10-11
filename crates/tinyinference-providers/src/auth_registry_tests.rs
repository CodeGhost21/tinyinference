use super::*;

fn entry(provider: &str, status: u16, timestamp_ms: u64) -> ProviderAuthError {
    ProviderAuthError {
        provider: provider.into(),
        status,
        message: format!("{provider} rejected the supplied credential"),
        timestamp_ms,
    }
}

#[test]
fn record_returns_true_only_on_first_episode_and_refreshes_status_and_timestamp() {
    let registry = ProviderAuthErrorRegistry::new();
    assert!(registry.record(entry("openrouter", 401, 10)));
    assert!(!registry.record(entry("openrouter", 403, 20)));
    assert!(registry.record(entry("openai", 401, 30)));
    assert_eq!(
        registry.snapshot(),
        vec![entry("openai", 401, 30), entry("openrouter", 403, 20)]
    );
}

#[test]
fn clear_removes_entry_and_rearms_latch() {
    let registry = ProviderAuthErrorRegistry::new();
    assert!(registry.record(entry("openrouter", 401, 10)));
    assert!(registry.clear("openrouter"));
    assert!(!registry.clear("openrouter"));
    assert!(registry.record(entry("openrouter", 401, 20)));
}

#[test]
fn independent_instances_and_clear_all_keep_host_scopes_separate() {
    let left = ProviderAuthErrorRegistry::new();
    let right = ProviderAuthErrorRegistry::new();
    left.record(entry("openrouter", 401, 10));
    assert!(right.snapshot().is_empty());
    assert!(right.record(entry("openrouter", 403, 20)));
    left.clear_all();
    assert!(left.snapshot().is_empty());
    assert_eq!(right.snapshot(), vec![entry("openrouter", 403, 20)]);
}

#[test]
fn snapshot_preserves_host_serialized_shape_and_message() {
    let registry = ProviderAuthErrorRegistry::new();
    let expected = serde_json::json!({
        "provider": "openrouter", "status": 401,
        "message": "Reconnect this provider.", "timestamp_ms": 123
    });
    registry.record(serde_json::from_value(expected.clone()).unwrap());
    assert_eq!(
        serde_json::to_value(registry.snapshot()[0].clone()).unwrap(),
        expected
    );
}

#[test]
fn concurrent_records_open_only_one_episode() {
    let registry = std::sync::Arc::new(ProviderAuthErrorRegistry::new());
    let threads: Vec<_> = (0..16)
        .map(|timestamp| {
            let registry = registry.clone();
            std::thread::spawn(move || registry.record(entry("openrouter", 401, timestamp)))
        })
        .collect();
    let new_episodes = threads
        .into_iter()
        .map(|thread| thread.join().unwrap() as usize)
        .sum::<usize>();
    assert_eq!(new_episodes, 1);
    assert_eq!(registry.snapshot().len(), 1);
}

#[test]
fn poisoned_registry_recovers_existing_episodes() {
    let registry = ProviderAuthErrorRegistry::new();
    registry.record(entry("openrouter", 401, 1));
    let _ = std::panic::catch_unwind(|| {
        let _guard = registry.entries.write().unwrap();
        panic!("simulated caller panic");
    });
    assert!(!registry.record(entry("openrouter", 403, 2)));
    assert_eq!(registry.snapshot(), vec![entry("openrouter", 403, 2)]);
    assert!(registry.clear("openrouter"));
    registry.clear_all();
}
