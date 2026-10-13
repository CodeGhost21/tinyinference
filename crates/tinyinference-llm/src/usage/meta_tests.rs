//! A standalone `invoke` stays usage-faithful: token breakdowns ride the crate
//! `Usage`; charged USD and the context window ride `ModelResponse.raw`.
use super::*;

fn response_for(report: &ProviderUsageReport) -> ModelResponse {
    ModelResponse {
        usage: Some(report.to_usage()),
        raw: report.raw_meta(),
        ..ModelResponse::assistant("hi")
    }
}

#[test]
fn usage_round_trips_charged_usd_and_all_token_breakdowns() {
    let report = ProviderUsageReport {
        input_tokens: 100,
        output_tokens: 20,
        cached_input_tokens: 40,
        cache_creation_tokens: 10,
        reasoning_tokens: 7,
        charged_amount_usd: 0.0123,
        charge_reported: true,
        cost_is_estimate: false,
        context_window: 128_000,
    };
    let response = response_for(&report);

    let usage = response.usage.expect("usage present");
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 20);
    assert_eq!(usage.cache_read_tokens, 40);
    assert_eq!(usage.cache_creation_tokens, 10);
    assert_eq!(usage.reasoning_tokens, 7);
    assert_eq!(
        usage.charged_amount.map(|amount| amount.micros),
        Some(12_300)
    );
    assert_eq!(usage.context_window_tokens, Some(128_000));

    let recovered = recover_usage(&response).expect("usage info");
    assert_eq!(recovered.usage.input_tokens, 100);
    assert_eq!(recovered.context_window, 128_000);
    assert!((recovered.charged_amount_usd.unwrap() - 0.0123).abs() < 1e-9);
}

#[test]
fn no_billing_metadata_leaves_raw_clean() {
    let report = ProviderUsageReport {
        input_tokens: 5,
        output_tokens: 3,
        ..Default::default()
    };
    let response = response_for(&report);
    assert!(response.raw.is_none());
    let recovered = recover_usage(&response).expect("usage info");
    assert_eq!(recovered.charged_amount_usd, None);
    assert_eq!(recovered.context_window, 0);
    assert_eq!(recovered.usage.input_tokens, 5);
}

#[test]
fn estimated_cost_is_not_stashed_as_a_charge() {
    let report = ProviderUsageReport {
        input_tokens: 1,
        charged_amount_usd: 0.5,
        charge_reported: true,
        cost_is_estimate: true,
        ..Default::default()
    };
    assert!(report.raw_meta().is_none());
}

#[test]
fn provider_neutral_total_cost_metadata_is_recovered() {
    let response = ModelResponse {
        usage: Some(Usage::new(8, 3)),
        raw: Some(serde_json::json!({"total_cost_usd": 0.0042})),
        ..ModelResponse::assistant("done")
    };
    let recovered = recover_usage(&response).expect("usage info");
    assert!((recovered.charged_amount_usd.unwrap() - 0.0042).abs() < 1e-9);
}

#[test]
fn no_usage_recovers_to_none() {
    assert!(recover_usage(&ModelResponse::assistant("hi")).is_none());
}

#[test]
fn merge_preserves_wire_json_and_noops_when_empty() {
    let wire = serde_json::json!({"id": "abc"});
    assert_eq!(
        merge_usage_meta(Some(wire.clone()), None, 0),
        Some(wire.clone())
    );
    let merged = merge_usage_meta(Some(wire), Some(0.25), 8_000).unwrap();
    assert_eq!(merged["id"], "abc");
    assert_eq!(merged[USAGE_META_KEY]["context_window"], 8_000);
    let fresh = merge_usage_meta(Some(serde_json::json!("not an object")), Some(0.1), 0).unwrap();
    assert_eq!(fresh[USAGE_META_KEY]["charged_amount_usd"], 0.1);
}

/// Literal-JSON pin so the key and field names cannot drift, and a payload
/// written by a current release still reconstructs.
#[test]
fn usage_meta_raw_wire_shape_is_stable_and_old_payloads_load() {
    let report = ProviderUsageReport {
        input_tokens: 100,
        output_tokens: 20,
        charged_amount_usd: 0.0123,
        charge_reported: true,
        context_window: 128_000,
        ..Default::default()
    };
    assert_eq!(
        report.raw_meta(),
        Some(serde_json::json!({
            "openhuman_usage_meta": {"charged_amount_usd": 0.0123, "context_window": 128000}
        }))
    );

    let old = ModelResponse {
        usage: Some(Usage::new(7, 2)),
        raw: Some(serde_json::json!({
            "openhuman_usage_meta": {"charged_amount_usd": 0.5, "context_window": 32000}
        })),
        ..ModelResponse::assistant("x")
    };
    let recovered = recover_usage(&old).expect("usage");
    assert_eq!(recovered.charged_amount_usd, Some(0.5));
    assert_eq!(recovered.context_window, 32_000);
    assert_eq!(recovered.usage.input_tokens, 7);
}
