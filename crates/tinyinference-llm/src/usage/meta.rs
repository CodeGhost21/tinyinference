//! Provider billing/context metadata that [`Usage`] cannot carry directly.
//!
//! [`Usage`] expresses every token breakdown, but a standalone `invoke` also
//! needs the provider-charged USD amount and the model's context window to
//! stay usage-faithful. Those ride [`ModelResponse::raw`] under
//! [`USAGE_META_KEY`], written by [`ProviderUsageReport::raw_meta`] or
//! [`merge_usage_meta`] and read back by [`recover_usage`].

use serde::{Deserialize, Serialize};

use crate::model::ModelResponse;
use crate::usage::{ChargedAmount, Usage};

/// JSON key in `ModelResponse.raw` under which the metadata is stashed. It is a
/// wire contract: payloads written under this key by earlier releases must keep
/// loading.
pub const USAGE_META_KEY: &str = "openhuman_usage_meta";

/// Metadata with no [`Usage`] home, ferried through [`ModelResponse::raw`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageMeta {
    /// Provider-charged amount in USD. `None` when no charge was reported;
    /// `Some(0.0)` is a free call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub charged_amount_usd: Option<f64>,
    /// Model context window in tokens (`0` when unknown).
    #[serde(default)]
    pub context_window: u64,
}

/// A host's provider-reported usage, in the shape needed to build a crate
/// [`Usage`] and its `raw` passthrough.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ProviderUsageReport {
    /// Prompt tokens.
    pub input_tokens: u64,
    /// Completion tokens.
    pub output_tokens: u64,
    /// Prompt tokens served from the provider cache.
    pub cached_input_tokens: u64,
    /// Prompt tokens written to the provider cache.
    pub cache_creation_tokens: u64,
    /// Reasoning tokens.
    pub reasoning_tokens: u64,
    /// Charged amount in USD (meaningful with `charge_reported`).
    pub charged_amount_usd: f64,
    /// Whether the provider actually reported a charge.
    pub charge_reported: bool,
    /// Whether `charged_amount_usd` is a catalog estimate, not a provider charge.
    pub cost_is_estimate: bool,
    /// Model context window in tokens (`0` when unknown).
    pub context_window: u64,
}

impl ProviderUsageReport {
    /// Builds the crate [`Usage`], carrying every token breakdown. A finite,
    /// positive charge becomes `charged_amount` in USD micros, but only when the
    /// provider reported it: a catalog estimate is never recorded as a charge.
    #[must_use]
    pub fn to_usage(&self) -> Usage {
        let mut usage = Usage::new(self.input_tokens, self.output_tokens);
        usage.cache_read_tokens = self.cached_input_tokens;
        usage.cache_creation_tokens = self.cache_creation_tokens;
        usage.reasoning_tokens = self.reasoning_tokens;
        if self.charge_reported
            && !self.cost_is_estimate
            && self.charged_amount_usd.is_finite()
            && self.charged_amount_usd > 0.0
        {
            usage.charged_amount = Some(ChargedAmount::usd_micros(
                (self.charged_amount_usd * 1_000_000.0).round() as i64,
            ));
        }
        if self.context_window > 0 {
            usage.context_window_tokens = Some(self.context_window);
        }
        usage
    }

    /// The `ModelResponse.raw` value for this report, or `None` when it carries
    /// neither a provider-reported (non-estimated) charge nor a context window,
    /// so billing-free responses stay `raw`-clean.
    #[must_use]
    pub fn raw_meta(&self) -> Option<serde_json::Value> {
        let charged =
            (self.charge_reported && !self.cost_is_estimate).then_some(self.charged_amount_usd);
        merge_usage_meta(None, charged, self.context_window)
    }
}

/// Merges the metadata into an existing `raw` value under [`USAGE_META_KEY`],
/// preserving the wire JSON in an object `raw`, or creating a fresh object.
/// A no-op (returns `raw` unchanged) when there is nothing to carry.
#[must_use]
pub fn merge_usage_meta(
    raw: Option<serde_json::Value>,
    charged_amount_usd: Option<f64>,
    context_window: u64,
) -> Option<serde_json::Value> {
    if charged_amount_usd.is_none() && context_window == 0 {
        return raw;
    }
    let meta = match serde_json::to_value(UsageMeta {
        charged_amount_usd,
        context_window,
    }) {
        Ok(value) => value,
        Err(_) => return raw,
    };
    match raw {
        Some(serde_json::Value::Object(mut object)) => {
            object.insert(USAGE_META_KEY.to_string(), meta);
            Some(serde_json::Value::Object(object))
        }
        // A non-object (or absent) raw cannot hold the key beside wire fields.
        _ => Some(serde_json::json!({ USAGE_META_KEY: meta })),
    }
}

/// Usage recovered from a [`ModelResponse`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecoveredUsage {
    /// The response's crate usage (token breakdowns).
    pub usage: Usage,
    /// Provider-reported charge in USD, from the stashed metadata, the
    /// provider-neutral `total_cost_usd` / `usage.cost` raw fields, or the
    /// usage's own `charged_amount`, in that order.
    pub charged_amount_usd: Option<f64>,
    /// Model context window in tokens (`0` when unknown).
    pub context_window: u64,
}

/// Recovers usage from `response`, or `None` when it carried no usage.
#[must_use]
pub fn recover_usage(response: &ModelResponse) -> Option<RecoveredUsage> {
    let usage = response.usage?;
    let mut meta = response
        .raw
        .as_ref()
        .and_then(|raw| raw.get(USAGE_META_KEY))
        .and_then(|value| serde_json::from_value::<UsageMeta>(value.clone()).ok())
        .unwrap_or_default();
    if meta.charged_amount_usd.is_none() {
        meta.charged_amount_usd = response
            .raw
            .as_ref()
            .and_then(|raw| raw.get("total_cost_usd"))
            .and_then(serde_json::Value::as_f64)
            .or_else(|| response.raw.as_ref()?.get("usage")?.get("cost")?.as_f64())
            .or_else(|| {
                usage
                    .charged_amount
                    .map(|charge| charge.micros as f64 / 1_000_000.0)
            });
    }
    Some(RecoveredUsage {
        usage,
        charged_amount_usd: meta.charged_amount_usd,
        context_window: meta.context_window,
    })
}

#[cfg(test)]
#[path = "meta_tests.rs"]
mod tests;
