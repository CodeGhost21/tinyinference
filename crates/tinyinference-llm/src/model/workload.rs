//! Workload-route inventory: the role aliases a host routes models by, and the
//! declarative same-family fallback chains and capability gates between them.
//!
//! This is plain data plus pure lookups. A host with a registry-style router
//! builds it from [`WORKLOAD_ROUTES`]; a host without one can use
//! [`fallback_chain`] and [`required_capabilities`] directly.

use super::CapabilitySet;

/// Light conversational workload.
pub const ROUTE_CHAT: &str = "hint:chat";
/// Heavy reasoning workload.
pub const ROUTE_REASONING: &str = "hint:reasoning";
/// Tool-heavy agentic workload.
pub const ROUTE_AGENTIC: &str = "hint:agentic";
/// Coding workload.
pub const ROUTE_CODING: &str = "hint:coding";
/// Fast, cheap burst workload.
pub const ROUTE_BURST: &str = "hint:burst";
/// Summarization workload.
pub const ROUTE_SUMMARIZATION: &str = "hint:summarization";
/// Image-input workload.
pub const ROUTE_VISION: &str = "hint:vision";

/// Every workload route alias, in canonical order. `memory` is intentionally
/// absent: it rides the chat route rather than being a distinct entry.
pub const WORKLOAD_ROUTE_TIERS: &[&str] = &[
    ROUTE_CHAT,
    ROUTE_REASONING,
    ROUTE_AGENTIC,
    ROUTE_CODING,
    ROUTE_BURST,
    ROUTE_SUMMARIZATION,
    ROUTE_VISION,
];

/// One workload route: its alias, ordered fallbacks, and whether it is gated on
/// image input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkloadRouteSpec {
    /// Route alias, also the default target name.
    pub name: &'static str,
    /// Same-family alternates tried in order after `name` fails.
    pub fallbacks: &'static [&'static str],
    /// Whether the route requires `image_in`; such a route is primary-only.
    pub requires_image_in: bool,
}

/// The routing table:
/// - light/fast conversational siblings `chat <-> burst`;
/// - heavy siblings `reasoning <-> agentic`;
/// - `coding -> agentic`;
/// - `summarization -> chat`;
/// - `vision` is `image_in`-gated and primary-only (a text fallback cannot
///   satisfy the gate).
pub const WORKLOAD_ROUTES: &[WorkloadRouteSpec] = &[
    WorkloadRouteSpec { name: ROUTE_CHAT, fallbacks: &[ROUTE_BURST], requires_image_in: false },
    WorkloadRouteSpec { name: ROUTE_BURST, fallbacks: &[ROUTE_CHAT], requires_image_in: false },
    WorkloadRouteSpec {
        name: ROUTE_REASONING,
        fallbacks: &[ROUTE_AGENTIC],
        requires_image_in: false,
    },
    WorkloadRouteSpec {
        name: ROUTE_AGENTIC,
        fallbacks: &[ROUTE_REASONING],
        requires_image_in: false,
    },
    WorkloadRouteSpec { name: ROUTE_CODING, fallbacks: &[ROUTE_AGENTIC], requires_image_in: false },
    WorkloadRouteSpec {
        name: ROUTE_SUMMARIZATION,
        fallbacks: &[ROUTE_CHAT],
        requires_image_in: false,
    },
    WorkloadRouteSpec { name: ROUTE_VISION, fallbacks: &[], requires_image_in: true },
];

/// Looks up a route by alias.
#[must_use]
pub fn workload_route(name: &str) -> Option<&'static WorkloadRouteSpec> {
    WORKLOAD_ROUTES.iter().find(|route| route.name == name)
}

/// The `[primary, alternate...]` chain for `model`, or `None` when the model is
/// not a workload route or has no alternate (the turn is primary-only).
#[must_use]
pub fn fallback_chain(model: &str) -> Option<Vec<&'static str>> {
    let route = workload_route(model)?;
    if route.fallbacks.is_empty() {
        return None;
    }
    let mut chain = vec![route.name];
    chain.extend_from_slice(route.fallbacks);
    Some(chain)
}

/// The capability gate `model` imposes on every call, or `None` for routes
/// (and raw model strings) with no requirement.
#[must_use]
pub fn required_capabilities(model: &str) -> Option<CapabilitySet> {
    let route = workload_route(model)?;
    route.requires_image_in.then(|| CapabilitySet {
        image_in: true,
        ..CapabilitySet::default()
    })
}

#[cfg(test)]
#[path = "workload_tests.rs"]
mod tests;
