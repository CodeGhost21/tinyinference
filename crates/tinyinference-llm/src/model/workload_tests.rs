use super::*;

#[test]
fn every_tier_has_exactly_one_route_entry() {
    assert_eq!(WORKLOAD_ROUTE_TIERS.len(), WORKLOAD_ROUTES.len());
    for tier in WORKLOAD_ROUTE_TIERS {
        assert_eq!(
            WORKLOAD_ROUTES
                .iter()
                .filter(|route| route.name == *tier)
                .count(),
            1,
            "{tier}"
        );
    }
}

#[test]
fn fallback_chains_lead_with_the_primary_then_same_family_alternates() {
    assert_eq!(
        fallback_chain(ROUTE_CHAT),
        Some(vec![ROUTE_CHAT, ROUTE_BURST])
    );
    assert_eq!(
        fallback_chain(ROUTE_BURST),
        Some(vec![ROUTE_BURST, ROUTE_CHAT])
    );
    assert_eq!(
        fallback_chain(ROUTE_REASONING),
        Some(vec![ROUTE_REASONING, ROUTE_AGENTIC])
    );
    assert_eq!(
        fallback_chain(ROUTE_AGENTIC),
        Some(vec![ROUTE_AGENTIC, ROUTE_REASONING])
    );
    assert_eq!(
        fallback_chain(ROUTE_CODING),
        Some(vec![ROUTE_CODING, ROUTE_AGENTIC])
    );
    assert_eq!(
        fallback_chain(ROUTE_SUMMARIZATION),
        Some(vec![ROUTE_SUMMARIZATION, ROUTE_CHAT])
    );
}

#[test]
fn vision_and_raw_models_are_primary_only() {
    assert_eq!(fallback_chain(ROUTE_VISION), None);
    assert_eq!(fallback_chain("gpt-4o"), None);
}

#[test]
fn only_vision_requires_image_input() {
    let gate = required_capabilities(ROUTE_VISION).expect("vision is gated");
    assert!(gate.image_in);
    assert!(!gate.tool_calling);
    for tier in WORKLOAD_ROUTE_TIERS
        .iter()
        .filter(|tier| **tier != ROUTE_VISION)
    {
        assert_eq!(required_capabilities(tier), None, "{tier}");
    }
    assert_eq!(required_capabilities("gpt-4o"), None);
}

#[test]
fn every_fallback_names_a_known_route_that_is_not_gated() {
    for route in WORKLOAD_ROUTES {
        for fallback in route.fallbacks {
            let target = workload_route(fallback).expect("fallback is a route");
            assert!(!target.requires_image_in, "{} -> {fallback}", route.name);
        }
    }
}
