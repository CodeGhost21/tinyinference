use super::*;

#[test]
fn route_preserves_namespaces_tags_and_case_with_a_temperature_suffix() {
    let route = ProviderRoute::parse("  My-Provider : Org/Model:Tag @ 0.7 ").unwrap();
    assert_eq!(route.provider, "My-Provider");
    assert_eq!(route.model, "Org/Model:Tag");
    assert_eq!(route.temperature, Some(0.7));
}

#[test]
fn route_leaves_host_sentinels_and_empty_models_for_host_and_builder() {
    assert!(ProviderRoute::parse("cloud").is_none());
    assert!(ProviderRoute::parse("openhuman").is_none());
    assert_eq!(ProviderRoute::parse("ollama:").unwrap().model, "");
}

#[test]
fn temperature_parser_preserves_absent_invalid_and_non_finite_suffixes() {
    for raw in [
        " model ",
        " model@bad ",
        " model@NaN ",
        " model@inf ",
        " @0.7 ",
    ] {
        assert_eq!(split_model_and_temperature(raw), (raw.trim().into(), None));
    }
}

#[test]
fn temperature_parser_uses_the_last_at_sign_and_trims_components() {
    assert_eq!(
        split_model_and_temperature(" org@revision/model @ -0.5 "),
        ("org@revision/model".into(), Some(-0.5))
    );
    assert_eq!(
        split_model_and_temperature("model@0"),
        ("model".into(), Some(0.0))
    );
}
