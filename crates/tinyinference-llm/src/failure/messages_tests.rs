use super::*;

#[test]
fn transient_statuses_are_not_reported() {
    for s in [408, 429, 502, 503, 504, 520] {
        assert!(!should_report_provider_http_status(s), "{s}");
    }
    for s in [400, 401, 402, 500] {
        assert!(should_report_provider_http_status(s), "{s}");
    }
}

#[test]
fn ollama_cloud_message_round_trips_through_matcher() {
    for model in [None, Some("minimax-m3:cloud")] {
        let msg = ollama_cloud_internal_500_user_message(model, 500);
        assert!(is_ollama_cloud_internal_500_message(&msg));
        assert!(msg.contains("HTTP 500"));
        assert_eq!(msg.contains("minimax-m3:cloud"), model.is_some());
    }
    assert!(!is_ollama_cloud_internal_500_message("some other error"));
}

#[test]
fn no_model_loaded_guidance_is_actionable() {
    assert!(local_provider_no_model_loaded_user_message().contains("lms load"));
}
