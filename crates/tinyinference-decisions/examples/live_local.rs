//! Live Choice, Noul, and Score smoke test for a user-managed decisions server.
//!
//! Usage: cargo run -p tinyinference-decisions --example live_local --
//! http://127.0.0.1:8000/v1/systemone <served-model-id>
//! Optional bearer authentication comes from LOCAL_DECISIONS_API_KEY.

use std::{collections::BTreeMap, time::Duration};

use serde_json::json;
use tinyinference_decisions::{
    Choice, Client, ClientConfig, EvaluationRequest, Noul, NoulCriteria, Question, Score,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: live_local <exact-endpoint-url> <served-model-id>".into());
    }
    let key = std::env::var("LOCAL_DECISIONS_API_KEY").unwrap_or_default();
    let mut config = ClientConfig::self_hosted(&args[0], key);
    config.timeout = Duration::from_secs(120);
    config.retry.max_retries = 0;
    let client = Client::new(config)?;
    let questions = BTreeMap::from([
        (
            "route".to_owned(),
            Question::Choice(Choice {
                instructions: json!("Which support team should handle this ticket?"),
                criteria: BTreeMap::from([
                    ("billing".to_owned(), Some(json!("Charges and refunds"))),
                    (
                        "technical".to_owned(),
                        Some(json!("Software bugs and outages")),
                    ),
                ]),
            }),
        ),
        (
            "duplicate_charge".to_owned(),
            Question::Noul(Noul {
                instructions: json!("Was the customer charged twice?"),
                criteria: Some(NoulCriteria {
                    r#true: json!("The customer reports a duplicate charge"),
                    r#false: json!("The customer reports only one charge"),
                }),
            }),
        ),
        (
            "severity".to_owned(),
            Question::Score(Score {
                instructions: json!("How severe is the support issue?"),
                criteria: vec![
                    json!("Minor inconvenience with no monetary loss"),
                    json!("Billing error with monetary loss"),
                    json!("Widespread outage affecting all customers"),
                ],
            }),
        ),
    ]);
    for state in [
        json!({"ticket": "My card was charged twice for order A-104. Please refund the duplicate charge."}),
        json!({"ticket": "The app crashes when I open settings. My card was charged only once."}),
    ] {
        let request = EvaluationRequest::new(&args[1], state, questions.clone());
        let result = client.evaluate(&request).await?;
        println!("{}", serde_json::to_string_pretty(&result.response)?);
        println!(
            "attempts={} latency_ms={}",
            result.attempts,
            result.latency.as_millis()
        );
    }
    Ok(())
}
