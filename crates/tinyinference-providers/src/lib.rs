//! Provider routing, authentication protocols, and error classification.

pub mod auth;
pub mod billing;
pub mod builtin;
pub mod config_rejection;
pub mod oauth;
pub mod route;

pub use auth::is_openai_oauth_session_expired_message;
pub use billing::{BudgetMatch, is_budget_exhausted_message, is_budget_message};
pub use config_rejection::{
    NO_MODEL_CONFIGURED_ANCHOR, is_openai_compatible_unknown_model_message,
    is_provider_config_rejection_message,
};
