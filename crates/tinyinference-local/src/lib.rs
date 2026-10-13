//! Endpoint-only local inference for TinyInference.
//!
//! This crate talks to a local inference runtime that the **user** runs and
//! manages: Ollama, LM Studio, MLX, OMLX, or any OpenAI-compatible server.
//! It resolves the configured endpoint and model IDs, probes the endpoint for
//! reachability, reads the models it already serves, and runs prompts,
//! summaries, vision, embeddings, and chat against it.
//!
//! It deliberately does **not** download, pull, or install models or voice
//! assets, and it never spawns, stops, or locates a runtime binary. Installing
//! a runtime and pulling models is the user's job; this crate only reports
//! whether the configured endpoint is reachable and what it serves.

#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod lm_studio;
pub mod local_runtime;
pub mod model_requirements;
pub mod models;
pub mod ollama;
pub mod process;
pub mod profile;
pub mod provider;
pub mod service;
pub mod status;

pub mod error;

pub use error::{Error, Result};
pub use models::LocalModelConfig;
pub use status::{LocalAiEmbeddingResult, LocalAiSpeechResult, LocalAiStatus, LocalAiTtsResult};
