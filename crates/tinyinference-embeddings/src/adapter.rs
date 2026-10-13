//! Provider-binding seam for hosts.
//!
//! A host owns credentials, rate limits and fallback policy; the model
//! contract is [`EmbeddingModel`]. A host implements the small
//! [`BorrowedEmbedBackend`] (borrowed `&str` batches, `anyhow` errors) and
//! wraps it in [`BorrowedEmbeddingModel`] to obtain an [`EmbeddingModel`].

use std::sync::Arc;

use async_trait::async_trait;

use crate::{EmbeddingModel, Error, Result};

/// Host-side embedding backend: credential resolution and transport only.
#[async_trait]
pub trait BorrowedEmbedBackend: Send + Sync {
    /// Provider name, e.g. `"ollama"`.
    fn name(&self) -> &str;
    /// Stable model identifier within the provider.
    fn model_id(&self) -> &str;
    /// Dimensions of every produced vector (0 when disabled or undiscovered).
    fn dimensions(&self) -> usize;
    /// Embeds a batch of borrowed strings, one vector per input in order.
    ///
    /// # Errors
    /// Transport, authentication and quota failures from the provider.
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// Adapts a [`BorrowedEmbedBackend`] to [`EmbeddingModel`].
///
/// The signature is the trait default, so it routes through
/// [`crate::format_embedding_signature`] and is byte-identical to one computed
/// from configuration.
pub struct BorrowedEmbeddingModel {
    backend: Arc<dyn BorrowedEmbedBackend>,
}

impl BorrowedEmbeddingModel {
    /// Wraps a backend.
    #[must_use]
    pub fn new(backend: Arc<dyn BorrowedEmbedBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl EmbeddingModel for BorrowedEmbeddingModel {
    fn is_enabled(&self) -> bool {
        self.backend.dimensions() > 0
    }

    fn name(&self) -> &str {
        self.backend.name()
    }

    fn model_id(&self) -> &str {
        self.backend.model_id()
    }

    fn dimensions(&self) -> usize {
        self.backend.dimensions()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let borrowed: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.backend
            .embed(&borrowed)
            .await
            .map_err(|error| Error::Embedding(format!("{error:#}")))
    }
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
