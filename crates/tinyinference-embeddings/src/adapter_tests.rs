use super::*;

struct Fixed;

#[async_trait]
impl BorrowedEmbedBackend for Fixed {
    fn name(&self) -> &str {
        "ollama"
    }
    fn model_id(&self) -> &str {
        "nomic-embed-text"
    }
    fn dimensions(&self) -> usize {
        2
    }
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.first() == Some(&"boom") {
            anyhow::bail!("down");
        }
        Ok(texts.iter().map(|t| vec![t.len() as f32, 0.0]).collect())
    }
}

struct Empty;

#[async_trait]
impl EmbeddingModel for Empty {
    fn name(&self) -> &str {
        "empty"
    }
    fn model_id(&self) -> &str {
        "empty"
    }
    fn dimensions(&self) -> usize {
        0
    }
    async fn embed(&self, _: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(Vec::new())
    }
}

#[test]
fn adapter_signature_is_byte_identical_to_the_persisted_form() {
    let model = BorrowedEmbeddingModel::new(Arc::new(Fixed));
    assert_eq!(
        model.signature(),
        "provider=ollama;model=nomic-embed-text;dims=2"
    );
    assert_eq!(
        model.signature(),
        crate::format_embedding_signature("ollama", "nomic-embed-text", 2)
    );
}

#[tokio::test]
async fn adapter_batches_in_order_and_maps_errors() {
    let model = BorrowedEmbeddingModel::new(Arc::new(Fixed));
    let out = model.embed(&["a".into(), "bb".into()]).await.unwrap();
    assert_eq!(out, vec![vec![1.0, 0.0], vec![2.0, 0.0]]);
    assert_eq!(model.embed_one("ccc").await.unwrap(), vec![3.0, 0.0]);
    let err = model.embed(&["boom".into()]).await.unwrap_err();
    assert!(matches!(err, Error::Embedding(m) if m.contains("down")));
}

#[tokio::test]
async fn embed_one_rejects_a_model_that_returns_no_vectors() {
    let err = Empty.embed_one("text").await.unwrap_err();
    assert!(err.to_string().contains("Empty embedding result"));
}

#[tokio::test]
async fn noop_model_embed_one_is_empty_and_signature_pinned() {
    let noop = crate::NoopEmbeddingModel;
    assert_eq!(noop.signature(), "provider=none;model=none;dims=0");
    assert!(noop.embed_one("a").await.unwrap().is_empty());
}
