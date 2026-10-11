//! Provider-route grammar shared by hosts and provider factories.

/// A provider identifier and its case-sensitive model selection.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderRoute {
    /// Host-selected provider slug; no case normalization is applied.
    pub provider: String,
    /// Model identifier without a successfully parsed temperature suffix.
    pub model: String,
    /// Finite per-route temperature override, when supplied.
    pub temperature: Option<f64>,
}

impl ProviderRoute {
    /// Parses `<provider>:<model>[@<temperature>]`.
    ///
    /// Splits at the first colon so model namespaces and tags survive. Routes
    /// without a colon are left to host sentinel/default handling. Empty
    /// models are retained for the provider builder to validate.
    #[must_use]
    pub fn parse(route: &str) -> Option<Self> {
        let (provider, model) = route.trim().split_once(':')?;
        let (model, temperature) = split_model_and_temperature(model);
        Some(Self {
            provider: provider.trim().to_owned(),
            model,
            temperature,
        })
    }
}

/// Parse a `<model>[@<temp>]` tail into `(model, override)`.
///
/// Tolerates whitespace around the components. Returns `temperature = None`
/// when the suffix is absent or unparseable — the model text is taken as-is.
pub fn split_model_and_temperature(raw: &str) -> (String, Option<f64>) {
    let trimmed = raw.trim();
    if let Some(at_pos) = trimmed.rfind('@') {
        let head = trimmed[..at_pos].trim();
        let tail = trimmed[at_pos + 1..].trim();
        if !head.is_empty()
            && let Ok(parsed) = tail.parse::<f64>()
            && parsed.is_finite()
        {
            return (head.to_string(), Some(parsed));
        }
    }
    (trimmed.to_string(), None)
}

#[cfg(test)]
#[path = "route_tests.rs"]
mod tests;
