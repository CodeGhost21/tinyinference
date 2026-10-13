//! Host-free helpers for probing a provider's `/models` listing.

/// Whether a provider entry can be probed at `{endpoint}/models`.
///
/// Claude Code is a local CLI provider and a `cli://...` endpoint is a
/// placeholder, not an API base. Any other scheme is a misconfiguration that is
/// left to the request path to surface; an endpoint that does not parse as
/// `scheme://host` is likewise left to the request path.
pub fn endpoint_hosts_models_listing(slug: &str, endpoint: &str) -> bool {
    if slug == "claude-code" {
        return false;
    }
    match reqwest::Url::parse(endpoint.trim()) {
        Ok(url) if url.has_host() => url.scheme() != "cli",
        _ => true,
    }
}

/// Appends `key=value` to a URL's query string, percent-encoding as needed.
/// Falls back to plain concatenation when `url` does not parse.
pub fn append_query_param(url: &str, key: &str, value: &str) -> String {
    if let Ok(mut parsed) = reqwest::Url::parse(url) {
        parsed.query_pairs_mut().append_pair(key, value);
        return parsed.to_string();
    }
    let separator = if url.contains('?') { '&' } else { '?' };
    format!("{url}{separator}{key}={value}")
}

/// Whether a provider (by slug or endpoint host) is OpenRouter, including its
/// subdomains.
pub fn is_openrouter_provider(slug: &str, endpoint: &str) -> bool {
    if slug.eq_ignore_ascii_case("openrouter") {
        return true;
    }
    reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "openrouter.ai" || host.ends_with(".openrouter.ai"))
}

#[cfg(test)]
#[path = "listing_tests.rs"]
mod tests;
