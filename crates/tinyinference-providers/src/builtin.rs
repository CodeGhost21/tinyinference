//! Known cloud endpoints and their protocol capabilities.
//!
//! Host-specific managed providers are deliberately absent. Hosts register
//! their own descriptors alongside this provider-neutral catalog.

/// Authentication protocol for a builtin cloud provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinAuthStyle {
    /// OpenAI-compatible Authorization bearer token.
    Bearer,
    /// Anthropic x-api-key and version headers.
    Anthropic,
}

/// A known cloud-provider preset, independent of any saved host configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinCloudProvider {
    /// Stable provider identifier.
    pub slug: &'static str,
    /// Display name of the provider.
    pub label: &'static str,
    /// Canonical API base URL.
    pub endpoint: &'static str,
    /// Provider authentication protocol.
    pub auth_style: BuiltinAuthStyle,
}

/// Known third-party cloud-provider presets. Managed host routes are extensions.
pub const BUILTIN_CLOUD_PROVIDERS: &[BuiltinCloudProvider] = &[
    BuiltinCloudProvider {
        slug: "openai",
        label: "OpenAI",
        endpoint: "https://api.openai.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "anthropic",
        label: "Anthropic",
        endpoint: "https://api.anthropic.com/v1",
        auth_style: BuiltinAuthStyle::Anthropic,
    },
    BuiltinCloudProvider {
        slug: "openrouter",
        label: "OpenRouter",
        endpoint: "https://openrouter.ai/api/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "orcarouter",
        label: "OrcaRouter",
        endpoint: "https://api.orcarouter.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "gmi",
        label: "GMI",
        endpoint: "https://api.gmi-serving.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "fireworks",
        label: "Fireworks",
        endpoint: "https://api.fireworks.ai/inference/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "moonshot",
        label: "Kimi (Moonshot)",
        endpoint: "https://api.moonshot.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "groq",
        label: "Groq",
        endpoint: "https://api.groq.com/openai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "mistral",
        label: "Mistral",
        endpoint: "https://api.mistral.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "deepseek",
        label: "DeepSeek",
        endpoint: "https://api.deepseek.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "together",
        label: "Together AI",
        endpoint: "https://api.together.xyz/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "google",
        label: "Google Gemini",
        endpoint: "https://generativelanguage.googleapis.com/v1beta/openai",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "cerebras",
        label: "Cerebras",
        endpoint: "https://api.cerebras.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "xai",
        label: "xAI",
        endpoint: "https://api.x.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "huggingface",
        label: "Hugging Face",
        endpoint: "https://router.huggingface.co/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "nvidia",
        label: "NVIDIA",
        endpoint: "https://integrate.api.nvidia.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "zai",
        label: "Z.AI",
        endpoint: "https://api.z.ai/api/paas/v4",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "minimax",
        label: "MiniMax",
        // MiniMax exposes a full OpenAI-compatible surface at `/v1`
        // (`/v1/chat/completions`, `/v1/models`). The previous `/anthropic`
        // base + Anthropic auth pointed at MiniMax's Messages-protocol API,
        // which OpenHuman does not speak — it only builds OpenAI-style
        // `/chat/completions` and `/models` — so both chat and model-listing
        // 404'd (`/anthropic/chat/completions`, `/anthropic/models`). The
        // 404 on model-listing was Sentry TAURI-RUST-8X3. Use the `/v1`
        // OpenAI surface with Bearer auth so both paths resolve.
        endpoint: "https://api.minimax.io/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "stepfun",
        label: "StepFun",
        endpoint: "https://api.stepfun.ai/step_plan/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "kilocode",
        label: "Kilo Code",
        endpoint: "https://api.kilo.ai/api/gateway",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "deepinfra",
        label: "DeepInfra",
        endpoint: "https://api.deepinfra.com/v1/openai",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "novita",
        label: "Novita",
        endpoint: "https://api.novita.ai/v3/openai",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "venice",
        label: "Venice",
        endpoint: "https://api.venice.ai/api/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "vercel-ai-gateway",
        label: "Vercel AI Gateway",
        endpoint: "https://ai-gateway.vercel.sh/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "sumopod",
        label: "SumoPod",
        endpoint: "https://ai.sumopod.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "modelscope",
        label: "ModelScope",
        endpoint: "https://api-inference.modelscope.cn/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
];

/// Finds a builtin preset by its exact provider identifier.
#[must_use]
pub fn builtin_cloud_provider(type_str: &str) -> Option<&'static BuiltinCloudProvider> {
    BUILTIN_CLOUD_PROVIDERS
        .iter()
        .find(|provider| provider.slug == type_str)
}

/// Whether `slug` matches a built-in cloud provider preset.
///
/// The chat factory uses this to decide capability defaults (e.g. whether the
/// provider exposes the OpenAI Responses API) only for providers we ship and
/// therefore know the API surface of. Custom / user-defined slugs are treated
/// as unknown and keep the permissive defaults.
pub fn is_builtin_cloud_slug(slug: &str) -> bool {
    builtin_cloud_provider(slug).is_some()
}

/// Whether a builtin preset supports the OpenAI Responses API.
///
/// Only the first-party OpenAI preset enables the fallback. Unknown proxies
/// retain their caller-selected capability; they are not builtin presets.
pub fn builtin_cloud_supports_responses_api(slug: &str) -> bool {
    matches!(slug, "openai")
}

/// Extract the lowercased authority host from an endpoint URL, dropping the
/// scheme, any userinfo, the port, and the path. Returns `None` when no host
/// can be parsed. Tolerates a missing scheme and bracketed IPv6 literals.
/// Bare IPv6 authorities are unsupported; use `[::1]` or `[::1]:8080`.
pub fn endpoint_host(endpoint: &str) -> Option<String> {
    let s = endpoint.trim();
    // Drop the scheme (`https://…`); tolerate a bare `host/path` form.
    let after_scheme = s.split_once("://").map(|(_, rest)| rest).unwrap_or(s);
    // The authority ends at the first path / query / fragment delimiter.
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    // Strip any `user:pass@` userinfo prefix.
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, host)| host)
        .unwrap_or(authority);
    // Strip the port, handling bracketed IPv6 literals (`[::1]:8080`).
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        rest.split_once(']').map(|(h, _)| h).unwrap_or(rest)
    } else {
        host_port
            .rsplit_once(':')
            .map(|(h, _)| h)
            .unwrap_or(host_port)
    };
    let host = host.trim().to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

/// Whether an authority host belongs to a known inference provider.
///
/// Comparisons are case-insensitive and derived from the builtin catalog.
pub fn host_is_builtin_cloud_provider(host: &str) -> bool {
    let host = host.trim().to_ascii_lowercase();
    if host.is_empty() {
        return false;
    }
    BUILTIN_CLOUD_PROVIDERS
        .iter()
        .any(|p| endpoint_host(p.endpoint).as_deref() == Some(host.as_str()))
}

/// Whether an endpoint **host** is a known cloud host that does NOT serve the
/// OpenAI Responses API (`/v1/responses`) — i.e. it is chat-completions-only,
/// regardless of which user slug points at it.
///
/// Derived entirely from [`BUILTIN_CLOUD_PROVIDERS`]: a host is chat-only when
/// some built-in preset uses it AND no preset at that host advertises the
/// Responses API (only OpenAI's `api.openai.com` does). This closes the
/// custom-slug gap behind the builtin-slug gate
/// ([`builtin_cloud_supports_responses_api`]): a user slug pointed at, e.g.,
/// `integrate.api.nvidia.com` must never attempt `/responses` (TAURI-RUST-5A1),
/// while a genuinely unknown proxy host keeps the permissive fallback so a real
/// OpenAI proxy still gets `/responses`.
pub fn endpoint_host_is_chat_completions_only(endpoint: &str) -> bool {
    let Some(host) = endpoint_host(endpoint) else {
        return false;
    };
    let mut matched_chat_only = false;
    for provider in BUILTIN_CLOUD_PROVIDERS {
        if endpoint_host(provider.endpoint).as_deref() == Some(host.as_str()) {
            if builtin_cloud_supports_responses_api(provider.slug) {
                // A Responses-capable built-in lives at this host → not chat-only.
                return false;
            }
            matched_chat_only = true;
        }
    }
    matched_chat_only
}

#[cfg(test)]
#[path = "builtin_tests.rs"]
mod tests;
