//! Portable compatibility shapes for aggregated chat and incremental provider events.
//!
//! These contracts preserve the optional text/reasoning and raw tool-call shapes
//! used by existing provider bridges. They are separate from the normalized
//! [`crate::model::ModelResponse`] and [`crate::message::MessageDelta`]: those
//! carry an assistant message and parsed tool arguments or combined fragments.
//! A caller that must retain raw argument bytes and provider signature metadata
//! supplies its own tool-call type; product billing similarly stays on a caller's
//! usage type. No implicit conversion to the normalized types is provided.

/// An aggregated chat response containing optional text, tool calls and usage.
///
/// `U` defaults to portable token usage; hosts may retain their billing extension.
/// `T` defaults to normalized tool calls; a dialect bridge may instead retain
/// raw provider arguments and signature metadata without parsing or rewriting.
#[derive(Debug, Clone)]
pub struct ChatResponse<U = crate::usage::Usage, T = crate::tool::ToolCall> {
    /// Visible response text. None remains distinct from an empty string.
    pub text: Option<String>,
    /// Tool calls in provider order, with identities preserved by the caller's T.
    pub tool_calls: Vec<T>,
    /// Usage, when reported. Absence is not treated as a zero-token report.
    pub usage: Option<U>,
    /// Raw reasoning content, separate from visible text and retained verbatim
    /// for providers that require it on the next assistant turn.
    pub reasoning_content: Option<String>,
}

impl<U, T> Default for ChatResponse<U, T> {
    fn default() -> Self {
        Self {
            text: None,
            tool_calls: Vec::new(),
            usage: None,
            reasoning_content: None,
        }
    }
}

/// One incremental provider event from a compatibility chat stream.
///
/// Unlike MessageDelta, each event carries exactly one visible, reasoning,
/// tool-start or argument-fragment update. Fragments are not parsed, joined or
/// assigned new identities here. Aggregation and delivery remain the caller's.
#[derive(Debug, Clone)]
pub enum ProviderDelta {
    /// A visible text fragment.
    TextDelta {
        /// Text retained verbatim from the provider.
        delta: String,
    },
    /// A reasoning/thinking fragment, separate from visible output.
    ThinkingDelta {
        /// Reasoning text retained verbatim from the provider.
        delta: String,
    },
    /// The start of a tool call under its provider-assigned identifier.
    ToolCallStart {
        /// Identifier that also appears on the corresponding tool result.
        call_id: String,
        /// Provider-reported tool name.
        tool_name: String,
    },
    /// A raw argument fragment for an existing provider-assigned tool call.
    ToolCallArgsDelta {
        /// Identifier used by the corresponding ToolCallStart event.
        call_id: String,
        /// Partial JSON text; it need not be valid JSON until the call completes.
        delta: String,
    },
}

#[cfg(test)]
#[path = "response_tests.rs"]
mod tests;
