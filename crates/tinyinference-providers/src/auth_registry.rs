//! Episode tracking for provider authentication failures.
//!
//! The host supplies presentation text and timestamps and decides when to
//! record or clear failures. Each instance owns its state; notification and
//! telemetry delivery remain host responsibilities.

use std::collections::HashMap;
use std::sync::RwLock;

/// A recorded authentication failure, suitable for a host settings snapshot.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProviderAuthError {
    /// Provider slug used by the host's configuration.
    pub provider: String,
    /// Rejecting HTTP status; the host's classifier controls which statuses
    /// are recorded.
    pub status: u16,
    /// Host-supplied presentation text. Raw credentials must not be included.
    pub message: String,
    /// Host-supplied wall-clock milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

/// Thread-safe authentication failure episodes, keyed by provider slug.
///
/// Repeated records refresh the entry without opening a new episode. Clearing
/// it re-arms the first-failure latch, allowing the host to notify once again.
/// Poisoned locks recover their contents, matching the persisted process
/// registry behavior expected by hosts.
#[derive(Debug, Default)]
pub struct ProviderAuthErrorRegistry {
    entries: RwLock<HashMap<String, ProviderAuthError>>,
}

impl ProviderAuthErrorRegistry {
    /// Creates an empty, independent registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores a failure and returns true only for a new episode.
    pub fn record(&self, entry: ProviderAuthError) -> bool {
        let mut entries = self.entries.write().unwrap_or_else(|error| error.into_inner());
        let is_new = !entries.contains_key(&entry.provider);
        entries.insert(entry.provider.clone(), entry);
        is_new
    }

    /// Removes a provider's episode and returns whether one existed.
    pub fn clear(&self, provider: &str) -> bool {
        self.entries.write().unwrap_or_else(|error| error.into_inner())
            .remove(provider).is_some()
    }

    /// Returns a snapshot sorted by provider slug.
    pub fn snapshot(&self) -> Vec<ProviderAuthError> {
        let entries = self.entries.read().unwrap_or_else(|error| error.into_inner());
        let mut snapshot: Vec<_> = entries.values().cloned().collect();
        snapshot.sort_by(|left, right| left.provider.cmp(&right.provider));
        snapshot
    }

    /// Clears all episodes. Hosts use this when replacing runtime identity
    /// or resetting an explicitly scoped registry.
    pub fn clear_all(&self) {
        self.entries.write().unwrap_or_else(|error| error.into_inner()).clear();
    }
}

#[cfg(test)]
#[path = "auth_registry_tests.rs"]
mod tests;
