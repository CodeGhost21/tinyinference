//! Capacity waiting shares the ledger's atomic admission and settlement events.
use super::*;

impl Budget {
    /// Wait only while live reservations prevent admission against any ancestor.
    /// A call that fits is admitted immediately, without serializing other calls.
    /// Dropping this future before admission reserves and charges nothing.
    ///
    /// # Errors
    /// Returns [`BudgetExceeded`] immediately when settled spend makes admission
    /// impossible on this ledger or an ancestor. Unknown completed calls retain
    /// their conservative charge and can therefore make a waiter unaffordable.
    pub async fn reserve_wait(&self, requested: Spend) -> Result<Reservation, BudgetExceeded> {
        // Arm before checking: a settlement between the refusal and await is
        // retained by watch, so the waiter cannot lose its wakeup.
        let mut changes = self.changes.subscribe();
        loop {
            match self.reserve_inner(requested, true) {
                Ok(reservation) => return Ok(reservation),
                Err(refusal) => {
                    if refusal.terminal {
                        return Err(refusal.error);
                    }
                    let error = refusal.error;
                    log_wait(&error);
                    // The borrowed Budget keeps the sender alive. If that
                    // invariant changes, refusing remains conservative.
                    if changes.changed().await.is_err() {
                        return Err(error);
                    }
                }
            }
        }
    }
}

fn log_wait(error: &BudgetExceeded) {
    tracing::debug!(
        target: "tinyinference::budget", reserved_tokens = error.snapshot.reserved.tokens,
        reserved_cost_micros = error.snapshot.reserved.cost_micros,
        requested_tokens = error.requested.tokens, requested_cost_micros = error.requested.cost_micros,
        "waiting for live budget capacity"
    );
}
