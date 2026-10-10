# Shared provider-call budgets

`tinyinference_llm::model::budget` supplies `Budget`, `SpendLimits`,
`CallBudget` and `BudgetedModel`. Hosts own policy and select conservative
per-call input/output token and microdollar bounds for every permitted route.

Wrap concrete providers **below** retry, fallback and cache adapters. Every
invocation atomically reserves against its ledger and all ancestor ledgers.
`Budget::child` gives a turn a local ceiling while charging the shared run.
Completed spend and live reservations share one lock, so concurrent callers
cannot each reserve the same capacity. `Error::BudgetExceeded` carries the
refusing limits, requested amount, completed spend and reservations.

Admission is fail-fast by default. Hosts may opt a ledger handle into capacity
waiting with `Budget::wait_for_capacity()` before passing it to `BudgetedModel`.
Clones and child handles made from it inherit that preference; previously
created handles keep their own preference. Calls whose conservative bounds fit
still run concurrently. Only calls blocked by live reservations wait, and every
wakeup rechecks all ancestor limits atomically. If settled spend plus the call's
bound cannot fit any ceiling, admission returns `BudgetExceeded` immediately.
Direct `Budget::reserve` and provider-internal retry hooks remain fail-fast.
`Budget::reserve_wait` exposes the same asynchronous reservation operation.
Temporary capacity waits do not publish a terminal admission refusal, including
when canceled. An existing fail-fast refusal remains recorded unchanged.

Settlement notifications are retained if settlement occurs before a waiter
starts awaiting the change. Dropping a queued admission future makes no
reservation, incurs no charge and sends no provider request. Dropping an
already admitted paid call still charges its full bound, wakes waiting calls,
and can make them permanently unaffordable. Hosts retain responsibility for
cancellation or a logical deadline on an admission wait; this API does not set
a wall-clock deadline or promise FIFO ordering.

The wrapper caps output tokens and refuses pass-through output cap overrides.
It currently admits text requests only, using complete serialized request
bytes, including response schemas and provider prompt options as a conservative input bound; leave room for provider framing. An
incorrect price or token bound is not a provider-enforced billing cap. Actual
usage exceeding it is retained and stops further calls.

Success reconciles reported usage. Missing or negative charges retain the
reserved cost. Missing token usage retains the token reservation. Provider
errors, dropped streams and cancelled calls retain the complete reservation
because a lost response may have been billed. Cache replays charge nothing.
Streaming items and transport metadata pass through unchanged; a completed
stream reconciles exactly once. Terminal failures charge their reservation
when observed, even if the consumer retains the stream. The physical-attempt
scope remains active while lazy streams are polled.

OpenAI's shared transport tail admits internal request-shape retry attempts
under the invocation's physical-call scope. The initial attempt owns the outer
reservation; each extra attempt reserves again before HTTP and conservatively
retains that reservation. Provider adapters which introduce internal retries
must use this same admission hook before every extra physical send. Ordinary
harness retries enter the wrapper once per attempt.

Tests in `model/budget_tests.rs` cover concurrent admission, ancestor ceilings,
reconciliation, cancellation, streaming completion/drop, cache replay, unknown
charges, overflow, output cap overrides, oversized schemas/provider prompts
and provider-internal retries. `model/budget_wait_tests.rs` also covers affordable
parallel calls, capacity waiting, retained settlement notifications, sibling
contention, permanent ancestor refusal and cancellation before/after admission. The
OpenHuman facade additionally verifies the HTTP boundary with wiremock.
