# Repository Guidelines

## Scope

TinyInference is a Rust 2024 library workspace. It owns provider-neutral model,
message, tool-call, usage, streaming, provider transport, cache, and embedding
APIs, including stateless construction of provider adapters from a host-resolved
ProviderSpec and matching listing authentication. Agent loops, middleware,
graphs, saved provider registries, provider selection, managed gateway policy,
orchestration, and workspace policy belong in consuming runtimes such as TinyAgents.
Hosts supply credential resolution and HTTP clients; TinyInference does not load
their configuration, own credential stores, or enforce their access/egress gates.

## Structure

The public crates are `crates/tinyinference-core` and
`crates/tinyinference-local`. Core owns provider-neutral inference and hosted
transports; local depends on core and owns hardware, process, filesystem, and
local-runtime integrations. Keep feature areas in module directories with
`mod.rs`, `types.rs`, and `mod_tests.rs` where the area is large. Centralize
deliberate exports in each `src/lib.rs`. Keep provider wire types private unless
callers must construct them.

## Workflow

Make new implementation changes on a feature branch. Prefer direct execution
for clear tasks. Preserve unrelated work and do not rewrite or squash existing
commits.

Run from the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets --all-features
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Do not redirect Cargo output to a temporary target directory. Use the normal
workspace target configuration.

## Code and API

Use Rust 2024 idioms and standard rustfmt. Public fallible APIs return the
crate-wide `Result<T>`. Add a typed `Error` variant when callers need to
distinguish a failure. Never expose provider-specific JSON above the provider
adapter when a normalized type can represent it.

`ChatModel` and `EmbeddingModel` implementations must be `Send + Sync`. Stream
adapters emit terminal `Completed` or `ProviderFailed` items and must preserve
tool-call ids, incremental arguments, reasoning, finish reasons, and usage.
Never log or expose API keys; custom `Debug` implementations must redact them.

Every public item needs rustdoc. Document `# Errors` and `# Panics` where
applicable. Keep Markdown files at 500 lines or fewer.

Generic provider failure facts and authentication failure episode state belong
to these library APIs. Hosts own state lifetime, timestamps, presentation text,
managed-provider exclusions, classifier precedence, notifications and telemetry
delivery. A library registry of authentication episodes does not load or select
a host's saved provider configuration.

## Tests

Keep unit tests beside their module in sibling `*_tests.rs` files, never an inline `mod tests` block. Tests must not depend on network access, wall-clock
timing, ambient credentials, or mutable process environment. Use synthetic HTTP
payloads and byte streams for provider behavior. Add tests for serialization,
stream reconstruction, malformed tool calls, error classification, and vector
dimension contracts whenever those surfaces change.

## Commits and Pull Requests

Keep commits small and focused. Pull requests should summarize API/behavior
changes and list exact verification commands. Open ready-for-review PRs against
`tinyhumansai/tinyinference`; use drafts only for genuinely incomplete work.

## Tests live in `*_tests.rs` files

- Unit tests are never inline. Do not write a `#[cfg(test)] mod tests { ... }`
  block in a source file. Put the tests in a sibling `<module>_tests.rs`
  (`mod_tests.rs` beside a `mod.rs`, `lib_tests.rs` beside `lib.rs`) and declare
  it at the bottom of the module:

  ```rust
  #[cfg(test)]
  #[path = "foo_tests.rs"]
  mod tests;
  ```

- The test file starts with `use super::*;` and carries no `#[cfg(test)]` of its
  own. It is still a child module, so it reaches private items exactly as an
  inline module did.
- Name test files `<module>_tests.rs`; a second group for the same module is
  `<module>_<topic>_tests.rs`. Never `test.rs`, `tests.rs` or `<module>_test.rs`.
- Integration tests stay in the crate's `tests/` directory.
- OpenHuman's `scripts/externalize-inline-tests.mjs <repo-root> --write` moves
  inline test modules out mechanically; without `--write` it only reports.
