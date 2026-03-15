# Graph Observability And Resume

## Purpose

This note documents the current development-grade execution visibility and the intended insertion points for future human review.

## Checkpointing

- `src/app/graph/checkpoint.py` is the only place that chooses the LangGraph saver backend.
- The current `memory` backend is process-local but shared by namespace, so repeated invocations with the same `thread_id` can inspect or resume checkpoints during one local session.
- `build_checkpoint_config()` builds the base config.
- `build_resume_config()` resolves the latest checkpoint for a `thread_id` and returns a config with `checkpoint_id` populated when available.

## Node Observability

- `src/app/graph/observability.py` wraps every workflow node.
- For each node execution it logs:
  - `thread_id`
  - `node_name`
  - compact state summary before execution
  - compact update summary after execution
- When `APP_GRAPH__ENABLE_STATE_SNAPSHOT_LOGGING=true`, a JSON snapshot is written under `APP_GRAPH__TRACE_DIR/<thread_id>/`.
- Snapshots are intentionally lightweight; they keep counts, route state, coverage label, and manual-review metadata instead of full raw state.

## Manual Review Hook

- `src/app/graph/review_hooks.py` defines the `ManualReviewHook` protocol.
- The default implementation is `NoOpManualReviewHook`.
- Instrumentation can attach review requests to state without changing individual node logic.
- Future implementations can emit blocking requests for:
  - high-severity conflicts
  - unsupported strong claims
  - reduced-mode transcript gaps
  - final-thesis approval gates

## Future Interrupt Insertion Point

- `manual_review_gate` sits between `evidence_verifier` and `synthetic_agent`.
- Today it is non-blocking and only records pending review counts.
- Later it can become a true interrupt node that:
  - pauses on blocking `ManualReviewRequest`
  - persists reviewer decisions
  - resumes with the same `thread_id` and checkpoint lineage

## Practical Development Flow

1. Run the graph with a stable `thread_id`.
2. Inspect trace JSON under `data/graph_traces/<thread_id>/`.
3. Use `build_resume_config(thread_id=...)` to continue from the latest checkpoint in the same process.
4. Replace the saver backend in `checkpoint.py` when durable local or remote persistence is needed.
