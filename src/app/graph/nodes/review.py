"""Conflict checking, verification, and synthesis node skeletons."""

from __future__ import annotations

from app.domain import CoverageLabel, CoverageStatus, VerificationLabel
from app.graph.conflicts import collect_conflicts
from app.graph.nodes.common import make_memory_entry, make_packet, state_update
from app.graph.state import ResearchState
from app.graph.verification import HeuristicEvidenceVerifier
from app.synthesis.service import collect_verified_findings


def conflict_checker(state: ResearchState) -> dict[str, object]:
    """Stub for cross-agent conflict detection."""

    conflicts = collect_conflicts(state.get("agent_packets", []))
    packet = make_packet(
        state=state,
        agent_name="conflict_checker",
        summary=f"Conflict checker produced {len(conflicts)} explainable conflict candidates.",
        coverage_label=CoverageLabel.COMPLETE if conflicts else CoverageLabel.PARTIAL,
    )
    return state_update(
        step_name="conflict_checker",
        packet=packet,
        conflicts=conflicts,
    )


def evidence_verifier(state: ResearchState) -> dict[str, object]:
    """Run deterministic claim verification and expose claim-level results."""

    verifier = HeuristicEvidenceVerifier()
    results = verifier.verify_packets(
        state.get("agent_packets", []),
        state.get("conflicts", []),
    )
    supported_count = sum(1 for result in results if result.label == VerificationLabel.SUPPORTED)
    partial_count = sum(
        1 for result in results if result.label == VerificationLabel.PARTIALLY_SUPPORTED
    )
    issue_count = sum(
        1
        for result in results
        if result.label
        in {
            VerificationLabel.CONFLICTING,
            VerificationLabel.INSUFFICIENT,
            VerificationLabel.UNAVAILABLE,
        }
    )
    summary = (
        "Evidence verifier checked "
        f"{len(results)} claims; supported={supported_count}, "
        f"partial={partial_count}, issues={issue_count}."
    )
    preserved_missing_topics = list(
        dict.fromkeys(
            [
                *state.get("coverage_status", CoverageStatus.not_started()).missing_topics,
                *[topic for result in results for topic in result.missing_coverage_topics],
            ]
        )
    )
    packet = make_packet(
        state=state,
        agent_name="evidence_verifier",
        summary=summary,
        coverage_label=CoverageLabel.COMPLETE if results else CoverageLabel.PARTIAL,
        extra_memory_updates=[
            make_memory_entry(
                state=state,
                source_agent="evidence_verifier",
                key="verification.summary",
                value=summary,
            )
        ],
    )
    return state_update(
        step_name="evidence_verifier",
        packet=packet,
        verification_results=results,
        verifier_notes=[summary],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE if results else CoverageLabel.PARTIAL,
            covered_topics=["verification"],
            missing_topics=preserved_missing_topics,
            notes="Claim-level evidence verification completed.",
        ),
    )


def synthetic_agent(state: ResearchState) -> dict[str, object]:
    """Prepare verified-only synthesis inputs for the final report."""

    verification_results = state.get("verification_results", [])
    verified_findings, excluded_findings = collect_verified_findings(
        state.get("agent_packets", []),
        verification_results,
    )
    summary = (
        "Synthetic agent consolidated "
        f"{len(verified_findings)} verified findings and excluded "
        f"{len(excluded_findings)} unsupported claims."
    )
    packet = make_packet(
        state=state,
        agent_name="synthetic_agent",
        summary=summary,
        coverage_label=CoverageLabel.PARTIAL,
        extra_memory_updates=[
            make_memory_entry(
                state=state,
                source_agent="synthetic_agent",
                key="synthesis.verification_count",
                value=str(len(verification_results)),
            ),
            make_memory_entry(
                state=state,
                source_agent="synthetic_agent",
                key="synthesis.verified_finding_count",
                value=str(len(verified_findings)),
            ),
        ],
    )
    return state_update(step_name="synthetic_agent", packet=packet)


def manual_review_gate(state: ResearchState) -> dict[str, object]:
    """Placeholder gate for future interrupt and human-review workflows."""

    pending_requests = state.get("manual_review_requests", [])
    summary = (
        "Manual review gate observed "
        f"{len(pending_requests)} pending review requests and continued in non-blocking mode."
    )
    packet = make_packet(
        state=state,
        agent_name="manual_review_gate",
        summary=summary,
        coverage_label=CoverageLabel.PARTIAL,
        extra_memory_updates=[
            make_memory_entry(
                state=state,
                source_agent="manual_review_gate",
                key="manual_review.pending_count",
                value=str(len(pending_requests)),
            )
        ],
    )
    return state_update(
        step_name="manual_review_gate",
        packet=packet,
        verifier_notes=[summary],
    )
