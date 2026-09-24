"""Own fictional claim/policy records and the tools that read or edit them.

Each ClaimStore isolates one demonstration session. Tools expose records only
when invoked by the agent; constructing the store does not load model context.
The agent owns tool registration and instructions; the workflow owns context
invalidation. No real insurance decisions or disk writes occur here.

AI attribution: Generated with AI assistance (Northstar).
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from dataclasses import asdict, dataclass, replace

from langchain_core.tools import tool


@dataclass(frozen=True)
class Policy:
    """Immutable fictional policy, retrieved only if the agent requests it.

    These invented terms are teaching data, not an insurance recommendation.
    Keeping policy separate lets a claim edit invalidate only claim context.
    """

    # The tools have no record-ID parameter: this lesson holds one policy only.
    policy_id: str = "POL-001"
    # Terms are facts available for an answer, not executable adjudication rules.
    covered_event: str = "Accidental damage to a laptop at home"
    deductible_cad: int = 150
    limit_cad: int = 2000


@dataclass(frozen=True)
class Claim:
    """One fictional record; revision identifies the snapshot used for an edit.

    The description and status are the editable teaching fields. Replacing the
    frozen value makes previously captured snapshots stable and lets a failed
    edit leave the entire authoritative record unchanged.
    """

    # Stable identity survives every replacement; this store has no claim lookup.
    claim_id: str = "CLM-001"
    # Increases on every successful edit, even if the supplied fields are equal.
    # The workflow uses this as a change signal without inspecting record fields.
    revision: int = 1
    # The deliberately incorrect starting story makes stale context visible in
    # the audit after the client supplies the accident-at-home correction.
    description: str = "A laptop was stolen from a parked car."
    status: str = "pending"


class ClaimStore:
    """Own one session's in-memory claim and expose real read/edit tools.

    Every run starts fresh. Tools cannot access disk or approve real claims.
    The model chooses whether and how to call them; both context modes get
    exactly the same tools and revision checks.
    """

    def __init__(self, claim: Claim | None = None):
        """Create tools bound to this store, preventing cross-run state sharing."""
        # Each closure accesses the current immutable record, not a snapshot
        # captured at tool registration. A later read therefore sees edits.
        self.claim = claim or Claim()
        self.policy = Policy()

        # @tool turns these functions into LangChain tool definitions. Their
        # signatures and docstrings become model-facing schemas/descriptions;
        # teaching commentary belongs outside those docstrings to keep the
        # experiment's model instructions stable. Decoration does not run a read.
        @tool
        def read_claim() -> str:
            """Read the complete current claim, including its authoritative revision."""
            # Serialize only when called. Creating the store or exposing the tool
            # schema does not send this description to the model.
            return json.dumps(asdict(self.claim))

        @tool
        def edit_claim(expected_revision: int, description: str, status: str) -> str:
            """Replace the claim description and status using a previously read revision.

            Status must be pending, approved, or rejected. A stale revision fails
            without changes; read again before retrying. Success returns the new
            revision, not the resulting claim contents.
            """
            # Validate everything before replacement. A rejected tool request is
            # evidence for the model, but must not trigger context purging.
            # Returning ok=False is a normal tool observation, not an exception:
            # the agent may choose a fresh read and retry within the same turn.
            if expected_revision != self.claim.revision:
                return json.dumps({"ok": False, "error": "Stale revision; read again."})
            if status not in {"pending", "approved", "rejected"}:
                return json.dumps({"ok": False, "error": "Unsupported status."})
            if not description.strip():
                return json.dumps({"ok": False, "error": "Description is required."})
            # strip() above tests emptiness only; successful input is stored as
            # supplied. Status membership is checked, but no coverage decision or
            # human-approval step is implemented by this fictional-record tool.
            # Commit both field replacements together after every check passes.
            # Replacing the frozen record keeps earlier read results historical;
            # it does not magically update text already in model context.
            # This is a local check followed by replacement, not a database
            # transaction or locking mechanism for concurrent claim writers.
            self.claim = replace(
                self.claim,
                revision=self.claim.revision + 1,
                description=description,
                status=status,
            )
            # Return a receipt, not a new snapshot. edit-with-patched-state context must combine
            # the old read with edit arguments; edit-with-reloaded-state context may later lead
            # the agent to request a fresh read. That contrast is the lesson.
            return json.dumps({"ok": True, "revision": self.claim.revision})

        @tool
        def read_policy() -> str:
            """Read fictional policy POL-001, including coverage, deductible, and limit."""
            # Policy retrieval is independent of the claim. Nothing here reads
            # the claim as a side effect of a question about policy terms.
            return json.dumps(asdict(self.policy))

        # The agent registers these definitions; merely assembling this list
        # invokes none of them and transfers no stored record into context.
        self.tools = [read_claim, edit_claim, read_policy]
