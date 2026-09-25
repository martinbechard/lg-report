"""Own a fictional correction scenario and explicitly scripted answer outcomes.

This module declares the edit-with-reloaded-state sample and owns the shared five-turn
scenario. edit-with-patched-state selects the same implementation with its own mode
default, so the comparison uses identical questions, agent instructions, and tools.
The runtime workflow composes the agent and context policy; its ClaimStore owns
the records. None of the dictionaries below directly edits or reads that store.

Offline model decisions demonstrate tool execution and context mechanics only.
Both modes answer correctly in this fixture: we do not manufacture model failure
by hard-coding a wrong edit-with-patched-state answer. Live mode reuses only the client entries and lets
the configured model choose actions; only live runs can reveal model confusion.

AI attribution: Generated with AI assistance (Northstar).
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.simulated_model import SimulatedModel

# Keeping a separate catalog ID gives this comparison its own configuration
# directory and default report destination. Both experiments share the scenario,
# agent instructions, and authoritative in-memory store implementation.
SAMPLE = {
    "id": "edit-with-reloaded-state",
    "name": "edit-with-reloaded-state",
    "description": "Discard stale claim context after edits; let the agent reload when needed.",
    # This default selects both the workflow policy and the offline fixture's
    # extra follow-up read. In live mode only the model chooses whether to reread.
    # Explicit run options may override the catalog default.
    "options": {"mode": "edit-with-reloaded-state"},
    # The public ID uses hyphens; imports use this Python module name. The patched
    # sibling selects this same script and workflow with a different mode option.
    "implementation": "edit_with_reloaded_state",
}


# Read this as a chronological teaching script, not as preloaded agent context:
# - client entries become the successive human requests in a scripted run;
# - ai entries become offline model responses, including proposed tool calls;
# - tool entries explain expected observations, but are never replayed as results.
# LangChain's agent loop executes the requested tools and supplies their real
# results. A call's id links its AI request to the corresponding tool observation.
# An optional mode marks only the edit-with-reloaded-state fixture's reread; unmarked entries are
# shared. Live execution uses client questions and ignores these authored replies.
CONVERSATION = [
    # Turn 1 establishes policy context before any claim data is needed. This
    # read will remain useful after a later claim edit because policy is immutable.
    {
        "role": "client",
        "content": "What deductible does policy POL-001 apply to accidental laptop damage?",
    },
    {
        "role": "claims_agent",
        "content": "",
        "tool_calls": [
            {
                "name": "read_policy",
                "args": {},
                "id": "policy-read",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_policy",
        "tool_call_id": "policy-read",
        "content": "Expected: actual read_policy result for {}; supplied by the running tool.",
    },
    {"role": "claims_agent", "content": "The policy deductible is CAD 150."},
    # Turn 2 captures the original claim at revision 1. This snapshot deliberately
    # becomes stale later; changing the store cannot retroactively change it.
    {
        "role": "client",
        "content": "What happened in claim CLM-001, and what is its status?",
    },
    {
        "role": "claims_agent",
        "content": "",
        "tool_calls": [
            {
                "name": "read_claim",
                "args": {},
                "id": "initial-read",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_claim",
        "tool_call_id": "initial-read",
        "content": "Expected: actual read_claim result for {}; supplied by the running tool.",
    },
    {
        "role": "claims_agent",
        "content": "The laptop was stolen from a parked car. Status: pending.",
    },
    # Turn 3 corrects both mutable fields using the revision just read. The edit
    # returns a success receipt and new revision, not a refreshed claim snapshot.
    # edit-with-reloaded-state invalidation happens only after this entire agent turn finishes,
    # including its final acknowledgement, rather than immediately inside a tool.
    {
        "role": "client",
        "content": "Correct the claim: the laptop was not stolen. Its screen cracked when it fell from a desk at "
        "home. Replace the description with 'The laptop screen cracked when it fell from a desk at "
        "home.' and change the status to approved.",
    },
    {
        "role": "claims_agent",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_claim",
                "args": {
                    "expected_revision": 1,
                    "description": "The laptop screen cracked when it fell from a desk at home.",
                    "status": "approved",
                },
                "id": "correction",
            }
        ],
    },
    {
        "role": "tool",
        "name": "edit_claim",
        "tool_call_id": "correction",
        "content": "Expected: actual edit_claim result for {'expected_revision': 1, 'description': 'The laptop "
        "screen cracked when it fell from a desk at home.', 'status': 'approved'}; supplied by the "
        "running tool.",
    },
    {"role": "claims_agent", "content": "The correction was saved."},
    # Turn 4 isolates selective retention: the earlier policy observation is
    # sufficient in either mode. The edit-with-reloaded-state fixture must not need a claim reload
    # merely because a previous turn edited the claim.
    {"role": "client", "content": "What is the policy's coverage limit?"},
    {"role": "claims_agent", "content": "The policy coverage limit is CAD 2,000."},
    # Turn 5 asks for current claim facts. edit-with-patched-state history contains the original
    # read plus the successful replacement arguments. edit-with-reloaded-state history no longer
    # contains either, so its scripted model asks for a fresh authoritative read.
    # The mode filter below selects that response; this is not live model reasoning.
    {
        "role": "client",
        "content": "What happened to the laptop, where did it happen, and what is the current claim status?",
    },
    {
        "role": "claims_agent",
        "content": "",
        "tool_calls": [
            {
                "name": "read_claim",
                "args": {},
                "id": "follow-up-read",
            }
        ],
        "mode": "edit-with-reloaded-state",
    },
    {
        "role": "tool",
        "name": "read_claim",
        "tool_call_id": "follow-up-read",
        "content": "Expected: actual read_claim result for {}; supplied by the running tool.",
        "mode": "edit-with-reloaded-state",
    },
    {
        "role": "claims_agent",
        "content": "The laptop screen cracked when it fell from a desk at home. The status is approved.",
    },
]


def make_simulated_model(mode):
    """Replay agent tool choices to inspect context, not to measure answer quality.

    Both modes produce the same correct final answer. The edit-with-patched-state fixture combines
    the original snapshot and successful edits; the edit-with-reloaded-state fixture requests a
    fresh read because its claim snapshot was removed. All choices are scripted.
    """
    # Defaulting an absent marker to the requested mode keeps every shared entry.
    # Only explicitly edit-with-reloaded-state entries disappear from the edit-with-patched-state response queue.
    # Workflow construction validates the mode; this fixture assumes that choice.
    conversation = [entry for entry in CONVERSATION if entry.get("mode", mode) == mode]
    # The simulator filters named-agent replies. The explanatory tool entries
    # above cannot substitute canned observations for the real tool executions.
    return SimulatedModel(conversation=conversation, cache_reuse=False)


def build_scripted_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    # The key matches build_model(caller="workflow") in the shared workflow.
    # Return a fresh adapter so each run starts at the first scripted response.
    # The catalog uses this hook only in demo mode; live construction bypasses it.
    return {
        "workflow": make_simulated_model(
            options.get("mode", "edit-with-reloaded-state")
        )
    }
