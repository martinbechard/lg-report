"""Select the claims lesson that retains old record snapshots after an edit.

This is a catalog entry, not another agent or a copy of the scenario. Its sibling
edit_with_reloaded_state/sample.py supplies the five questions and offline model replies;
agent_runtime/workflows/edit_with_reloaded_state.py applies the selected context policy.
The edit-with-reloaded-state variant uses those components with a different default.

The edit-with-patched-state mode carries the original read, edit arguments, receipts,
and conversation into later requests. The model reconstructs facts from history;
updating the store does not update text already sent to the model. The offline
fixture deliberately answers correctly, so this variant demonstrates retained
context rather than proving that retained context causes an incorrect answer.

Importing this file exposes metadata only; the catalog constructs the workflow
when a run is selected. Model/tool execution begins when a client submits turns.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# SampleCatalog discovers both samples. This variant reuses the scenario in
# edit_with_reloaded_state, selecting its own context policy without copying the
# questions or responses. Each folder owns its configuration and report identity.
SAMPLE = {
    "id": "edit-with-patched-state",
    "name": "edit-with-patched-state",
    "description": "Reconstruct current claim state from retained reads and successful edits.",
    # Passed to build_workflow and the offline model factory. Explicit run options
    # can override this default; the ClaimsAgent itself never receives a mode.
    "options": {"mode": "edit-with-patched-state"},
    # Resolve the sibling's scenario and its matching runtime workflow.
    "implementation": "edit_with_reloaded_state",
}
