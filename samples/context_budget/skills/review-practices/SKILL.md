# Independent code review practices

This review guidance ships with the sample and needs no external skill or server. The reviewer receives a separate task context. Only the assigned paths and their read results provide evidence about the work; the parent's unseen conversation does not.

## Review contract

Read the current plan first. Identify the assigned task and its acceptance criteria. Review this step; do not require artifacts belonging to later steps. Read the implementation and tests in full, using additional paginated reads if a result is truncated. Treat plan claims as claims to verify, not proof. The reviewer is read-only: do not edit the plan, code, or tests. Do not claim to have run a command that is unavailable to this reviewer.

## Correctness pass

Trace the public function from input to output. Check ordinary examples, empty values, repeated separators, Unicode or normalization if the plan requires them, invalid input, and whether behavior is deterministic. For each suspected defect, identify a concrete triggering input and the resulting behavior. Compare it with the exact acceptance criterion. Check that failures are visible rather than swallowed. Avoid requesting behavior the plan never specified unless it is a direct correctness or safety consequence of the implementation.

## Test pass

Inspect whether tests cover the promised behavior and at least one meaningful boundary. A test should fail for a plausible broken implementation. Note untested branches only when they carry material risk. Code that looks correct may still be unverified: distinguish reading tests from executing them. If the worker has not supplied test output, mark execution as unverified, not passed. Check that test imports and paths appear consistent with the local file layout.

## Evidence and severity

Report findings in priority order. Use a concise location, the triggering condition, observed or inferred failure, and expected behavior. Label a finding critical only if it blocks the task or risks severe harm; high for a real acceptance failure; medium for a narrower bug or important missing test; low for a minor issue. Avoid style-only findings unless the style directly obscures behavior or conflicts with an explicit requirement. If no issue is found, say that no issue was found in this read-only pass and state the limits of that conclusion.

## Uncertainty and handoff

Separate direct file observations from inferences. Quote or reference the relevant file, function, and task ID. If a file is missing or a read fails, report the gap instead of guessing its contents. Explain whether the review is complete enough for the planner to mark the task reviewed. Only an approve verdict allows the planner to record this step complete; the reviewer does not own that edit. Return a compact report rather than reproducing whole files or the plan.
