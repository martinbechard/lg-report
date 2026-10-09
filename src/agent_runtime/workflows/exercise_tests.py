"""Execute the local exercise's generated unittest suite and retain real evidence.

The workflow owns this fixed command; models cannot choose shell commands. Each
attempt uses a fresh Python process so repaired imports cannot reuse stale code.
This is local code execution, not an operating-system sandbox. A minimal child
environment avoids passing provider credentials, and a timeout bounds hangs.
AI attribution: Generated with AI assistance by Codex.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

# Keep machine-readable counts separate from arbitrary test output. Writing the
# receipt after the runner returns also distinguishes an early sys.exit(0) from
# a completed suite. The child uses only the standard library and exercise code.
RUNNER = """
import json, os, sys, unittest
sys.path.insert(0, os.getcwd())
suite = unittest.defaultTestLoader.discover('.', pattern='test_slug.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
with open(sys.argv[1], 'w') as receipt:
    json.dump(dict(tests_run=result.testsRun, failures=len(result.failures),
                   errors=len(result.errors), skipped=len(result.skipped),
                   expected_failures=len(result.expectedFailures),
                   unexpected_successes=len(result.unexpectedSuccesses)), receipt)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def run_tests(directory: Path, *, required: bool, timeout: float = 15) -> dict:
    """Return an observed result, never infer passing tests from written files.

    An implementation-only step may precede test creation. Once tests exist,
    every worker attempt reruns them, including changes to implementation alone.
    Missing required tests, zero tests, all-skipped suites and missing receipts
    cannot authorize approval. Output is bounded for subsequent model context.
    """
    evidence = {'command': 'python -I -B -c <unittest runner>',
                'suite': 'test_slug.py', 'exit_code': None, 'tests_run': 0}
    if not (directory / 'test_slug.py').is_file():
        return {**evidence, 'status': 'missing' if required else 'not_yet_written',
                'stdout': '', 'stderr': 'test_slug.py does not exist.'}
    started = monotonic()
    with TemporaryDirectory(prefix='lg-test-receipt-') as temporary:
        receipt = Path(temporary) / 'result.json'
        # File-backed output prevents a verbose generated test from exhausting
        # the parent process's memory. No shell or inherited API keys are used.
        with (Path(temporary) / 'stdout').open('w+b') as stdout, \
                (Path(temporary) / 'stderr').open('w+b') as stderr:
            process = subprocess.Popen(
                [sys.executable, '-I', '-B', '-c', RUNNER, str(receipt)],
                cwd=directory, env={'PATH': os.defpath}, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Kill the process group, including children started by a test.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            evidence.update(exit_code=process.returncode,
                            duration_ms=round((monotonic() - started) * 1000, 2))
            for name, stream in [('stdout', stdout), ('stderr', stderr)]:
                stream.seek(0)
                data = stream.read(8193)
                evidence[name] = data[:8192].decode('utf-8', errors='replace')
                evidence[name + '_truncated'] = len(data) > 8192
        counts = json.loads(receipt.read_text()) if receipt.exists() else {}
        evidence.update(counts)
        passed = (process.returncode == 0 and counts.get('tests_run', 0) >
                  counts.get('skipped', 0) and not counts.get('failures') and
                  not counts.get('errors') and not counts.get('unexpected_successes'))
        evidence['status'] = 'timeout' if timed_out else 'passed' if passed else 'failed'
    return evidence
