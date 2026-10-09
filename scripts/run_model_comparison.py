"""Run one sample against a list of provider:model choices and compare results.

The normal sample launcher owns each conversation, adaptive user, optional QA,
and recording. This script only supplies shared settings and runs the selected
models sequentially, so concurrent trials cannot compete for local resources.
Saved criteria default to the sample rubric; missing criteria are generated.
Saved-only mode composes existing recordings without any model or price lookup.
Fresh runs require unused bundle folders to preserve previous measurements.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from dotenv import dotenv_values

from agent_runtime.harness.model_providers import get_provider
from agent_runtime.harness.qa_judge import QAJudge, create_rubric, load_rubric
from agent_runtime.harness.sample_catalog import SampleCatalog
from reporting.compare import render_comparison
from reporting.pricing import load_prices
from reporting.render import render
from reporting.schema import QARubric, Run

ROOT = Path(__file__).resolve().parents[1]


def selection(value):
    """Resolve an explicit transport and model without constructing an adapter."""
    provider, separator, model = value.partition(":")
    if not separator or not model.strip() or model in {".", ".."}:
        raise argparse.ArgumentTypeError("Use provider:model, for example codex:gpt-5.5")
    provider = provider.lower()
    try:
        get_provider(provider)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return provider, model


def bundle_name(provider, model):
    """Keep published API folder names, with transport prefixes for other routes.

    URL-encoding model codes prevents slashes in deployment names from escaping
    the selected output directory. Provider/model collisions are rejected below.
    """
    return ("" if provider == "openai" else provider + "-") + quote(model, safe="-._")


def validate_run(path, provider, model, effort, *, allow_failed=False):
    """Reject wrong-provider, simulated, or differently configured input.

    Explicit saved pricing aliases allow a dated API snapshot to resolve to its
    requested name. They never allow one provider to stand in for another.
    Failed executions require an explicit opt-in so benchmarks can retain them.
    """
    run = Run.model_validate_json(path.read_text(encoding="utf-8"))
    prices = load_prices(path.with_name("prices.json"))
    calls = [step for step in run.steps if step.kind == "model"
             and step.context.get("model_role") not in {"user", "qa"}]
    matching = [step for step in calls if step.provider == provider and (
        step.model == model or prices.aliases.get(f"{provider}:{step.model}") == f"{provider}:{model}")]
    accepted = {"ok", "error", "interrupted"} if allow_failed else {"ok"}
    if run.demo or run.status not in accepted or not matching or any(step.effort != effort for step in matching):
        raise ValueError(f"{path}: expected live {provider}:{model} with {effort} effort and status in {sorted(accepted)}")
    return run


def model_effort(provider, model, requested):
    """Omit reasoning configuration for GPT-4.1, including its dated snapshot.

    The comparison can mix reasoning and non-reasoning models. The same
    resolved setting must govern both dispatch and saved-record validation.
    """
    if provider == "openai" and model in {"gpt-4.1", "gpt-4.1-2025-04-14"}:
        return None
    return requested or None


def main(argv=None):
    """Execute the model list once, or rebuild from saved evidence explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=selection, required=True)
    parser.add_argument("--sample", default="simple_chat")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--user-model", default="gpt-6-luna")
    parser.add_argument("--user-provider", default="codex")
    parser.add_argument("--user-turns", type=int, default=3)
    parser.add_argument("--qa", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qa-model", default="gpt-6-sol")
    parser.add_argument("--qa-provider", default="codex")
    parser.add_argument("--qa-rubric", type=Path,
                        help="Reuse saved scoring criteria for the same sample goal")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    parser.add_argument("--prices", type=Path, default=ROOT / "models.json")
    parser.add_argument("--fx-file", type=Path, default=ROOT / "exchange-rate.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/model-comparison")
    parser.add_argument("--title", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse validated existing run bundles and execute missing models")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep recorded execution failures in the comparison and run later models")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--saved-only", action="store_true",
                        help="Build the report from existing selected runs; no model calls")
    modes.add_argument("--rescore", action="store_true",
                       help="Judge saved executions with shared criteria; no Agent reruns")
    args = parser.parse_args(argv)
    names = [bundle_name(*model) for model in args.models]
    if len(names) < 2 or len(set(names)) != len(names):
        parser.error("Select at least two distinct models with distinct output folder names")
    output = args.out.resolve()
    paths = [output / name / "run.json" for name in names]
    try:
        # Preflight every destination before starting paid work. Keeping old
        # bundles makes transport/timing experiments inspectable and recoverable.
        if not args.saved_only and not args.rescore:
            for path in paths:
                if path.parent.exists() and args.resume:
                    index = paths.index(path)
                    provider, model = args.models[index]
                    validate_run(path, provider, model, model_effort(provider, model, args.effort),
                                 allow_failed=args.continue_on_error)
                elif path.parent.exists():
                    raise ValueError(f"Existing bundle: {path.parent}. Choose a new --out or use --saved-only.")
        output.mkdir(parents=True, exist_ok=True)
        rubric = None
        qa_values = {**dotenv_values(args.env_file), **os.environ,
                     "LG_QA_MODEL": args.qa_model, "LG_QA_PROVIDER": args.qa_provider,
                     "LG_QA_EFFORT": "high"}
        if (args.qa or args.rescore) and not args.saved_only:
            sample = SampleCatalog().get(args.sample)
            goal = sample.goal or sample.description
            if args.rescore:
                # Check every saved execution before buying a rubric or changing
                # any assessments. A static run never enters the judge path.
                for (provider, model), path in zip(args.models, paths):
                    run = validate_run(path, provider, model, model_effort(provider, model, args.effort),
                                       allow_failed=args.continue_on_error)
                    if not run.qa or run.qa.goal != goal:
                        raise ValueError(f"{path}: saved goal does not match the selected sample")
            rubric_path = output / "qa-rubric.json"
            if rubric_path.exists() and not args.resume:
                raise ValueError("Existing rubric: use a new output directory for rescoring")
            # Re-running a subset must retain the other columns' scoring
            # contract. An explicit rubric never triggers a planning call.
            if args.resume and rubric_path.exists():
                rubric = QARubric.model_validate_json(rubric_path.read_text(encoding="utf-8"))
                if rubric.goal != goal:
                    raise ValueError("Saved rubric goal does not match the selected sample")
            else:
                # Explicit criteria override the sample default. Only absence
                # permits planning; invalid or outdated files fail visibly.
                source = args.qa_rubric or qa_values.get("LG_QA_RUBRIC") or sample.qa_rubric_path
                rubric = (load_rubric(source, goal) if source else
                          create_rubric(qa_values, goal, sample.name))
            rubric_path.write_text(rubric.model_dump_json(indent=2), encoding="utf-8")
            print("Shared scoring criteria saved.", flush=True)
        for (provider, model), path in zip(args.models, paths):
            effort = model_effort(provider, model, args.effort)
            if not args.saved_only and not args.rescore and not (args.resume and path.exists()):
                path.parent.mkdir()
                env = {**os.environ, "LG_PROVIDER": provider, "LG_MODEL": model,
                       "LG_EFFORT": effort or "", "LG_AVAILABLE_MODELS": "",
                       "LG_USER_EFFORT": "high", "LG_QA_EFFORT": "high"}
                if rubric is not None:
                    env["LG_QA_RUBRIC"] = str(rubric_path)
                # Compare OpenAI without a caller-imposed output cap, even if
                # shared settings contain one. Local login adapters reject caps.
                if provider == "openai" or not get_provider(provider).policy.supports_max_tokens:
                    env["LG_MAX_TOKENS"] = ""
                command = [sys.executable, "-m", "agent_runtime", "--sample", args.sample,
                           "--demo", "--live", "--client", "agent",
                           "--user-model", args.user_model, "--user-provider", args.user_provider,
                           "--user-turns", str(args.user_turns),
                           "--qa" if args.qa else "--no-qa",
                           "--qa-model", args.qa_model, "--qa-provider", args.qa_provider,
                           "--env-file", str(args.env_file.resolve()),
                           "--prices", str(args.prices.resolve()),
                           "--fx-file", str(args.fx_file.resolve()), "--out", str(path.parent)]
                print(f"Running {provider}:{model} ({effort or 'no reasoning effort'}) ...", flush=True)
                with (path.parent / "run.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode and not (args.continue_on_error and path.exists()):
                    raise ValueError(f"{provider}:{model} failed; see {path.parent / 'run.log'}")
            run = validate_run(path, provider, model, effort, allow_failed=args.continue_on_error)
            if run.status != "ok":
                print(f"{provider}:{model}: execution {run.status}; preserved in comparison", flush=True)
            if args.rescore:
                prices = load_prices(path.with_name("prices.json"))
                run.qa = QAJudge(qa_values, goal=rubric.goal, rubric=rubric)(run, prices)
                path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
                path.with_name("prices.json").write_text(prices.model_dump_json(indent=2), encoding="utf-8")
                render(run, prices, path.with_name("report.html"))
                print(f"Assessed {provider}:{model}: {run.qa.status}", flush=True)
            if run.qa and run.qa.status != "completed":
                print(f"{provider}:{model}: QA {run.qa.status}; see individual report", flush=True)
        render_comparison(paths, output / "report.html", title=args.title)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print((output / "report.html").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
