"""Discover teaching samples and construct their workflows for any client.

Each sample.py declares SAMPLE metadata beside conversation data and optional
model factories. Discovery imports these trusted local modules but does not call
the factories. Workflow paths follow the ID or shared implementation name.
CLI and HTTP use this same catalog, while each conversation owns its models, prompts, and resources.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import re
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from dotenv import dotenv_values

import samples

from .model_factory import model_factory_scope


@dataclass(frozen=True)
class Sample:
    """Keep local sample metadata separate from any running conversation."""

    id: str
    name: str
    description: str
    implementation: str
    directory: Path
    options: dict = field(default_factory=dict)
    tracing: str = "local"
    mcp_tools: tuple[str, ...] = ()
    interaction: str | None = None
    default_client: str = "static"

    @property
    def workflow(self) -> str:
        """Locate the conventional workflow builder without storing a module path."""
        return f"agent_runtime.workflows.{self.implementation}:build_workflow"

    @property
    def definition_module(self) -> str:
        """Locate scenario data alongside the implementation's teaching sample."""
        return f"samples.{self.implementation}.sample"


class SampleCatalog:
    """Read installed sample folders once; instantiate for a fresh discovery pass.

    root may select another local sample collection. Metadata is trusted local
    configuration, not an endpoint for importing names supplied by HTTP clients.
    Duplicate IDs and malformed metadata fail with their source path instead of
    silently hiding a sample. Workflow imports remain deferred until selection.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else Path(samples.__file__).parent
        self.samples: dict[str, Sample] = {}
        for path in sorted(self.root.glob("*/sample.py")):
            try:
                if self.root.resolve() == Path(samples.__file__).parent.resolve():
                    module = import_module(f"samples.{path.parent.name}.sample")
                else:
                    # Alternate collections are trusted local Python too. Load
                    # their actual file, not an installed module with the same name.
                    spec = spec_from_file_location(f"sample_{path.parent.name}", path)
                    module = module_from_spec(spec)
                    spec.loader.exec_module(module)
                # A helper module without SAMPLE contributes no catalog entry.
                # Shared implementations may also declare a selectable sample.
                if not hasattr(module, "SAMPLE"):
                    continue
                entry = module.SAMPLE
                if not isinstance(entry, dict) or "samples" in entry:
                    raise ValueError("sample.py must contain one sample object")
                if (
                    "workflow" in entry
                    or "scripted_run" in entry
                    or "definition_module" in entry
                ):
                    raise ValueError(
                        "workflow and definition_module are derived from the sample ID or implementation"
                    )
                for key in ("id", "name", "description"):
                    if not isinstance(entry.get(key), str) or not entry[key].strip():
                        raise ValueError(f"{key} must be a nonempty string")
                # Public IDs also serve as CLI choices and report directory
                # names, so allow descriptive hyphenated names. Python imports
                # still require a valid implementation module, checked below.
                if not re.fullmatch(r"[a-z][a-z0-9_-]*", entry["id"]):
                    raise ValueError(
                        "id must use lowercase letters, digits, underscores, or hyphens"
                    )
                if entry["id"] in self.samples:
                    raise ValueError(f"Duplicate sample id: {entry['id']}")
                # Most lessons use their ID as the Python module name. Variants
                # name shared code once instead of repeating two import strings.
                implementation = entry.get("implementation", entry["id"])
                if not isinstance(implementation, str) or not re.fullmatch(
                    r"[a-z][a-z0-9_]*", implementation
                ):
                    raise ValueError(
                        "implementation must be a lowercase sample module name"
                    )
                if not isinstance(entry.get("options", {}), dict):
                    raise TypeError("options must be an object")
                if entry.get("tracing", "local") not in {"local", "langfuse"}:
                    raise ValueError("tracing must be local or langfuse")
                if entry.get("interaction") not in {None, "approval", "clarification"}:
                    raise ValueError("interaction must be approval or clarification")
                if entry.get("default_client", "static") not in {
                    "static",
                    "console",
                    "angular",
                }:
                    raise ValueError(
                        "default_client must be static, console, or angular"
                    )
                tools = entry.get("mcp_tools", [])
                if not isinstance(tools, list) or not all(
                    isinstance(t, str) for t in tools
                ):
                    raise ValueError("mcp_tools must be a list of names")
                sample = Sample(
                    id=entry["id"],
                    name=entry["name"],
                    description=entry["description"],
                    implementation=implementation,
                    directory=path.parent,
                    options=entry.get("options", {}),
                    tracing=entry.get("tracing", "local"),
                    mcp_tools=tuple(tools),
                    interaction=entry.get("interaction"),
                    default_client=entry.get("default_client", "static"),
                )
                self.samples[sample.id] = sample
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid sample metadata {path}: {exc}") from exc

    def get(self, sample_id: str) -> Sample:
        """Resolve a registered ID; arbitrary module names cannot select a sample."""
        try:
            return self.samples[sample_id]
        except KeyError:
            raise ValueError(f"Unknown sample: {sample_id}") from None

    def script(self, sample_id: str):
        """Resolve the selected implementation module for prompts and model factories."""
        return import_module(self.get(sample_id).definition_module)

    def prompts(self, sample_id: str) -> list[str]:
        """Return a fresh list without owning a conversation's sequence position."""
        from .model_factory import client_prompts

        script = self.script(sample_id)
        prompts = (
            client_prompts(script.CONVERSATION)
            if hasattr(script, "CONVERSATION")
            else list(script.USER_PROMPTS)
        )
        if not all(isinstance(p, str) for p in prompts):
            raise ValueError(f"Invalid client prompts for {sample_id}")
        return prompts

    def info(self, sample_id: str) -> dict:
        """Project public menu data without credentials or executable configuration."""
        sample = self.get(sample_id)
        return {
            "id": sample.id,
            "title": sample.name,
            "description": sample.description,
            "prompts": self.prompts(sample_id),
            "mcpTools": list(sample.mcp_tools),
        }

    def configuration(self, sample_id, env_file=None):
        """Read one sample configuration without leaking values into later selections."""
        path = env_file or self.get(sample_id).directory / ".env"
        return {
            key: value
            for key, value in dotenv_values(path).items()
            if value is not None
        }

    def create_run(
        self, sample_id: str, live: bool, *, options=None, tracing=True, env_file=None
    ):
        """Construct models inside one factory scope; tools execute only during runs.

        The workflow owns any temporary workspace. Langfuse access is checked
        before constructing models and its recorder is closed on setup failure.
        tracing=False lets execute_conversation attach console tracing scopes itself.
        """
        sample = self.get(sample_id)
        arguments = {**sample.options, **(options or {})}
        settings = self.configuration(sample_id, env_file)
        recorder = None
        if tracing and sample.tracing == "langfuse":
            from .langfuse_runtime import LangfuseCapture

            recorder = LangfuseCapture(settings=settings)
        try:
            script = None if live else self.script(sample_id)

            # CONVERSATION supplies chronological data. An optional build_scripted_models
            # callback preserves specialized adapters and option-dependent scripts.
            # Special fixtures react to real file/shell observations or maintain
            # separate ledgers. They remain behind the same build_model boundary.
            def resolve(caller):
                # Each request gets a fresh response cursor, even when one
                # caller constructs multiple adapters in the same workflow.
                models = script.build_scripted_models(arguments)
                try:
                    return models[caller]
                except KeyError:
                    raise ValueError(
                        f"No scripted model for {sample_id}: {caller}"
                    ) from None

            conversation = getattr(script, "CONVERSATION", [])
            resolver = (
                resolve
                if script is not None and hasattr(script, "build_scripted_models")
                else None
            )
            module, function = sample.workflow.split(":")
            with model_factory_scope(
                live=live,
                conversation=conversation,
                resolver=resolver,
                settings=settings,
                sample_description=sample.description,
            ) as identity:
                graph = getattr(import_module(module), function)(**arguments)
            # Context-policy lessons return their context owner for direct tests.
            # All clients receive its graph, with the audit attached for reporting.
            if hasattr(graph, "evidence") and hasattr(graph, "graph"):
                context = graph
                graph = context.graph
                graph.context_audit = context
            if recorder:
                graph.trace_recorder = recorder
            return graph, *identity
        except BaseException:
            if recorder:
                recorder.close()
            raise
