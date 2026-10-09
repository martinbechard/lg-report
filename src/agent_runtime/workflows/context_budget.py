"""Run one implementation step through review before the planner records it.

Planner and worker retain shared, compactable history. Each review starts fresh.
Generated tests run before review; failed execution blocks approval. Validated
model responses select the next task and approve or reject its current work; declared graph edges describe those possibilities in the generated report.
Task and verdict state live outside summarized messages so compaction cannot
silently turn a pending review into approval.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import dynamic_prompt
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    model_validator,
)

from agent_runtime.agents import isolated_reviewer, planner, worker
from agent_runtime.context_budget import ContextBudget
from agent_runtime.harness.gateway_model import GatewayChatModel
from agent_runtime.harness.gateway_retry import GatewayDecisionMiddleware
from agent_runtime.harness.model_factory import build_model
from agent_runtime.workflows.exercise_backend import ExerciseBackend
from agent_runtime.workflows.exercise_tests import run_tests

WORKFLOW_BUDGET = ContextBudget(max_input_tokens=18000, trigger_tokens=8500, keep_tokens=900)
SKILLS = files('samples.context_budget').joinpath('skills')


class Dispatch(BaseModel):
    """The planner selects one next task or ends the current user turn."""

    model_config = ConfigDict(extra='forbid')
    action: Literal['work', 'finish']
    task_id: str
    files: list[Literal['/slug.py', '/test_slug.py']]
    message: str = Field(min_length=1)

    @model_validator(mode='after')
    def check_task(self, info: ValidationInfo):
        """Reject ambiguous dispatch rather than silently choosing a task."""
        if (self.action == 'work') != bool(self.files):
            raise ValueError('work requires writable files; finish requires no files')
        if (self.action == 'work') != bool(self.task_id.strip()):
            raise ValueError('work requires a task ID; finish requires an empty task ID')
        if self.action == 'work' and self.task_id in (info.context or {}).get('completed', []):
            raise ValueError('Planner redispatched an already approved task; select a pending task or finish')
        if self.action == 'work' and (info.context or {}).get('plan_exists') is False:
            raise ValueError('Write /plan.md with the task plan before dispatching work')
        return self


class Review(BaseModel):
    """Approval belongs to one task and the files just inspected by the reviewer."""

    model_config = ConfigDict(extra='forbid')
    task_id: str = Field(min_length=1)
    verdict: Literal['approve', 'revise']
    evidence: str = Field(min_length=1)


class StepState(MessagesState):
    """Keep routing facts separate from the lossy conversation summary."""

    dispatch: dict  # Current planner assignment; replaced only after approval.
    review: dict  # Latest independent assessment, never inferred from prose.
    attempts: int  # Worker attempts for this task, including rejected versions.
    outcome: str  # Completed or explicitly stopped without approval.
    completed: list[str]  # Approved task IDs in this user turn; prevents redispatch.
    test_execution: dict  # Current worker attempt's observed suite result.


def _answer(result):
    """Use the final visible answer, never opaque reasoning or a tool request."""
    message = result['messages'][-1]
    if not isinstance(message, AIMessage) or message.tool_calls:
        raise ValueError('Role must finish with an assistant response')
    return message.text


def build_workflow(planner_model=None, worker_model=None, reviewer_model=None,
                   workflow_summary_model=None, *, workflow_budget=WORKFLOW_BUDGET,
                   workspace_dir=None, max_attempts=3):
    """Compile the response-driven loop with bounded handoff-format correction.

    A finite repair limit ends honestly with an unapproved task. An explicit
    workspace preserves files; otherwise the application owns temporary cleanup.
    Malformed handoffs get two correction calls; transport failures and invalid
    approval identity still propagate without retrying completed tool work.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError('max_attempts must be a positive integer')
    workspace = TemporaryDirectory(prefix='lg-context-budget-') if workspace_dir is None else None
    try:
        directory = Path(workspace.name if workspace else workspace_dir)
        coding = (SKILLS / 'coding-practices' / 'SKILL.md').read_text()
        reviewing = (SKILLS / 'review-practices' / 'SKILL.md').read_text()
        models = [planner_model, worker_model, reviewer_model, workflow_summary_model]
        for i, caller in enumerate(('planner', 'worker', 'isolated-reviewer', 'workflow-summary')):
            if models[i] is None:
                models[i] = build_model(caller=caller)
        # Structured handoffs need enough visible completion space after reasoning.
        for i in range(3):
            limit = getattr(models[i], 'max_tokens', None)
            if isinstance(limit, int) and limit < 4096:
                models[i] = models[i].model_copy(update={'max_tokens': 4096})
        summary = models[3].model_copy(update={'name': 'context_summarizer'})

        @dynamic_prompt
        def current_assignment(request):
            """Keep role and current-task authority on EVERY call after compaction.

            Native summaries may contain stale next steps. Runtime context is
            supplied by the graph and never summarized with conversation text.
            """
            return request.system_message.text + (
                '\n\nAuthoritative current workflow operation; do ONLY this operation, '
                'even if a history summary suggests later work:\n'
                + json.dumps(request.runtime.context)
            )

        backends = {}

        def parameters(model, paths, *, shared):
            """Give each role only its write authority; all can read the plan."""
            backend = ExerciseBackend(directory, writable_paths=paths)
            backends[tuple(paths)] = backend
            middleware = [current_assignment, FilesystemMiddleware(backend=backend, tools=(
                ['read_file', 'write_file', 'edit_file'] if paths else ['read_file']))]
            # Gateway corrections run inside compaction/input guards so every
            # corrected request still passes the same context ceiling.
            if isinstance(model, GatewayChatModel):
                middleware.append(GatewayDecisionMiddleware())
            if shared:
                middleware.extend(workflow_budget.middleware(summary))
            return {'model': model, 'middleware': middleware, 'context_schema': dict}

        planning = planner.build_agent(parameters(models[0], ['/plan.md'], shared=True), coding)
        working = worker.build_agent(parameters(models[1], ['/slug.py', '/test_slug.py'], shared=True), coding)
        reviewer = isolated_reviewer.build_agent(parameters(models[2], [], shared=False), reviewing)

        def role_node(agent, phase):
            """Frame current routing facts anew, even if earlier messages compacted."""
            def directive(state):
                """Supply authoritative task facts separately from summarized history."""
                if phase == 'planning':
                    directive = {'operation': 'plan', 'instruction': 'Prepare pending implementation steps for the latest user request and assign one, or finish with the requested report when no code/test changes are needed. Reporting and /plan.md edits belong to you, never the worker. Do not record unapproved completion.'}
                elif phase == 'update':
                    if state['review']['verdict'] != 'approve':
                        raise ValueError('Plan completion requires independent approval')
                    directive = {'operation': 'record_approved_step', 'assignment': state['dispatch'],
                                 'approval': state['review'],
                                 'test_execution': state.get('test_execution', {}),
                                 # Compaction must never erase this routing
                                 # constraint or make an old task eligible again.
                                 'completed_task_ids': [*state['completed'], state['dispatch']['task_id']]}
                else:
                    directive = {'operation': phase, 'assignment': state['dispatch'], 'review': state.get('review', {})}
                    directive['test_execution'] = state.get('test_execution', {})
                return directive

            def payload(state):
                """Reviewers start with just the assignment, never parent history."""
                # Reviewers see only this task and actual file reads. The worker's
                # prose and the shared history cannot masquerade as approval.
                history = [] if phase == 'review' else state['messages']
                return {'messages': [*history, HumanMessage(json.dumps(directive(state)))]}

            def finish(state, result):
                """Validate the role output before exposing any next graph route."""
                text = _answer(result)
                if phase == 'review':
                    verdict = Review.model_validate_json(text)
                    if verdict.task_id != state['dispatch']['task_id']:
                        raise ValueError('Review task ID does not match current assignment')
                    # Source approval cannot erase failed execution. Preserve
                    # the review findings, but keep the task in the repair loop.
                    execution = state.get('test_execution', {})
                    if execution.get('status') not in ('passed', 'not_yet_written'):
                        verdict.verdict = 'revise'
                        verdict.evidence += ' Test execution requires repair: ' + json.dumps(execution)
                    # Only the final assessment crosses the isolated boundary.
                    return {'review': verdict.model_dump(), 'messages': [AIMessage(content='Independent review: ' + text)]}
                update = {'messages': [RemoveMessage(id=REMOVE_ALL_MESSAGES), *result['messages']]}
                if phase in ('planning', 'update'):
                    completed = [] if phase == 'planning' else [*state['completed'], state['dispatch']['task_id']]
                    dispatch = Dispatch.model_validate_json(text, context={
                        'completed': completed, 'plan_exists': (directory / 'plan.md').is_file()})
                    update.update(dispatch=dispatch.model_dump(), attempts=0, review={}, completed=completed)
                else:
                    update['attempts'] = state['attempts'] + 1
                return update

            def invocation_config(state, config):
                """Give each fresh review its own truthful report context identity."""
                if phase == 'work':
                    # A later task's file stays unwritable even if a summary
                    # erroneously tells the worker to advance before approval.
                    backends[('/slug.py', '/test_slug.py')].writable_paths = set(state['dispatch']['files'])
                result = {**config, 'recursion_limit': 100}
                if phase == 'review':
                    result['metadata'] = {
                        **config.get('metadata', {}),
                        'report_history_id': str(uuid4()),
                        'report_history_label': f"Isolated review {state['dispatch']['task_id']} attempt {state['attempts']}",
                    }
                return result

            def correction(result, error):
                """Ask the same role to fix format without granting new authority.

                Retain its completed tool observations, so correction does not
                replay the original invocation or discard its metered work.
                Reviewer task-ID and approval checks remain hard failures. A
                duplicate planner assignment can be corrected, but never executes.
                """
                schema = Review if phase == 'review' else Dispatch
                return {'messages': [*result['messages'], HumanMessage(
                    'Your final handoff failed JSON or current-task validation. '
                    'Correct the reported problem and final handoff; do not repeat completed file operations. '
                    'If /plan.md is missing, write the required plan before returning a work assignment. '
                    'Return exactly the schema fields, with no Markdown or extra fields. '
                    + json.dumps({'schema': schema.model_json_schema(),
                                  'errors': error.errors(include_input=False, include_context=False,
                                                         include_url=False)})
                )]}

            def run(state, config):
                """Meter every correction and bound malformed handoffs to three attempts."""
                request = payload(state)
                role_config = invocation_config(state, config)
                for attempt in range(3):
                    result = agent.invoke(request, role_config, context=directive(state))
                    try:
                        return finish(state, result)
                    except ValidationError as error:
                        if attempt == 2:
                            raise
                        request = correction(result, error)

            async def arun(state, config):
                """Preserve the same authority and routing in asynchronous clients."""
                request = payload(state)
                role_config = invocation_config(state, config)
                for attempt in range(3):
                    result = await agent.ainvoke(request, role_config, context=directive(state))
                    try:
                        return finish(state, result)
                    except ValidationError as error:
                        if attempt == 2:
                            raise
                        request = correction(result, error)

            return RunnableLambda(run, afunc=arun)

        def dispatch(state):
            """The planner response selects work or finish; no inferred intent."""
            return state['dispatch']['action']

        def assess(state):
            """Route the actual review result; every repair receives a new review."""
            if state['review']['verdict'] == 'approve':
                return 'approved'
            return 'repair' if state['attempts'] < max_attempts else 'limit'

        def execute_tests(state):
            """Run current files before review and expose evidence to QA/history.

            A fresh receipt replaces the previous attempt's result. Review,
            repair and planner updates receive it outside compacted history.
            """
            result = run_tests(directory, required='/test_slug.py' in state['dispatch']['files'])
            result.update(task_id=state['dispatch']['task_id'], attempt=state['attempts'])
            return {'test_execution': result,
                    'messages': [HumanMessage('Observed test execution: ' + json.dumps(result))]}

        def finish(state):
            """Publish a readable final answer, preserving unapproved outcomes."""
            if state.get('review', {}).get('verdict') == 'revise':
                text = 'Review limit reached; task ' + state['dispatch']['task_id'] + ' remains incomplete. ' + state['review']['evidence']
                outcome = 'review_limit'
            else:
                text, outcome = state['dispatch']['message'], 'complete'
            return {'messages': [AIMessage(content=text)], 'outcome': outcome}

        # START -> planner -- work -> worker -> tests -> reviewer -- approve -> update
        #             |                 ^          |                    |
        #           finish              +--revise--+            work / finish
        # The responses choose routes; these edges are also the report's source.
        graph = StateGraph(StepState)
        for name, agent, phase in (
            ('planning_step', planning, 'planning'), ('working_step', working, 'work'),
            ('review_step', reviewer, 'review'), ('plan_update_step', planning, 'update'),
        ):
            graph.add_node(name, role_node(agent, phase), metadata={
                'report_history_id': 'isolated-review' if phase == 'review' else 'shared-workflow',
                'report_comment': {'planning': 'Prepare the file plan and assign one step', 'work': 'Implement or repair the assigned step', 'review': 'Review current files in a fresh context', 'update': 'Record the approved step and choose what comes next'}[phase],
            })
        graph.add_node('finish', finish)
        graph.add_node('test_step', execute_tests, metadata={
            'report_comment': 'Execute current unittest suite before independent review'})
        graph.add_edge(START, 'planning_step')
        graph.add_conditional_edges('planning_step', dispatch, {'work': 'working_step', 'finish': 'finish'})
        graph.add_edge('working_step', 'test_step')
        graph.add_edge('test_step', 'review_step')
        graph.add_conditional_edges('review_step', assess, {'approved': 'plan_update_step', 'repair': 'working_step', 'limit': 'finish'})
        graph.add_conditional_edges('plan_update_step', dispatch, {'work': 'working_step', 'finish': 'finish'})
        graph.add_edge('finish', END)
        compiled = graph.compile(name='context_budget_workflow')
        # Wrapper composition references expose real compiled children, not copied edges.
        compiled.report_subgraphs = {'planning_step': planning, 'working_step': working, 'review_step': reviewer, 'plan_update_step': planning}
        compiled.report_context = 'One task: worker, test execution, independent review, planner completion; failures return for repair'
        compiled.workspace, compiled.workspace_dir = workspace, directory
        compiled.plan_file = directory / 'plan.md'
        return compiled
    except BaseException:
        if workspace is not None:
            workspace.cleanup()
        raise
