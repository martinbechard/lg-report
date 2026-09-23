# Agent construction

DeepAgents' `create_deep_agent` builds an agent with standard middleware and tools.

```python
from deepagents import create_deep_agent

agent = create_deep_agent(model=model)
```

This constructor includes DeepAgents' default middleware and tools, including filesystem and
delegation. Their definitions contribute to model input even when no tool is called.
Estimates should reflect that actual configuration.

## Workflow configuration

The workflow selects dependencies and passes a plain dictionary to each agent:

```python
parameters = {
    "model": model,
    "backend": backend,
    "middleware": [approval_middleware],
    "checkpointer": checkpointer,
}
agent = build_agent(parameters)
```

These variables are instances created or selected by the workflow. Include only
settings needed by the receiving agent. Each independently constructed agent
receives its own configuration from the workflow.

## Defining tools

The agent constructor adds its instructions and domain tools:

```python
from deepagents import create_deep_agent
from agent_runtime.tools.echo_tool import echo_tool


def build_agent(parameters: dict):
    return create_deep_agent(**parameters, tools=[echo_tool])
```

`**parameters` unpacks dictionary entries as keyword arguments: `{"model": model,
"backend": backend}` becomes `model=model, backend=backend`. Middleware and
checkpointers are passed the same way. Do not repeat a keyword already in the
dictionary; Python rejects duplicate arguments.

## Focused LangChain agents

When a role needs only specified tools or structured output, use LangChain's
native `create_agent`. With no tools or middleware supplied, it adds neither:

```python
from langchain.agents import create_agent

agent = create_agent(model=model)
```

LangChain has no `backend` keyword. The workflow supplies a backend through the
DeepAgents middleware that uses it:

```python
from deepagents.middleware.filesystem import FilesystemMiddleware

parameters = {
    "model": model,
    "middleware": [
        FilesystemMiddleware(backend=backend, tools=["read_file"]),
    ],
}
agent = create_agent(**parameters)
```

The file-approval workflow uses this pattern with read, write, and edit tools,
its approval middleware, and a checkpointer. This also keeps delegation out of
that editor, where a child must not bypass approval. The author, judge, and quote
roles use LangChain agent loops with workflow-supplied middleware. Their adapters
translate domain inputs and outputs. The outer LangGraph workflows own
routing, retained context, and human interaction.

Tool definitions cost input tokens even when unused. Select a native constructor
and middleware appropriate to the lesson, and account for what they actually
send. Live reports retain provider usage when available; missing usage and prices
remain explicit.
