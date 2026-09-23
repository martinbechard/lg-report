"""Inspect retained model context without running a graph, tool, or provider.

Checkpoint messages are authoritative after a turn, including context purges.
A transient callback remembers the entry agent's actual instructions and tool
schemas, which are not necessarily present in that checkpoint. This preview
uses that observed configuration; next-turn middleware may still change it.
Nothing here opts report files into content capture or counts provider usage.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from copy import deepcopy

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.utils import count_tokens_approximately

from agent_runtime.harness.demo_meter import message_record
from reporting.context import context_capacity


class ContextPreview(BaseCallbackHandler):
    """Remember only the entry model's envelope, never the last child context.

    The first model call identifies the entry agent for a turn. Its system
    messages and advertised tools are copied in memory; history instead comes
    from the checkpoint after all workflow retention policies have completed.
    """

    def __init__(self):
        self.envelope = None
        self.observed_this_turn = False

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Observe configuration without changing prompts or saving it to disk."""
        if self.observed_this_turn:
            return
        self.observed_this_turn = True
        parameters = kwargs.get("invocation_params") or {}
        self.envelope = {
            "system": deepcopy(
                [m for batch in messages for m in batch if m.type == "system"]
            ),
            "tools": deepcopy(
                parameters.get("tools", parameters.get("tool_definitions"))
            ),
        }

    def describe(self, history, provider, model, prices, *, prepared=None):
        """Combine retained history with observed instructions, estimating tokens.

        UTF-8 bytes measure this application's JSON, not provider wire framing.
        Tokens use LangChain's local approximation plus tool-schema characters;
        no tokenization API call, hidden prompt, or paid generation is performed.
        """
        if self.envelope is None and prepared is None:
            return {
                "bytes": None,
                "utilization": None,
                "model": model,
                "messages": [],
                "tools": None,
                "note": "Available after the first model call; instructions and tools have not been observed yet.",
            }
        # Some workflows retain system notices. Keep these in place and avoid
        # duplicating one already captured as part of the model's system prefix.
        envelope = self.envelope or {"system": [], "tools": None}
        system = [] if prepared is not None else envelope["system"]
        retained = list(prepared["messages"] if prepared is not None else history)
        prefix = [
            m
            for m in system
            if not any(message_record(m) == message_record(item) for item in retained)
        ]
        messages = [*prefix, *retained]
        tools = prepared["tools"] if prepared is not None else envelope["tools"]
        records = [message_record(m) for m in messages]
        payload = {"messages": records, "tools": tools}
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=str
        )
        tokens = count_tokens_approximately(messages)
        if tools:
            tokens += round(len(json.dumps(tools, ensure_ascii=False)) / 4)
        capacity = context_capacity(provider, model, prices)
        utilization = (
            None
            if capacity is None
            else {
                **capacity,
                "tokens": tokens,
                "percent": tokens / capacity["capacity"] * 100,
            }
        )
        return {
            "bytes": len(encoded.encode("utf-8")),
            "utilization": utilization,
            "model": model,
            "messages": records,
            "tools": tools,
            "note": "Retained history plus the entry agent's last observed instructions and tools. Excludes your next message and queued attachments. Next-turn middleware may change this context; tokens are estimated.",
        }
