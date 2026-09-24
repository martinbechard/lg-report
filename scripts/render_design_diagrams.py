"""Render the technical design's editable diagrams as self-contained SVG.

Explicit rows and edges keep teaching diagrams predictable without a browser
layout engine or CDN. Labels describe responsibilities, not implementation traces.
Run this script after changing a diagram to update the embedded HTML figures.
AI assistance: Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import re
from html import escape
from pathlib import Path


# Each row contains (id, title, detail, role). Roles give the same responsibility
# the same visual treatment throughout the document. Edges remain authoritative;
# proximity alone never implies execution or an ownership relationship.
def n(key, title, detail="", role="agent"):
    return (key, title, detail, role)


DRIVER = n("d", "LangGraphAgent", "Start · resume · stream", "driver")


def flow(agent, middle=()):
    """Describe a model-directed tool flow with a visible return to the agent."""
    rows = [[n("c", "User / client", role="client"), DRIVER, n("a", agent)]]
    edges = [("c", "d", "request / response", True), ("d", "a", "run / output", True)]
    if middle:
        rows += [list(middle)]
        edges += [("a", x[0], "call / result", True) for x in middle]
    return rows, edges


DIAGRAMS = []


def add(title, rows, edges, note=""):
    DIAGRAMS.append((title, rows, edges, note))


add(
    "Shared execution architecture",
    [
        [
            n("c", "Console", "Python objects · in process", "client"),
            n("s", "Scripted client", "Python objects · in process", "client"),
            n("b", "Angular", "HttpAgent · HTTP / SSE", "client"),
        ],
        [DRIVER],
        [n("w", "Workflow harness", "Context · approval · routing", "workflow")],
        [n("a", "Agent and tools", "Prompt · model loop · domain tools")],
        [
            n(
                "i",
                "Session lifecycle",
                "Identity · expiry · run protection",
                "support",
            ),
            n("p", "Checkpointer", "State across turns and pauses", "dependency"),
            n("r", "Recorder", "Callbacks → reports / tracing", "support"),
        ],
    ],
    [
        ("c", "d", "AG-UI", True),
        ("s", "d", "AG-UI", True),
        ("b", "d", "AG-UI", True),
        ("d", "w", "execute"),
        ("w", "a", "delegate"),
    ],
    "Supporting components below are composed with execution; they are not additional workflow steps.",
)
add(
    "Interrupt and resume",
    [
        [
            n("c", "Client request", role="client"),
            DRIVER,
            n("w", "Workflow runs", role="workflow"),
        ],
        [n("p", "Interrupt + checkpoint", "Workflow pauses", "workflow")],
        [n("h", "Client presents interaction", "Human answers or cancels", "client")],
        [n("r", "Resume through driver", "Same thread + interrupt ID", "driver")],
        [
            n(
                "v",
                "Workflow continues",
                "Validate answer · next pause or finish",
                "workflow",
            )
        ],
    ],
    [
        ("c", "d", "request / response", True),
        ("d", "w", ""),
        ("w", "p", ""),
        ("p", "h", "interrupted outcome"),
        ("h", "r", "response"),
        ("r", "v", "resume checkpoint"),
    ],
)
add(
    "Simple chat",
    *flow("Chat agent"),
    "Each follow-up uses the same thread and checkpoint; the agent receives retained history.",
)
add(
    "Tool chat",
    *flow(
        "Reference agent",
        [n("t", "Echo tool", "Return supplied text with a prefix", "tool")],
    ),
)
add(
    "Parent and subagent",
    *flow(
        "Parent agent",
        [n("t", "Specialist agent", "Delegated task · separate context")],
    ),
)
add(
    "Expert dispatcher",
    *flow(
        "Dispatcher",
        [n("m", "Movie expert"), n("s", "Sports expert"), n("h", "History expert")],
    ),
    "The dispatcher selects an expert when needed; this is not a mandatory parallel fan-out.",
)
add(
    "Thinking agent",
    *flow(
        "Investigator",
        [
            n("i", "Inspect evidence", "Fictional service evidence", "tool"),
            n("t", "Test plan", "Validate a proposed plan", "tool"),
        ],
    ),
    "The agent can repeat inspection and plan testing before answering.",
)
add(
    "Wikipedia RAG",
    *flow(
        "Wikipedia agent",
        [n("t", "Search tool + Chroma", "Return relevant passages", "tool")],
    ),
)
add(
    "MCP Wikipedia RAG",
    [
        [n("c", "Client request", role="client"), DRIVER, n("a", "MCP RAG agent")],
        [n("m", "MCP server", "Tool request over stdio", "tool")],
        [n("i", "Local index", "Retrieve passages", "tool")],
    ],
    [
        ("c", "d", "request / response", True),
        ("d", "a", "run / output", True),
        ("a", "m", "search / passages", True),
        ("m", "i", "query / results", True),
    ],
)
add(
    "Author and judge",
    [
        [
            n("c", "Client request", role="client"),
            DRIVER,
            n("a", "Author", "Draft or revise"),
        ],
        [n("j", "Judge", "Assess draft")],
        [
            n("v", "Revision route", "Fewer than 3 drafts", "workflow"),
            n("f", "Final outcome", "Approved or draft limit reached", "workflow"),
        ],
    ],
    [
        ("c", "d", "request / response", True),
        ("d", "a", "run / output", True),
        ("a", "j", "draft"),
        ("j", "v", "revise"),
        ("j", "f", "finish"),
        ("v", "a", "revision", False, "left"),
    ],
)
add(
    "File approval",
    [
        [
            n("c", "Request through driver", role="driver"),
            n("a", "Editor agent", "Propose a tool call"),
        ],
        [n("g", "Workflow approval gate", "Inspect proposed action", "workflow")],
        [
            n("read", "Allowed read", "Execute tool", "tool"),
            n("h", "Restricted write", "Pause for human decision", "client"),
        ],
        [
            n("ok", "Approve", "Write with stale-file guard", "tool"),
            n("no", "Reject", "No write", "workflow"),
            n("end", "Cancel", "End workflow", "workflow"),
        ],
        [n("r", "Editor agent", "Receive tool or rejection result")],
    ],
    [
        ("c", "a", ""),
        ("a", "g", ""),
        ("g", "read", "read"),
        ("g", "h", "write"),
        ("h", "ok", ""),
        ("h", "no", ""),
        ("h", "end", ""),
        ("ok", "r", ""),
        ("no", "r", ""),
        ("read", "r", "result", False, "left"),
    ],
    "All restricted calls in a proposed batch are reviewed before tool execution. Cancellation does not undo earlier writes.",
)
add(
    "Quote clarification",
    [
        [
            n("d", "Request through driver", role="driver"),
            n("a", "Quote interpreter", "Assess whole request"),
        ],
        [
            n("q", "Ask human", "Pause with question + reason", "client"),
            n("f", "Complete", "Agreed scope", "workflow"),
        ],
        [
            n("r", "Resume answer", "Retain question and answer", "workflow"),
            n("x", "Cancel", "End workflow", "workflow"),
        ],
    ],
    [
        ("d", "a", "run / output", True),
        ("a", "q", "unclear"),
        ("a", "f", "clear"),
        ("q", "r", "answer"),
        ("q", "x", "cancel"),
        ("r", "a", "reassess", False, "left"),
    ],
)
add(
    "Claims context strategies",
    [
        [
            n("d", "Query through driver", role="driver"),
            n("a", "Claims agent", "Choose tools on demand"),
        ],
        [
            n("rc", "Read claim", role="tool"),
            n("ec", "Edit claim", role="tool"),
            n("rp", "Read policy", role="tool"),
        ],
        [n("w", "Completed agent turn", "Harness detects successful edit", "workflow")],
        [
            n("nv", "Naive strategy", "Retain previous context", "workflow"),
            n("mg", "Managed strategy", "Remove claim-dependent context", "workflow"),
        ],
        [n("f", "Follow-up → claims agent", "Agent chooses any needed reread")],
    ],
    [
        ("d", "a", "run / output", True),
        ("a", "rc", ""),
        ("a", "ec", ""),
        ("a", "rp", ""),
        ("rc", "w", ""),
        ("ec", "w", ""),
        ("rp", "w", ""),
        ("w", "nv", ""),
        ("w", "mg", ""),
        ("nv", "f", ""),
        ("mg", "f", ""),
    ],
    "Managed mode can retain unaffected policy evidence. Display history and audit remain separate from working model context.",
)
add(
    "Simple chat with Langfuse",
    [
        [n("c", "Client", role="client"), DRIVER, n("a", "Chat agent")],
        [n("r", "Langfuse recorder", "Session / turn observations", "support")],
    ],
    [
        ("c", "d", "request / response", True),
        ("d", "a", "run / output", True),
        ("d", "r", "callbacks"),
    ],
)
add(
    "Parent and subagent with Langfuse",
    [
        [n("c", "Client", role="client"), DRIVER, n("a", "Parent agent")],
        [
            n("r", "Langfuse recorder", "Parent / child observations", "support"),
            n("s", "Specialist", "Separate context"),
        ],
    ],
    [
        ("c", "d", "request / response", True),
        ("d", "a", "run / output", True),
        ("a", "s", "task / result", True),
        ("d", "r", "callbacks"),
    ],
)

COLORS = {
    "sample_code": ("#fff0d9", "#b78939"),
    "shared_code": ("#e8f0fc", "#6688b5"),
    "client": ("#eef4fa", "#7296b5"),
    "driver": ("#e0f5f2", "#398579"),
    "dependency": ("#e0f5f2", "#398579"),
    "agent": ("#f6f1fb", "#8462a7"),
    "workflow": ("#eef5ef", "#6c9176"),
    "tool": ("#fff7e9", "#b39659"),
    "support": ("#f4f5f6", "#9098a0"),
}


def render(index, title, rows, edges, note):
    """Route connectors behind fixed-size nodes and keep text inside its box.

    A dedicated outer lane carries backward edges so loops never cut through
    another node. Horizontal scrolling preserves legible type on small screens.
    """
    width, boxw, boxh, step = 960, 250, 72, 142
    height = 36 + (len(rows) - 1) * step + boxh + 28
    pos = {}
    for row, nodes in enumerate(rows):
        for col, node in enumerate(nodes):
            pos[node[0]] = (
                (width - len(nodes) * 290 + 40) / 2 + col * 290,
                30 + row * step,
            )
    marker = f"arrow-{index}"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title-{index}"><title id="title-{index}">{escape(title)}</title><defs><marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#687887"/></marker></defs>'
    ]
    for edge in edges:
        source, target, label, *options = edge
        both = options[0] if options else False
        lane = options[1] if len(options) > 1 else None
        x, y = pos[source]
        u, v = pos[target]
        if lane:
            a, b = x, y + boxh / 2
            c, d = u, v + boxh / 2
            path = f"M {a} {b} H 28 V {d} H {c}"
            tx, ty = 34, (b + d) / 2
        elif y == v:
            a, b = x + boxw, y + boxh / 2
            c, d = u, v + boxh / 2
            path = f"M {a} {b} H {c}"
            tx, ty = (a + c) / 2, y - 12
        else:
            a, b = x + boxw / 2, y + boxh
            c, d = u + boxw / 2, v
            mid = (b + d) / 2
            path = f"M {a} {b} V {mid} H {c} V {d}"
            tx, ty = (a + c) / 2, mid - 10
        start = f' marker-start="url(#{marker})"' if both else ""
        out.append(
            f'<path d="{path}" fill="none" stroke="#687887" stroke-width="1.7" stroke-linejoin="round" marker-end="url(#{marker})"{start}/>'
        )
        if label:
            out.append(
                f'<text x="{tx}" y="{ty}" text-anchor="middle" class="edge-label">{escape(label)}</text>'
            )
    for row in rows:
        for key, label, detail, role in row:
            x, y = pos[key]
            fill, stroke = COLORS[role]
            out.append(
                f'<rect x="{x}" y="{y}" width="{boxw}" height="{boxh}" rx="8" fill="{fill}" stroke="{stroke}"/>'
            )
            out.append(
                f'<text x="{x + boxw / 2}" y="{y + (30 if detail else 42)}" text-anchor="middle" class="node-title">{escape(label)}</text>'
            )
            if detail:
                out.append(
                    f'<text x="{x + boxw / 2}" y="{y + 51}" text-anchor="middle" class="node-detail">{escape(detail)}</text>'
                )
    out.append("</svg>")
    caption = f"<figcaption>{escape(note)}</figcaption>" if note else ""
    return (
        f'<figure class="workflow-diagram" data-design-diagram="{index}"><div class="diagram-scroll" tabindex="0" aria-label="{escape(title)} diagram; scroll horizontally on small screens">'
        + "".join(out)
        + f"</div>{caption}</figure>"
    )


class Drawing:
    """Build accessible SVG with explicit folder boundaries and message routes.

    These diagrams explain architecture and protocols rather than graph topology.
    Explicit coordinates keep text and arrows reviewable, with no layout runtime.
    """

    def __init__(self, index, title, height, width=1100):
        self.index, self.title = index, title
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title-{index}"><title id="title-{index}">{escape(title)}</title><defs><marker id="arrow-{index}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#687887"/></marker></defs>'
        ]

    def text(self, x, y, lines, cls="node-detail", anchor="middle"):
        for offset, line in enumerate(lines.split("\n")):
            self.parts.append(
                f'<text x="{x}" y="{y + offset * 20}" text-anchor="{anchor}" class="{cls}">{escape(line)}</text>'
            )

    def box(self, x, y, w, h, title, detail="", role="support"):
        fill, stroke = COLORS[role]
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}"/>'
        )
        self.text(x + w / 2, y + 27, title, "node-title")
        if detail:
            self.text(x + w / 2, y + 49, detail)

    def folder(self, x, y, w, h, title, role=None):
        # Ownership colors apply to the overview; other diagrams keep neutral folders.
        fill = {"sample_code": "#fffbf4", "shared_code": "#f6f9fe"}.get(role, "#fafbfd")
        stroke = COLORS[role][1] if role else "#aebcca"
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-dasharray="5 3"/>'
        )
        self.text(x + 15, y + 25, title, "node-title", "start")

    def arrow(self, path, label="", x=0, y=0, both=False, dotted=False):
        start = f' marker-start="url(#arrow-{self.index})"' if both else ""
        pattern = ' stroke-dasharray="2 5" stroke-linecap="round"' if dotted else ""
        self.parts.append(
            f'<path d="{path}" fill="none" stroke="#687887" stroke-width="1.7" marker-end="url(#arrow-{self.index})"{start}{pattern}/>'
        )
        if label:
            self.text(x, y, label, "edge-label")

    def finish(self, note=""):
        caption = f"<figcaption>{escape(note)}</figcaption>" if note else ""
        return (
            f'<figure class="workflow-diagram" data-design-diagram="{self.index}"><div class="diagram-scroll" tabindex="0" aria-label="{escape(self.title)}; scroll horizontally on small screens">'
            + "".join(self.parts)
            + "</svg></div>"
            + caption
            + "</figure>"
        )


def architecture(index):
    """Show each source folder once so containment communicates ownership."""
    d = Drawing(index, "Components and source folders", 1985)
    d.folder(20, 15, 625, 110, "src/agent_runtime/", role="shared_code")
    d.box(
        45,
        56,
        575,
        52,
        "__main__.py · shared launcher",
        "Sample selection · configuration · client / reporting mode",
        "shared_code",
    )
    # Both conversation entry points and optional tracing share this boundary.
    d.folder(20, 150, 625, 745, "src/agent_runtime/harness/", role="shared_code")
    d.box(
        40,
        195,
        280,
        65,
        "main() · launch selection",
        "Console · static · web listener",
        "shared_code",
    )
    # Startup selection is distinct from the solid request/response paths below.
    d.box(
        340,
        195,
        280,
        65,
        "SampleCatalog",
        "sample.py · model factory scope",
        "shared_code",
    )
    d.arrow("M340 227 H320", dotted=True)
    d.arrow("M180 108 V195")
    d.arrow("M100 260 V280 H180 V300", dotted=True)
    d.arrow("M240 260 V280 H480 V300", dotted=True)
    d.box(40, 300, 280, 70, "ConsoleClient", "ConsoleApplication · input / output", "shared_code")
    d.box(
        340,
        300,
        280,
        70,
        "ScriptPrompter",
        "Authored requests for ConsoleClient",
        "shared_code",
    )
    d.box(
        40,
        410,
        280,
        70,
        "Conversation",
        "execute_conversation()",
        "shared_code",
    )
    d.box(
        340,
        410,
        280,
        70,
        "WebConversation",
        "Run status / pending interrupts",
        "shared_code",
    )
    d.box(
        155,
        700,
        350,
        75,
        "create_langgraph_agent()",
        "LangGraphAgent · AG-UI run / resume / events",
        "driver",
    )
    d.box(
        340,
        805,
        280,
        70,
        "LangfuseCapture",
        "Optional tracing lifecycle",
        "shared_code",
    )
    d.box(
        40,
        805,
        280,
        70,
        "TraceCapture",
        "Runtime callbacks → spans.jsonl",
        "shared_code",
    )
    d.arrow("M180 775 V805", "Local capture", 105, 795)
    # The web conversation supplies the optional Langfuse callback to the driver.
    d.arrow("M480 775 V805", "Optional callback", 555, 795, dotted=True)
    d.arrow("M180 370 V410", both=True)
    d.arrow("M340 335 H320", both=True)
    d.arrow("M40 460 H30 V735 H155", "AG-UI", 85, 665, True)
    d.arrow("M620 460 H630 V735 H505", "AG-UI", 575, 665, True)
    d.folder(695, 15, 385, 110, "frontend/src/", role="shared_code")
    d.box(
        715,
        56,
        345,
        52,
        "ChatStore + HttpAgent",
        "WebScriptPrompter · Angular presentation",
        "shared_code",
    )
    d.folder(695, 255, 385, 505, "src/agent_runtime/web/", role="shared_code")
    d.box(
        715,
        300,
        345,
        70,
        "create_app() HTTP handlers",
        "Validate requests / manage sessions",
        "shared_code",
    )
    # Draw this cross-folder startup edge after the web container so its
    # background cannot obscure the dotted arrowhead at the HTTP handlers.
    d.arrow("M320 245 H330 V275 H675 V320 H715", dotted=True)
    d.box(
        715,
        485,
        345,
        75,
        "FastAPI",
        "Application instance · uvicorn.run(app)",
        "dependency",
    )
    d.arrow("M888 370 V485", "creates app", 970, 470)
    d.box(
        715,
        600,
        345,
        95,
        "AG-UI protocol / SSE",
        "RunAgentInput · UserMessage\nEventEncoder + StreamingResponse",
        "dependency",
    )
    d.arrow("M1060 350 H1070 V645 H1060")
    d.text(888, 727, "No direct LangGraph / DeepAgents classes")
    d.arrow("M930 108 V300", "AG-UI / HTTP / SSE", 980, 185, True)
    d.arrow("M715 350 H655 V445 H620", "AG-UI", 665, 400, True)
    d.folder(20, 960, 625, 145, "src/agent_runtime/workflows/", role="sample_code")
    d.box(
        65,
        1002,
        535,
        80,
        "Custom workflow for each sample",
        "Process / routing / context strategy\nReferences: model · backend · middleware",
        "sample_code",
    )
    d.folder(695, 1140, 385, 140, "src/agent_runtime/middleware/", role="shared_code")
    d.box(
        715,
        1185,
        345,
        80,
        "RestrictedToolApproval",
        "Approval / cancellation / stale-content checks",
        "shared_code",
    )
    d.arrow(
        "M600 1080 H665 V1225 H715", "configures middleware", 810, 1125, dotted=True
    )
    # Keep the execution route clear of the optional recorder inside harness.
    d.arrow("M330 775 V1002", "execute / output", 420, 947, True)
    d.folder(20, 1290, 625, 140, "src/agent_runtime/agents/", role="sample_code")
    d.box(
        65,
        1330,
        535,
        90,
        "Custom agents for each sample",
        "Conceptual role / instructions / tools / model loop\nWorkflow supplies configuration; agents add prompts and tools",
        "sample_code",
    )
    # Construction passes references; the adjacent arrow retains runtime flow.
    d.arrow("M200 1082 V1330", "parameter dictionaries", 175, 1210, dotted=True)
    d.arrow("M460 1082 V1330", "invoke / result", 525, 1210, True)
    # Shared backend adapters sit beside their workflow configuration owner.
    # Agents receive these references through parameter dictionaries.
    d.folder(695, 790, 385, 315, "src/agent_runtime/backends/", role="shared_code")
    d.box(
        715,
        835,
        345,
        100,
        "FileAccessBackend",
        "file_access_backend.py\nSource / target file restrictions",
        "shared_code",
    )
    d.box(
        715,
        975,
        345,
        100,
        "ShellBackend",
        "shell_backend.py\nLocal commands + native file I/O",
        "shared_code",
    )
    d.arrow("M600 1042 H695", "references", 648, 1030, dotted=True)
    d.folder(
        20,
        1445,
        625,
        370,
        "src/agent_runtime/tools/ · 5 modules / 10 tools",
        role="sample_code",
    )
    d.box(
        40,
        1490,
        280,
        115,
        "search_reference.py · 3 tools",
        "search_movie_reference\nsearch_sports_reference\nsearch_history_reference",
        "sample_code",
    )
    d.box(
        340,
        1490,
        280,
        115,
        "claims.py · 3 tools",
        "read_claim\nedit_claim\nread_policy",
        "sample_code",
    )
    d.box(
        40,
        1625,
        280,
        90,
        "service_evidence.py · 2 tools",
        "inspect_service\ntest_plan",
        "sample_code",
    )
    d.box(
        340,
        1625,
        280,
        90,
        "echo_tool.py · 1 tool",
        "echo_tool",
        "sample_code",
    )
    d.box(
        340,
        1735,
        280,
        65,
        "semantic_search_wikipedia.py · 1 tool",
        "semantic_search_wikipedia",
        "sample_code",
    )
    # This edge addresses the tool package; each agent registers its own subset.
    d.arrow("M330 1420 V1445", "call / result", 405, 1435, True)
    # Library dependency is positioned near its owners, not an application class.
    d.box(
        155,
        540,
        350,
        90,
        "InMemorySaver · Checkpointer",
        "LangGraph dependency\nOne instance retained per conversation",
        "dependency",
    )
    d.arrow("M180 480 V540", "create / retain", 245, 515)
    d.arrow("M480 480 V540", "create / retain", 415, 515)
    # The harness driver binds existing persistence; workflow nodes do not call it.
    d.arrow("M330 700 V630", "bind to graph", 405, 670)
    d.folder(695, 1290, 385, 140, "src/agent_runtime/mcp_servers/", role="sample_code")
    d.box(
        715,
        1330,
        345,
        90,
        "Wikipedia FastMCP server",
        "semantic_search_wikipedia tool endpoint\nLocal retrieval index",
        "sample_code",
    )
    # The MCP agent owns the adapter; the server exposes the shared search tool.
    d.arrow("M600 1375 H715", "MCP / stdio", 657, 1364, True)
    # Route MCP outside the file-editor components so ownership stays clear.
    d.arrow(
        "M1060 1375 H1088 V1830 H630 V1770 H620",
        "MCP tool call / result",
        835,
        1820,
        True,
    )
    d.box(
        715,
        1520,
        345,
        90,
        "DeepAgent native tools",
        "read_file · write_file · edit_file · execute\nStorage / execution through the backend",
        "dependency",
    )
    d.arrow("M600 1410 H675 V1565 H715", "native tools", 760, 1505, True)
    # Native tools use the workflow-selected adapters at execution time.
    d.arrow("M1060 1565 H1095 V885 H1060")
    d.arrow("M1095 1025 H1060")
    # Keep the ownership key below every component and connection.
    d.text(20, 1855, "Color key · implementation ownership", "node-title", "start")
    d.box(
        20,
        1875,
        330,
        80,
        "Sample-specific code",
        "Develop or adapt for each sample",
        "sample_code",
    )
    d.box(
        385,
        1875,
        330,
        80,
        "Shared application code",
        "Reuse across samples",
        "shared_code",
    )
    d.box(
        750,
        1875,
        330,
        80,
        "Library dependencies",
        "LangGraph / AG-UI / FastAPI",
        "dependency",
    )
    return d.finish(
        "Workflows pass model, backend, and middleware references through parameter dictionaries (dotted arrow). Agents own domain tools and role instructions. Movies, sports, and history share one reference-expert implementation with independent role definitions."
    )


def sequence(index, title, actors, messages, note=""):
    """Show each participant once and order exchanges vertically over time."""
    width = 1100
    xs = [140 + i * 820 / (len(actors) - 1) for i in range(len(actors))]
    d = Drawing(index, title, 150 + len(messages) * 76, width)
    for x, (name, detail, role) in zip(xs, actors, strict=True):
        d.parts.append(
            f'<path d="M{x} 90 V{115 + len(messages) * 76}" stroke="#bdc8d1" stroke-dasharray="5 5"/>'
        )
        d.box(x - 125, 18, 250, 72, name, detail, role)
    for row, (a, b, label) in enumerate(messages):
        y = 135 + row * 76
        if a == b:
            label_x = xs[a] - 30 if a == len(actors) - 1 else xs[a] + 65
            d.arrow(f"M{xs[a]} {y} h70 v28 h-70", label, label_x, y - 12)
        else:
            d.arrow(f"M{xs[a]} {y} H{xs[b]}", label, (xs[a] + xs[b]) / 2, y - 12)
    return d.finish(note)


def interrupt_sequence(index):
    return sequence(
        index,
        "Clarification and workflow resume",
        [
            ("User / client", "Console or Angular", "client"),
            ("LangGraphAgent", "Shared execution driver", "driver"),
            ("Custom workflow + agent", "State saved by InMemorySaver", "workflow"),
        ],
        [
            (0, 1, "Submit task"),
            (1, 2, "Start graph"),
            (2, 2, "Agent works until input is needed"),
            (2, 1, "interrupt(question) · checkpoint state"),
            (1, 0, "Present question"),
            (0, 1, "Answer + thread / interrupt IDs"),
            (1, 2, "Resume checkpoint with answer"),
            (2, 2, "Agent reassesses request"),
            (2, 1, "Answer or next question"),
            (1, 0, "Present result / interaction"),
        ],
    )


def approval_sequence(index):
    return sequence(
        index,
        "Harness-triggered approval",
        [
            ("Agent / LLM", "Chooses tool + arguments", "agent"),
            ("RestrictedToolApproval", "after_model() middleware", "workflow"),
            ("User / client", "Approval presentation", "client"),
            ("Native file tool", "Bounded backend / stale guard", "tool"),
        ],
        [
            (0, 1, "Propose write_file / edit_file"),
            (1, 1, "Match RESTRICTED_TOOLS"),
            (1, 2, "Interrupt: exact action + arguments"),
            (2, 1, "Resume: approve / reject / cancel"),
            (1, 3, "If approved: release call for execution"),
            (3, 0, "Tool result → agent continues"),
            (1, 0, "If rejected: return rejection result"),
            (1, 2, "If cancelled: end workflow"),
        ],
        "These are alternative outcomes. The harness checks every restricted call before tools execute; read_file does not require approval.",
    )


def claims_sequence(index):
    """Align operations with store and working-context snapshots below each step."""
    d = Drawing(index, "Claims: edits, purges, and fresh reads", 665, 1400)
    steps = [
        (
            "1 · Read claim",
            "Agent calls read_claim",
            "Claim v1",
            "Snapshot v1\n+ any policy evidence",
            "Snapshot v1",
        ),
        (
            "2 · Edit → end turn",
            "Successful edit; managed purge",
            "Claim v2",
            "Policy evidence + notice\nNo claim snapshot",
            "Snapshot v1\n+ edit args / receipt v2",
        ),
        (
            "3 · Read → edit → end turn",
            "Read v2; edit to v3; purge",
            "Claim v3",
            "Policy evidence + notice\nNo claim snapshot",
            "Earlier evidence\n+ read v2 + edit receipt v3",
        ),
        (
            "4 · Follow-up read",
            "Agent calls read_claim",
            "Claim v3",
            "Fresh snapshot v3\n+ retained policy evidence",
            "Earlier evidence\n+ fresh snapshot v3",
        ),
    ]
    for i, (title, detail, store, managed, naive) in enumerate(steps):
        x = 190 + i * 295
        d.box(x, 30, 275, 85, title, detail, "agent")
        if i < 3:
            d.arrow(f"M{x + 275} 72 H{x + 295}")
        d.box(x, 165, 275, 65, store, role="tool")
        d.box(x, 275, 275, 105, "Managed context", managed, "workflow")
        d.box(x, 420, 275, 105, "Naive context", naive, "support")
    d.text(20, 195, "Current store", "node-title", "start")
    d.text(20, 305, "Working context\nafter the step", "node-title", "start")
    d.text(20, 450, "Working context\nafter the step", "node-title", "start")
    d.box(
        190,
        570,
        1160,
        65,
        "Transcript / audit: historical evidence is retained separately",
        "Content depends on capture policy. It is never automatically replayed into managed working context.",
        "support",
    )
    return d.finish(
        "Agent instructions are supplied on every invocation. Managed purge occurs at the end of an edited turn, not immediately after each edit tool call. A fresh read supplies current data; it does not merge old snapshots."
    )


def protocol_sequence(index):
    return sequence(
        index,
        "AG-UI requests and events",
        [
            ("Client", "Conversation or HttpAgent", "client"),
            ("LangGraphAgent", "AG-UI driver", "driver"),
            ("Compiled LangGraph", "Custom workflow / agents / tools", "workflow"),
        ],
        [
            (0, 1, "RunAgentInput: message + thread ID + run ID"),
            (1, 2, "Execute graph"),
            (2, 1, "Model output / tool activity / state"),
            (1, 0, "AG-UI message, tool, and state events"),
            (2, 1, "Interrupt or completion"),
            (1, 0, "Run outcome: paused or complete"),
            (0, 1, "If paused: resume entry with answer"),
            (1, 2, "Continue checkpointed graph"),
        ],
        "Console / scripted clients exchange Python objects in process. Angular sends HTTP requests and receives the same event types over SSE. AG-UI carries interaction data; the custom workflow owns policy.",
    )


# Complex flows use sequence diagrams: one lifeline per participant makes return
# messages unambiguous and avoids drawing the same agent as separate components.
CUSTOM = {
    0: architecture,
    1: interrupt_sequence,
    15: approval_sequence,
    16: claims_sequence,
    17: protocol_sequence,
    12: lambda i: sequence(
        i,
        "Claims agent and custom context workflow",
        [
            ("User / driver", "Requests and answers", "client"),
            ("ClaimsContext", "Select working context", "workflow"),
            ("ClaimsAgent", "Choose tools on demand", "agent"),
            ("ClaimStore tools", "Current claim and policy", "tool"),
        ],
        [
            (0, 1, "Submit query"),
            (1, 2, "Working context + new message"),
            (2, 3, "Read claim / read policy, when needed"),
            (3, 2, "Current record / policy evidence"),
            (2, 3, "Edit claim, if requested"),
            (3, 2, "Edit receipt with new revision"),
            (2, 1, "Completed agent turn"),
            (1, 1, "Apply naive or managed strategy"),
            (1, 0, "Answer; await follow-up"),
        ],
        "The workflow retains all working messages in naive mode. After a successful edited turn, managed mode removes claim-dependent context. A follow-up repeats this exchange with the retained working context.",
    ),
    9: lambda i: sequence(
        i,
        "Author and judge",
        [
            ("User / driver", "Request and final outcome", "client"),
            ("Author", "Own prompt and draft history", "agent"),
            ("Judge", "Own assessment instructions", "agent"),
        ],
        [
            (0, 1, "Submit writing task"),
            (1, 2, "Draft for assessment"),
            (2, 1, "Revision feedback, if needed"),
            (1, 2, "Revised draft (maximum 3 drafts total)"),
            (2, 0, "Workflow returns approved or limit-reached outcome"),
        ],
        "The custom workflow routes drafts and feedback and enforces the draft limit. Judge and author do not contact the user directly.",
    ),
    10: lambda i: sequence(
        i,
        "Native file tools with approval",
        [
            ("User / driver", "Task and approval responses", "client"),
            ("File editor", "LangChain create_agent()", "agent"),
            ("Approval middleware", "Review / stale-content guard", "workflow"),
            (
                "Native file tools",
                "FileAccessBackend → FilesystemBackend",
                "dependency",
            ),
        ],
        [
            (0, 1, "Request file edit"),
            (1, 3, "read_file: /source.txt or /target.txt"),
            (3, 2, "Target read: checkpoint observed content"),
            (3, 1, "Native read result"),
            (1, 2, "Propose write_file or edit_file"),
            (2, 0, "Pause: exact arguments + prior content"),
            (0, 2, "Resume: approve / reject / cancel"),
            (2, 3, "Approved: check stale content, then execute"),
            (3, 1, "Native tool result; reread before next edit"),
            (2, 1, "Rejected: result without executing tool"),
            (2, 0, "Cancelled: end; retain earlier writes"),
            (1, 0, "Answer after work completes"),
        ],
        "Approval, rejection, and cancellation are alternative paths. FileAccessBackend exposes only the configured source and target and permits mutations only on the target. Native FilesystemBackend performs the file operations. The stale-content check and write are not atomic.",
    ),
    18: lambda i: sequence(
        i,
        "Shell script through the native execute tool",
        [
            ("User / driver", "Request script execution", "client"),
            ("Shell DeepAgent", "Native execute tool", "agent"),
            ("ShellBackend", "SandboxBackendProtocol", "sample_code"),
            ("Local shell script", "Prepared working directory", "tool"),
        ],
        [
            (0, 1, "Run the requested script"),
            (1, 2, "execute(command)"),
            (2, 3, "sh -c command"),
            (3, 2, "Combined output + exit code"),
            (2, 1, "ExecuteResponse; timeout / truncation if needed"),
            (1, 0, "Answer based on actual tool output"),
        ],
        "ShellBackend lives in backends/shell_backend.py and reuses FilesystemBackend for native file tools. Commands run locally as the current user; a temporary working directory is not an OS sandbox. This lesson has no approval gate.",
    ),
    11: lambda i: sequence(
        i,
        "Quote clarification",
        [
            ("User / driver", "Question and answer exchange", "client"),
            (
                "Custom quote workflow",
                "Retain answers / interrupt / resume",
                "workflow",
            ),
            ("Quote interpreter", "Assess the whole request", "agent"),
        ],
        [
            (0, 1, "Submit request"),
            (1, 2, "Assess request + retained answers"),
            (2, 1, "Clarification question and reason"),
            (1, 0, "Interrupt: ask human"),
            (0, 1, "Resume with answer (or cancel)"),
            (1, 2, "Reassess with answer"),
            (2, 1, "Complete or ask again"),
            (1, 0, "Agreed scope or next question"),
        ],
    ),
}


def main():
    """Replace only diagram blocks, preserving the document's hand-written prose."""
    page = Path(__file__).resolve().parents[1] / "docs/unified-execution-design.html"
    source = page.read_text()
    pattern = (
        r'<figure class="workflow-diagram" data-design-diagram="(\d+)".*?</figure>'
    )

    def replace(match):
        index = int(match.group(1))
        if index in CUSTOM:
            return CUSTOM[index](index)
        return render(index, *DIAGRAMS[index])

    updated, count = re.subn(pattern, replace, source, flags=re.DOTALL)
    assert count == 19, f"Expected 19 diagram slots, found {count}"
    updated = updated.replace(
        "Diagrams are embedded text, with no CDN or diagram-rendering dependency.",
        'Diagrams are embedded SVG, generated from the editable definitions in <a href="../scripts/render_design_diagrams.py" target="_blank" rel="noopener noreferrer">render_design_diagrams.py</a>. Viewing the page requires no CDN or diagram-rendering library.',
    )
    page.write_text(updated)


if __name__ == "__main__":
    main()
