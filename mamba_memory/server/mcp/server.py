"""MCP Server — exposes MambaMemory as MCP tools.

Any MCP-compatible client (Claude Code, OpenClaw, Codex, etc.)
can connect to this server and use the memory tools directly.

Usage:
    mamba-memory serve --mcp
    # or
    python -m mamba_memory.server.mcp.server
"""

from __future__ import annotations

import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import GetPromptResult, Prompt, PromptArgument, PromptMessage, TextContent, Tool

from mamba_memory.core.engine import MambaMemoryEngine
from mamba_memory.core.types import EngineConfig

# ---------------------------------------------------------------------------
# Usage guide (injected as MCP prompt for LLM clients)
# ---------------------------------------------------------------------------

USAGE_GUIDE = """\
# MambaMemory Usage Guide (for LLM clients)

You are connected to MambaMemory, a cognitive memory engine. No API keys needed — \
you ARE the LLM, so you do the compression yourself.

## How to remember things

When the user says something worth remembering (decisions, preferences, facts, corrections):

1. Summarize the key information yourself
2. Extract entity tags
3. Call memory_ingest with both:

```
memory_ingest({
    "content": "<raw user message>",
    "summary": "<your concise summary>",
    "entities": ["entity1", "entity2"],
    "tags": ["entity1", "entity2"]
})
```

The `summary` field is the key — it goes directly into the cognitive state layer (L2) \
without needing any internal LLM call. You are the compressor.

## How to recall things

Before answering questions that might need past context:

```
memory_recall({"query": "what database are we using"})
```

## What gets auto-filtered

If you omit `summary`, the raw content goes into L1 (recent window). When the window \
overflows, a built-in compressor handles it — but it's basic without an LLM. \
Providing your own summary gives much better results.

## When to force-store

Use `force: true` for explicit user requests ("remember this", "don't forget"):

```
memory_ingest({
    "content": "API keys rotate every 90 days",
    "summary": "API key rotation policy: every 90 days",
    "entities": ["API keys", "security"],
    "force": true
})
```
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="memory_ingest",
        description=(
            "Write content into the MambaMemory cognitive memory system. "
            "Content passes through a selective gate that evaluates novelty "
            "and importance to decide whether/how to store it. "
            "Use this to remember important facts, decisions, and preferences. "
            "You can optionally provide a pre-compressed summary to skip "
            "internal compression (recommended when running inside an LLM client)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to remember",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity tags (people, projects, tools)",
                },
                "force": {
                    "type": "boolean",
                    "description": "Force store, bypassing the selective gate",
                    "default": False,
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Pre-compressed summary of the content. When provided, "
                        "this is written directly to L2 state layer, bypassing "
                        "internal LLM compression. Recommended when the caller "
                        "is an LLM (Claude, GPT, etc.) that can summarize itself."
                    ),
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Extracted entities from the content (people, projects, "
                        "tools, concepts). Used with summary for direct L2 write."
                    ),
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="memory_recall",
        description=(
            "Recall relevant memories from the cognitive memory system. "
            "Searches across three layers: recent conversation (L1), "
            "active cognitive state (L2), and long-term persistent storage (L3). "
            "Results are ranked by semantic similarity, activation level, and recency."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to recall — describe the topic or ask a question",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return",
                    "default": 5,
                },
                "layers": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["l1", "l2", "l3"]},
                    "description": "Which layers to search (default: all)",
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum relevance score (0-1)",
                    "default": 0.0,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_forget",
        description=(
            "Forget memories matching a description. Removes from active "
            "cognitive state (L2) and archives in persistent storage (L3)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of what to forget",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_status",
        description=(
            "Get the current status of the memory system: "
            "how many turns in the window, active cognitive slots, "
            "total persistent records, etc."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "detail": {
                    "type": "string",
                    "enum": ["summary", "slots", "full"],
                    "description": "Level of detail",
                    "default": "summary",
                },
            },
        },
    ),
    Tool(
        name="memory_compact",
        description=(
            "Manually trigger memory compaction. "
            "Evicts low-activation L2 slots to L3 and/or archives old L3 records."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "enum": ["l2", "l3", "all"],
                    "description": "Which layer(s) to compact",
                    "default": "all",
                },
            },
        },
    ),
]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def create_server(engine: MambaMemoryEngine) -> Server:
    """Create an MCP server wired to the given engine."""
    server = Server("mamba-memory")

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name="memory-compress",
                description=(
                    "Compress a conversation into a structured summary for memory storage. "
                    "Use this prompt to generate a summary + entities, then pass the result "
                    "to memory_ingest with the summary and entities fields."
                ),
                arguments=[
                    PromptArgument(
                        name="conversation",
                        description="The conversation text to compress",
                        required=True,
                    ),
                ],
            ),
            Prompt(
                name="memory-usage-guide",
                description="How to use MambaMemory tools effectively as an LLM client",
            ),
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
        if name == "memory-compress":
            conversation = (arguments or {}).get("conversation", "")
            return GetPromptResult(
                description="Compress conversation for memory storage",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=(
                                "Compress the following conversation into a concise summary "
                                "for long-term memory storage.\n\n"
                                "Rules:\n"
                                "1. PRESERVE: decisions, preferences, facts, corrections, "
                                "names, numbers, configurations\n"
                                "2. DISCARD: greetings, filler, repetition, verbose explanations\n"
                                "3. EXTRACT: key entities (people, projects, tools, concepts)\n"
                                "4. Keep it factual and third-person\n\n"
                                "Output JSON:\n"
                                '{"summary": "...", "entities": ["..."]}\n\n'
                                f"Conversation:\n{conversation}"
                            ),
                        ),
                    ),
                ],
            )

        if name == "memory-usage-guide":
            return GetPromptResult(
                description="MambaMemory usage guide for LLM clients",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=USAGE_GUIDE,
                        ),
                    ),
                ],
            )

        raise ValueError(f"Unknown prompt: {name}")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            return await _dispatch_tool(name, arguments)
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e), "tool": name}, ensure_ascii=False),
            )]

    async def _dispatch_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "memory_ingest":
            result = await engine.ingest(
                arguments["content"],
                tags=arguments.get("tags"),
                force=arguments.get("force", False),
                pre_summary=arguments.get("summary"),
                pre_entities=arguments.get("entities"),
            )
            return [TextContent(type="text", text=json.dumps(result.model_dump(), ensure_ascii=False))]

        if name == "memory_recall":
            result = await engine.recall(
                arguments["query"],
                limit=arguments.get("limit", 5),
                layers=arguments.get("layers"),
                min_score=arguments.get("min_score", 0.0),
            )
            data = {
                "count": len(result.memories),
                "total_tokens": result.total_tokens,
                "memories": [m.model_dump() for m in result.memories],
            }
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]

        if name == "memory_forget":
            count = await engine.forget(arguments["query"])
            return [TextContent(type="text", text=json.dumps({"forgotten": count}))]

        if name == "memory_status":
            detail = arguments.get("detail", "summary")
            status = engine.status()
            data = status.model_dump()

            if detail in ("slots", "full"):
                active_slots = engine.l2.get_active_slots()
                data["slots"] = [
                    {
                        "id": s.id,
                        "topic": s.topic,
                        "activation": round(s.activation, 3),
                        "state_preview": s.state[:100],
                        "entities": s.entities,
                    }
                    for s in active_slots[:20]
                ]

            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]

        if name == "memory_compact":
            result = await engine.compact(arguments.get("layer", "all"))
            return [TextContent(type="text", text=json.dumps(result))]

        return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]

    return server


async def serve(db_path: str | None = None) -> None:
    """Run the MCP server over stdio."""
    config = EngineConfig()
    if db_path:
        config.l3.db_path = db_path

    engine = MambaMemoryEngine(config)
    await engine.start()

    server = create_server(engine)

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await engine.shutdown()
