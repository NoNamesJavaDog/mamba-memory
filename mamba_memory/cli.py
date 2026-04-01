"""CLI entry point for MambaMemory.

Commands:
    mamba-memory init                 # Interactive setup
    mamba-memory serve --mcp          # Start MCP server (stdio)
    mamba-memory serve --http         # Start HTTP API server
    mamba-memory status               # Show memory status
    mamba-memory compact              # Trigger compaction
    mamba-memory export               # Export all memories as JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mamba_memory.config import DEFAULT_CONFIG_PATH, load_config, save_config
from mamba_memory.core.engine import MambaMemoryEngine
from mamba_memory.core.types import EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mamba-memory",
        description="MambaMemory — AI Agent Cognitive Memory Engine",
    )
    parser.add_argument("--config", "-c", default=None, help="Config file path (YAML)")
    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Interactive setup — create config file")

    # serve
    serve_parser = sub.add_parser("serve", help="Start a memory server")
    serve_parser.add_argument("--mcp", action="store_true", help="MCP server over stdio")
    serve_parser.add_argument("--http", action="store_true", help="HTTP REST API server")
    serve_parser.add_argument("--port", type=int, default=8420, help="HTTP port (default 8420)")
    serve_parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    serve_parser.add_argument("--db", default=None, help="Database path override")

    # status
    status_parser = sub.add_parser("status", help="Show memory system status")
    status_parser.add_argument("--db", default=None, help="Database path override")
    status_parser.add_argument("--detail", choices=["summary", "slots", "full"], default="summary")

    # compact
    compact_parser = sub.add_parser("compact", help="Trigger memory compaction")
    compact_parser.add_argument("--db", default=None, help="Database path override")
    compact_parser.add_argument("--layer", choices=["l2", "l3", "all"], default="all")

    # export
    export_parser = sub.add_parser("export", help="Export all memories as JSON")
    export_parser.add_argument("--db", default=None, help="Database path override")
    export_parser.add_argument("-o", "--output", default="-", help="Output file (- for stdout)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "init":
        _cmd_init()
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "status":
        asyncio.run(_cmd_status(args))
    elif args.command == "compact":
        asyncio.run(_cmd_compact(args))
    elif args.command == "export":
        asyncio.run(_cmd_export(args))


def _load(args: argparse.Namespace) -> EngineConfig:
    """Load config from file + CLI overrides."""
    config = load_config(getattr(args, "config", None))
    db = getattr(args, "db", None)
    if db:
        config.l3.db_path = db
    return config


# -- init -------------------------------------------------------------------

def _cmd_init() -> None:
    """Interactive setup wizard."""
    print("MambaMemory Setup")
    print("=" * 40)
    print()

    # Embedding provider
    print("Embedding provider:")
    print("  1. google   (Gemini, recommended)")
    print("  2. openai   (text-embedding-3)")
    print("  3. local    (sentence-transformers, no API)")
    print("  4. dummy    (testing only)")
    choice = input("Choose [1]: ").strip() or "1"
    provider_map = {"1": "google", "2": "openai", "3": "local", "4": "dummy"}
    embedding = provider_map.get(choice, "google")

    # API key hint
    if embedding == "google":
        print("\n  Set GOOGLE_API_KEY in your environment.")
        print("  export GOOGLE_API_KEY=your-key-here")
    elif embedding == "openai":
        print("\n  Set OPENAI_API_KEY in your environment.")
        print("  export OPENAI_API_KEY=your-key-here")

    # Database path
    print()
    db_default = "~/.mamba-memory/default.db"
    db_path = input(f"Database path [{db_default}]: ").strip() or db_default

    # Slot count
    print()
    print("Memory capacity (slot count):")
    print("  32  — lightweight")
    print("  64  — standard (recommended)")
    print("  128 — heavy usage")
    slots = input("Choose [64]: ").strip() or "64"

    # Compression
    print()
    print("LLM compression:")
    print("  1. none      (rule-based, no API calls)")
    print("  2. auto      (use available API key)")
    print("  3. openai")
    print("  4. anthropic")
    comp_choice = input("Choose [1]: ").strip() or "1"
    comp_map = {"1": "none", "2": "auto", "3": "openai", "4": "anthropic"}
    compression = comp_map.get(comp_choice, "none")

    # Build config
    from mamba_memory.core.types import L2Config, L3Config

    config = EngineConfig(
        embedding_provider=embedding,
        compression_model=compression,
        l2=L2Config(slot_count=int(slots)),
        l3=L3Config(db_path=db_path),
    )

    # Save
    path = save_config(config)
    print(f"\nConfig saved to: {path}")
    print()
    print("Quick start:")
    print("  mamba-memory serve --mcp       # MCP server")
    print("  mamba-memory serve --http      # HTTP API")
    print("  mamba-memory status            # Check status")
    print()


# -- serve ------------------------------------------------------------------

def _cmd_serve(args: argparse.Namespace) -> None:
    config = _load(args)

    if args.mcp:
        from mamba_memory.server.mcp.server import serve

        asyncio.run(serve(db_path=config.l3.db_path))
    elif args.http:
        try:
            import uvicorn
        except ImportError:
            print("uvicorn required: pip install mamba-memory[server]", file=sys.stderr)
            sys.exit(1)

        from mamba_memory.server.http.app import create_app

        app = create_app(config)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        print("Specify --mcp or --http", file=sys.stderr)
        sys.exit(1)


# -- status -----------------------------------------------------------------

async def _cmd_status(args: argparse.Namespace) -> None:
    config = _load(args)
    engine = MambaMemoryEngine(config)
    await engine.start()

    status = engine.status()
    data = status.model_dump()

    if args.detail in ("slots", "full"):
        active_slots = engine.l2.get_active_slots()
        data["active_slots_detail"] = [
            {
                "id": s.id,
                "topic": s.topic,
                "activation": round(s.activation, 3),
                "entities": s.entities,
                "state_preview": s.state[:80],
            }
            for s in active_slots[:20]
        ]

    print(json.dumps(data, indent=2, ensure_ascii=False))
    await engine.shutdown()


# -- compact ----------------------------------------------------------------

async def _cmd_compact(args: argparse.Namespace) -> None:
    config = _load(args)
    engine = MambaMemoryEngine(config)
    await engine.start()

    result = await engine.compact(args.layer)
    print(json.dumps(result, indent=2))

    await engine.shutdown()


# -- export -----------------------------------------------------------------

async def _cmd_export(args: argparse.Namespace) -> None:
    config = _load(args)
    engine = MambaMemoryEngine(config)
    await engine.start()

    data = {
        "status": engine.status().model_dump(),
        "l2_slots": [s.model_dump() for s in engine.l2.get_active_slots()],
        "l3_records": [
            r.model_dump()
            for r in engine.l3.search_by_time(0, float("inf"), limit=10000)
        ],
    }

    output = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w") as f:
            f.write(output)

    await engine.shutdown()


if __name__ == "__main__":
    main()
