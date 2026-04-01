# Contributing to MambaMemory

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/mamba-memory/mamba-memory.git
cd mamba-memory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
# Unit tests (fast, no API key needed)
python -m pytest tests/ --ignore=tests/test_mcp_integration.py -v

# MCP integration tests (needs MCP SDK)
python -m pytest tests/test_mcp_integration.py -v

# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=mamba_memory --cov-report=html
```

## Code Style

- Python 3.11+, strict typing, avoid `any`
- Formatting: `ruff format mamba_memory/ tests/`
- Linting: `ruff check mamba_memory/ tests/`
- Pydantic for data models, async/await throughout

## Pull Requests

1. One PR = one feature/fix
2. Include tests for new functionality
3. Run `ruff check` and `pytest` before submitting
4. Keep PRs focused and under ~500 lines when possible

## Architecture

```
mamba_memory/
├── core/          # Engine, L1/L2/L3 layers, text analysis
├── server/        # MCP + HTTP interfaces
├── sdk/           # Python client SDK
├── config.py      # YAML config loading
└── cli.py         # CLI entry point
```

Key design principles:
- L1 (session) → L2 (SSM state) → L3 (persistent) data flow
- Selective gate filters noise before L2 write
- Decay-based activation management
- Entity graph for relationship tracking
- asyncio.Lock for concurrency safety
