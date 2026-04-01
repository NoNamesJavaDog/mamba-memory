# Changelog

## 0.1.0 (2025-04-01) — Initial Release

### Core Engine
- Three-layer memory architecture: L1 (session) → L2 (SSM state) → L3 (persistent)
- SSM-inspired selective gate with 5-dimension importance scoring (F1=0.945)
- Decay-based state evolution with configurable decay rate (optimal: 0.98)
- Saturation guard prevents evicting important memories when all slots are active
- Force-write bypasses gate, evicts old content to L3 before overwriting
- asyncio.Lock concurrency safety on all public methods
- Cold-start recovery from L3 snapshots with catch-up decay

### Text Analysis
- Language-agnostic tokenizer (zh/en/ja/ko)
- Character bigram + token overlap relevance scoring
- Structured information extraction (IPs, ports, URLs, configs, commands)
- Entity auto-extraction (tech names, CJK person names, capitalized words)
- Information density scoring

### Embedding Providers
- Google Gemini (`gemini-embedding-001`)
- OpenAI (`text-embedding-3-small`)
- Local sentence-transformers
- Dummy hash-based (testing)
- Auto-detection from environment variables

### LLM Compression
- OpenAI / Anthropic backends for L1 compression + L2 slot merging
- MCP pre-summary mode: LLM client does compression, zero internal API calls
- Structured rule-based fallback when no LLM configured

### Persistent Storage
- SQLite with WAL mode for structured records
- HNSW vector index with disk persistence (save/load on startup/shutdown)
- Entity graph with co-occurrence relations
- State snapshot save/restore

### Interfaces
- MCP Server: 5 tools + 2 prompts + error handling
- HTTP REST API: 6 endpoints via FastAPI
- Python SDK: direct (in-process) + HTTP client
- CLI: init / serve / status / compact / export
- YAML config file support + environment variable overrides

### Testing
- 81 tests (71 unit + 10 MCP integration)
- 10-scenario E2E test suite
- 3-part benchmark (gate accuracy, decay tuning, recall weight sensitivity)

### Deployment
- Docker + docker-compose
- PyPI packaging
- GitHub Actions CI (Python 3.11/3.12/3.13)
