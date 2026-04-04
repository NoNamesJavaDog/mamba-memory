# MambaMemory

**AI Agent Cognitive Memory Engine** — Not another vector database. A memory system that thinks.

[中文文档 (Chinese README)](README.zh-CN.md)

Inspired by [Mamba3](https://arxiv.org/abs/2312.00752) (Selective State Space Model), MambaMemory brings SSM principles to AI agent memory management: selective gating, state decay, fixed-capacity evolution, and input-driven recall.

```
Input → [L1 Session] → overflow → [L2 State / SSM] → evict → [L3 Persistent]
         recent turns     gate      cognitive slots    archive   SQLite + HNSW
         full fidelity    filter    decay & evolve     forever   vector search
```

## Why Not Just Use a Vector Database?

| | Vector DB (Chroma, Qdrant) | MambaMemory |
|---|---|---|
| **Model** | Passive warehouse: store, then search | Active brain: gate, evolve, decay, recall |
| **Capacity** | Grows forever, you clean up | Fixed slots, auto-compresses |
| **Write** | Everything goes in | Selective gate filters noise (94.5% accuracy) |
| **Search** | Static similarity | Semantic-dominant + time-aware decay + rehearsal |
| **Old memories** | Stay forever until deleted | Ebbinghaus forgetting curve + spaced repetition |
| **Languages** | Depends on embedding | Built-in zh/en/ja/ko text analysis + any embedding model |
| **Scaling** | Single backend | Pluggable: SQLite / PostgreSQL / Redis |

MambaMemory is not for million-document RAG. It's for **single-agent personal memory** — the kind that remembers what matters, forgets what doesn't, and evolves with every conversation.

## Install

```bash
pip install mamba-memory

# With Google Gemini embeddings (recommended)
pip install mamba-memory[google]

# With OpenAI embeddings
pip install mamba-memory[openai]

# With MCP + HTTP server
pip install mamba-memory[server]

# Everything
pip install mamba-memory[all]
```

Requires Python 3.11+.

## Quick Start

```python
import asyncio
from mamba_memory.sdk import MambaMemory

async def main():
    async with MambaMemory() as mem:
        # Ingest — gate decides what's worth remembering
        await mem.ingest("I'm switching our database to PostgreSQL on port 5432")
        await mem.ingest("Hello!")  # gate discards: low importance
        await mem.ingest("Remember: rotate API keys every 90 days", force=True)

        # Recall — semantic-weighted, context-aware
        result = await mem.recall("database setup")
        for m in result.memories:
            print(f"[{m.layer}] (score={m.score:.2f}) {m.content}")

        # Status
        s = mem.status()
        print(f"L2: {s.l2_active_slots}/{s.l2_total_slots} slots active")

asyncio.run(main())
```

### MCP Mode (Claude Code / Codex / OpenClaw)

No API keys needed — the LLM client **is** the compressor:

```python
# Claude calls this MCP tool:
memory_ingest({
    "content": "long conversation about database migration...",
    "summary": "User decided PostgreSQL 16, reason: JSONB + performance",
    "entities": ["PostgreSQL", "MySQL"],
})
# summary goes directly to L2 — zero internal LLM calls
```

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────────┐
│  L1  Session Layer                                              │
│  Last K turns + recursive summarization chain                   │
│  Latency: <1ms  |  Capacity: fixed window  |  Storage: memory  │
└──────────────────────┬──────────────────────────────────────────┘
                  Gate ↕ rule engine + learned classifier (hybrid)
┌──────────────────────┴──────────────────────────────────────────┐
│  L2  State Layer (SSM Core)                                     │
│  64 cognitive slots with gating + Ebbinghaus decay + evolution  │
│  Latency: <10ms  |  Capacity: fixed dimension  |  Storage: mem │
└──────────────────────┬──────────────────────────────────────────┘
             Sync/Query ↕ persist / retrieve / knowledge graph
┌──────────────────────┴──────────────────────────────────────────┐
│  L3  Persistent Layer                                           │
│  SQLite/PostgreSQL + HNSW + typed knowledge graph + snapshots   │
│  Latency: <200ms  |  Capacity: unlimited  |  Storage: disk     │
└─────────────────────────────────────────────────────────────────┘
```

### SSM Concept Mapping

| Mamba3 SSM | MambaMemory L2 |
|---|---|
| Hidden state h(t) | Fixed-dimension memory slots (default: 64) |
| State transition A | Dual decay: step-based + Ebbinghaus time-based |
| Input projection B | Gate-In: rule engine + learned classifier |
| Output projection C | Recall: semantic-dominant selective read |
| Selection mechanism Delta(x) | Hybrid gate: 5-dim rules + logistic regression |
| Discretization | Each conversation turn = one discrete time step |

### How Memory Decisions Work

Every piece of input goes through a **three-stage decision pipeline** that determines whether it becomes long-term memory, short-term context, or gets discarded entirely.

#### Stage 1: Noise Pre-Filter (instant, <0.1ms)

Pure filler is killed immediately — zero computation wasted:

```
"你好"        → score 0, DISCARD     "ok"          → score 0, DISCARD
"嗯嗯"        → score 0, DISCARD     "こんにちは"    → score 0, DISCARD
"好的收到"     → score 0, DISCARD     "see you"     → score 0, DISCARD
```

Pattern: greetings, acknowledgments, farewells, emoji reactions, single-word responses in any supported language (zh/en/ja/ko). These never enter any memory layer.

#### Stage 2: Importance Scoring (rule engine + learned classifier)

Content that passes the pre-filter gets scored across **5 dimensions**:

```
"我决定把数据库迁移到PostgreSQL 16"
  ├── Intent:     0.67  (decision signal: "决定")
  ├── Structured: 0.50  (version number: "16")
  ├── Explicit:   0.00
  ├── Action:     0.00
  ├── Density:    0.54  (high lexical diversity)
  └── Total:      0.35  → STORE ✓ (threshold: 0.25)

"今天天气真好"
  ├── Intent:     0.00
  ├── Structured: 0.00
  ├── Explicit:   0.00
  ├── Action:     0.00
  ├── Density:    0.12  (low: mostly stop words)
  └── Total:      0.02  → DISCARD ✗

"记住：API密钥每90天轮换"
  ├── Intent:     0.00
  ├── Structured: 0.33  (number: "90天")
  ├── Explicit:   1.00  (signal: "记住")
  ├── Action:     0.00
  ├── Density:    0.40
  └── Total:      0.33  → STORE ✓
```

**What gets stored** (high-value signals):

| Signal | Examples | Why it matters |
|--------|----------|----------------|
| Decisions | "我决定用Docker", "switch to PostgreSQL" | Changes system state |
| Preferences | "我喜欢用VS Code", "prefer tabs" | Personalizes behavior |
| Corrections | "不对，端口是8443", "actually it's..." | Fixes wrong knowledge |
| Facts with data | "IP: 192.168.1.1, port 5432" | Concrete retrievable info |
| Explicit requests | "记住这个", "don't forget" | User intent to persist |
| Schedules | "每天凌晨3点备份", "daily at 3am" | Recurring operations |
| Configs | "maxmemory 256MB", "rate limit 100/min" | System parameters |
| Action items | "下一步部署到生产", "TODO: add tests" | Future tasks |

**What gets discarded** (low-value signals):

| Signal | Examples | Why it's filtered |
|--------|----------|-------------------|
| Greetings | "你好", "hello", "早上好" | Zero information content |
| Acknowledgments | "好的", "ok", "got it", "收到" | No new knowledge |
| Small talk | "天气真好", "周末有计划吗" | Not actionable |
| Filler | "嗯嗯", "哈哈", "..." | Noise |
| Vague responses | "可能吧", "我想想" | No commitment |

#### Stage 3: Long-Term vs Short-Term Placement

Content that passes scoring enters the memory system, but **where** it lands depends on context:

```
Score >= 0.25 + novel topic    → L2 new slot (long-term cognitive state)
Score >= 0.25 + similar topic  → L2 update existing slot (knowledge merge)
Score >= 0.20 + novel          → L2 moderate confidence slot
Score < 0.20                   → stays in L1 only (short-term window)
force=True                     → L2 directly (bypasses all scoring)
```

Once in L2, memories are **not permanent** — they compete for survival:
- **Active memories** (frequently recalled) grow stronger via rehearsal
- **Neglected memories** decay over time (Ebbinghaus curve)
- **Weakest memories** get evicted to L3 cold storage (still searchable, just slower)
- **All slots occupied + high activation?** Low-value new content is rejected entirely (saturation guard)

This means the system naturally converges on storing what the user **actually uses**, not just what looked important at write time.

### Key Mechanisms

**Selective Gate (Hybrid)** — Two evaluation paths, combined:

1. **Rule Engine** (always active) — 5-dimension weighted scoring:

| Dimension | Weight | Signals |
|-----------|--------|---------|
| Intent | 0.25 | Decisions, preferences, corrections (zh/en/ja/ko) |
| Structured data | 0.25 | IPs, ports, URIs, configs, commands, schedules, rate limits |
| Explicit memory | 0.20 | "remember", "don't forget", "覚えて" |
| Action/task | 0.15 | TODO, next steps, deadlines, "must", "下一步" |
| Info density | 0.15 | Lexical diversity, content-to-stopword ratio, numeric density |

2. **Learned Neural Classifier** (optional, self-evolving) — 37-feature two-layer neural network (608 parameters) that learns what THIS user considers important:

```
66 input features                      16 hidden (ReLU)       1 output (sigmoid)
├── 15 rule signals (regex patterns) ──┐
├──  4 negation/deferral detection   ──┤
├──  5 command detection             ──┼── hidden layer ──── store probability
├──  4 code/path patterns            ──┤    (non-linear)
├── 24 semantic embed (compressed)   ──┤
├──  8 context window (recent conv)  ──┤
└──  6 user profile (topic affinity) ──┘
```

**What the 7 feature groups capture:**

| Features | Dim | What it understands |
|----------|-----|---------------------|
| Rule signals | 15 | Decisions, preferences, corrections, facts, explicit memory |
| Negation/deferral | 4 | "到时候再决定" ≠ "我决定" — deferral flips meaning |
| Command detection | 5 | `terraform plan`, `psql -h` — executable commands |
| Code/path patterns | 4 | File paths, URIs, config files, SQL |
| Semantic embedding | 24 | Compressed embedding — understands meaning |
| Context window | 8 | Topic continuation vs topic shift |
| User profile | 6 | What topics this specific user recalls most |

**Negation/deferral detection (key innovation):**

The hardest gate problem is content that contains action words but means the opposite:

```
"我决定用PostgreSQL"     → decision signal ON  + deferral OFF → STORE ✓
"到时候再决定用什么数据库"  → decision signal ON  + deferral ON  → DISCARD ✓
"Haven't decided yet"    → decision signal ON  + deferral ON  → DISCARD ✓
"以后再选择框架"          → decision signal ON  + deferral ON  → DISCARD ✓
```

Feature[3] (decision+deferral combo) fires ONLY when both signals are present, giving the neural network a clean "poison pill" signal.

**Ensemble gate (rule + neural voting):**

```
Both agree        → trust neural (more accurate)
Neural conf > 0.9 → trust neural (very confident)
Neural conf < 0.3 → trust neural (catches deferral that rules miss)
Disagreement      → average both scores (cautious)
```

**Three self-evolution signals:**

```
1. Batch training:    417 labeled samples → 300 epochs → cold start
2. User corrections:  "remember this" / "forget that" → 1 backprop step, <0.1ms
3. Implicit feedback: recalled = positive, evicted-without-recall = negative
                      → self-improving loop, no user action needed
```

**User profile personalization:**

```
User recalls "PostgreSQL" 3x, "Redis" 2x → profile learns DB affinity
→ "PostgreSQL vacuum tuning" gets storage boost
→ "cooking recipe" gets no boost
```

**Evolution: v1 → v2 → current**

| | v1 | v2 | current |
|---|---|---|---|
| Model | Logistic regression | 2-layer NN | 2-layer NN + ensemble |
| Features | 15 | 37 | **66** |
| Parameters | 16 | 608 | **1072** |
| Negation detection | No | No | **Yes (4-dim)** |
| Command detection | No | Yes | Yes |
| Ensemble voting | No | No | **Rule + Neural** |
| CV Accuracy (5-fold) | — | 98.7% ± 1.4% | **99.3% ± 0.6%** |
| All categories 100% | — | 13/14 | **14/14** |

**Time-Aware Decay (Ebbinghaus)** — Dual decay model:

```
Step decay:   activation *= base_decay_rate              (per interaction)
Time decay:   activation *= 2^(-Δt / effective_halflife) (wall-clock)
Rehearsal:    effective_halflife *= (1 + 0.5 * recall_count)
```

A memory recalled 5 times has **3.5x the half-life** of one never recalled. This implements spaced repetition — frequently useful memories persist, forgotten ones fade naturally.

**Recursive Summarization Chain** — When L1 compressed segments overflow, instead of pushing each individually to L2, multiple segments are merged into a single higher-level "super-summary":

```
turns → segment → [overflow] → merge(N segments) → super-segment → L2 gate
```

This produces denser, more informative summaries spanning longer time ranges.

**Saturation Guard** — When all L2 slots have activation > 0.6, low-value content is rejected instead of evicting important memories.

**Knowledge Graph** — Entities are auto-extracted with typed relations (10 types: uses, depends_on, replaces, deployed_on, connects_to, etc.) and stored in L3. Supports:
- Multi-hop neighborhood expansion
- Path-based inference (A→B→C relationship chains)
- Indirect relationship discovery

### Recall Scoring

| Signal | Weight | Source |
|--------|--------|--------|
| Semantic similarity | **0.60** | Embedding cosine distance |
| Activation level | 0.15 | Decay history (includes time + rehearsal) |
| Recency | 0.15 | Steps since last access |
| Importance | 0.10 | Initial gate score |

Recalled slots get an activation boost + rehearsal count increment, creating a spaced-repetition feedback loop.

## Embedding Providers

| Provider | Config | Notes |
|----------|--------|-------|
| Google Gemini | `embedding_provider="google"` | `gemini-embedding-001`, 768 dim |
| OpenAI | `embedding_provider="openai"` | `text-embedding-3-small`, 256 dim |
| Local | `embedding_provider="local"` | sentence-transformers, zero API calls |
| Dummy | `embedding_provider="dummy"` | Hash-based, for testing only |
| Auto | `embedding_provider="auto"` | Tries google → openai → local → dummy |

## LLM Compression

| Mode | Config | When to use |
|------|--------|-------------|
| None | `compression_model="none"` | Default. Structured rule-based compression |
| Google Gemini | `compression_model="google"` | Fast + cheap, needs `GOOGLE_API_KEY` |
| OpenAI | `compression_model="openai"` | High quality, needs `OPENAI_API_KEY` |
| Anthropic | `compression_model="anthropic"` | Claude-based, needs `ANTHROPIC_API_KEY` |
| MCP pre-summary | Pass `summary` to `memory_ingest` | **Best for Claude/Codex** — zero extra cost |

## Serve

### MCP Server

```bash
mamba-memory serve --mcp
```

Tools: `memory_ingest`, `memory_recall`, `memory_forget`, `memory_status`, `memory_compact`

Prompts: `memory-compress`, `memory-usage-guide`

Register with OpenClaw:
```bash
openclaw mcp set mamba-memory '{"command":"mamba-memory","args":["serve","--mcp"]}'
```

### HTTP API

```bash
mamba-memory serve --http --port 8420
```

Authentication: set `MAMBA_MEMORY_API_KEY` env var to enable Bearer token auth. `/health` is always open.

Dashboard: visit `http://localhost:8420/ui` for a visual status panel.

Endpoints: `POST /ingest`, `POST /recall`, `POST /forget`, `POST /compact`, `GET /status`, `GET /health`, `GET /ui`

### CLI

```bash
mamba-memory init                     # Interactive setup wizard
mamba-memory serve --mcp              # MCP server
mamba-memory serve --http             # HTTP API
mamba-memory status --detail slots    # View cognitive slots
mamba-memory compact --layer all      # Force compaction
mamba-memory export -o memories.json  # Export everything
```

### Docker

```bash
docker compose up -d
# API at http://localhost:8420, dashboard at http://localhost:8420/ui
```

## Configuration

```python
from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config

config = EngineConfig(
    l1=L1Config(
        window_size=8,
        max_compressed_segments=20,
    ),
    l2=L2Config(
        slot_count=64,
        base_decay_rate=0.98,           # Step decay (benchmark optimal)
        eviction_threshold=0.05,
        snapshot_interval=50,
        time_decay_enabled=True,        # Ebbinghaus time-based decay
        time_decay_halflife=3600.0,     # 1 hour base half-life
        weight_semantic=0.60,           # Recall weights (benchmark validated)
        weight_activation=0.15,
        weight_recency=0.15,
        weight_importance=0.10,
    ),
    l3=L3Config(
        db_path="~/.mamba-memory/default.db",
    ),
    embedding_provider="google",
    compression_model="none",
    namespace="default",                # Multi-agent isolation
)
```

YAML config file (`~/.mamba-memory/config.yaml`):
```yaml
embedding_provider: google
compression_model: none
namespace: default

l2:
  slot_count: 64
  base_decay_rate: 0.98
  time_decay_enabled: true
  time_decay_halflife: 3600

l3:
  db_path: ~/.mamba-memory/default.db
```

Environment variable overrides: `MAMBA_MEMORY_DB`, `MAMBA_MEMORY_EMBEDDING`, `MAMBA_MEMORY_SLOTS`, `MAMBA_MEMORY_DECAY_RATE`.

## Benchmark Results

### Gate Accuracy — 5-Fold Cross-Validation (417 samples, 14 categories)

| Metric | Rule Engine | Neural Gate v2 |
|--------|-------------|----------------|
| Accuracy | 87.5% ± 1.7% | **99.3% ± 0.6%** |
| Precision | 95.2% ± 3.1% | **99.6% ± 0.8%** |
| Recall | 80.9% ± 5.3% | **99.0% ± 1.2%** |
| F1 | 87.2% ± 2.1% | **99.3% ± 0.6%** |
| Confidence gap | — | **0.957** (store avg 0.986, discard avg 0.028) |

Tested on 417 augmented samples (222 store / 195 discard) across 14 categories including tricky deferral cases ("到时候再决定", "Haven't decided yet"). 5-fold cross-validation ensures real generalization. Last fold: **zero errors, all 14 categories at 100%.**

### Decay Parameter Sweep

| decay_rate | L2 Utilization | Anchor Recall | L3 Evictions |
|------------|---------------|---------------|--------------|
| 0.80 | 12% | 0.049 | 5 |
| 0.90 | 62% | 0.615 | 1 |
| **0.98** | **75%** | **0.750** | 0 |

### Recall Weight Sensitivity

| Config | Precision |
|--------|-----------|
| **semantic=0.6 (default)** | **100%** |
| semantic=0.7 | **100%** |
| balanced (0.4/0.3/0.2) | 38% |

**Semantic weight must be >= 0.6.**

## Scene Presets

MambaMemory adapts to different usage scenarios via presets. Switch with one line — no core code changes.

### Fiction Writing Mode

For novel/story writing: xianxia, romance, sci-fi, and general fiction.

```python
from mamba_memory.presets.fiction import create_fiction_engine

engine = create_fiction_engine(db_path="~/.mamba-memory/my-novel.db")
await engine.start()

# Characters, plot, world-building — all recognized and stored
await engine.ingest("林月是月影门掌门独女，精通寒冰剑法", tags=["林月", "月影门"])
await engine.ingest("Captain Nova命令飞船进入曲率航行", tags=["Nova"])
await engine.ingest("苏晚发现陆远舟就是十年前为她撑伞的少年", tags=["苏晚", "陆远舟"])

# Greetings still filtered: "你好" → 0.0, "ok" → 0.0
```

**What changes in fiction mode:**

| | Technical (default) | Fiction |
|---|---|---|
| Slots | 64 | **128** (more characters/locations) |
| Tokens/slot | 300 | **500** (character descriptions need space) |
| Decay rate | 0.98 (100 steps → 13%) | **0.995** (100 steps → 60%) |
| Time halflife | 1 hour | **1 day** (characters don't fade between chapters) |
| Eviction threshold | 0.05 | **0.02** (harder to forget) |

**Fiction gate scoring (7 dimensions):**

| Dimension | Weight | Signals |
|-----------|--------|---------|
| Character | 0.25 | Names, traits, appearance, abilities, cultivation level |
| Plot | 0.20 | Events, twists, deaths, betrayals, chapter markers |
| Relationship | 0.20 | Love, hate, master/disciple, siblings, allies, rivals |
| World-building | 0.15 | Locations, factions, magic systems, civilizations |
| Names density | 0.10 | Proper nouns, faction/location names |
| Style | 0.05 | POV, narrative voice, pacing notes |
| Length | 0.05 | Longer content more likely substantive |

**Fiction entity types (6):** character, location, faction, artifact, event, concept

**Fiction relation types (14):** loves, hates, master_of, disciple_of, parent_of, sibling_of, ally_of, rival_of, member_of, located_in, possesses, killed, betrayed, successor_of

**Relation extraction (Chinese + English):**
```
"林月爱上了陈风"           → (林月, loves, 陈风)
"陈风是天剑宗弟子"         → (陈风, member_of, 天剑宗)
"张三杀了李四"             → (张三, killed, 李四)
"Alice betrayed the Guild" → (Alice, betrayed, Guild)
```

Also available via CLI: `mamba-memory init` → choose "fiction" mode.

### Custom Presets

Create your own preset for any domain:

```python
from mamba_memory.core.types import EngineConfig, L2Config, L3Config

# Medical notes preset
medical_config = EngineConfig(
    l2=L2Config(slot_count=256, base_decay_rate=0.999),  # never forget diagnoses
    namespace="medical",
)

# Business/meetings preset
business_config = EngineConfig(
    l2=L2Config(slot_count=64, base_decay_rate=0.98),
    namespace="business",
)
```

## Project Structure

```
mamba_memory/
├── core/
│   ├── types.py              # Pydantic data models
│   ├── embedding.py          # Embedding providers (Google/OpenAI/local/dummy)
│   ├── engine.py             # Main engine — L1 ↔ L2 ↔ L3 orchestration
│   ├── text.py               # Tokenizer, ngrams, entity extraction, info density
│   ├── llm.py                # LLM backends (Google/OpenAI/Anthropic)
│   ├── l1/session.py         # Session layer + recursive summarization chain
│   ├── l2/
│   │   ├── gate.py           # Hybrid gate (rules + neural classifier)
│   │   ├── learned_gate.py   # 66-feature self-evolving neural gate + negation detection
│   │   ├── evolver.py        # Dual decay (step + Ebbinghaus time) + evolution
│   │   ├── recaller.py       # Semantic-weighted recall with rehearsal tracking
│   │   └── state.py          # State manager + force_write + snapshot
│   └── l3/
│       ├── store.py          # SQLite + HNSW + entity graph + schema migration
│       ├── knowledge_graph.py # Typed relations + multi-hop inference
│       └── backend.py        # Pluggable backends (PostgreSQL, Redis stubs)
├── server/
│   ├── mcp/server.py         # MCP Server (5 tools + 2 prompts)
│   └── http/
│       ├── app.py            # FastAPI REST API + Bearer auth
│       └── dashboard.py      # Embedded Web UI
├── presets/
│   ├── __init__.py           # Preset registry
│   └── fiction.py            # Fiction writing (gate + entities + relations + config)
├── sdk/client.py             # Python SDK (direct + HTTP)
├── config.py                 # YAML config loading + env overrides
└── cli.py                    # CLI: init / serve / status / compact / export
```

**5,600+ lines of source, 29 modules, 71 unit tests + 10 MCP integration tests.**

## Integrations

### OpenClaw Plugin

A bundled OpenClaw plugin (`extensions/memory-mamba/`) provides:
- MCP bridge to MambaMemory subprocess
- `memory_search` + `memory_ingest` tools for the agent
- Memory flush plan integration
- System prompt guidance

Activate: `openclaw config set plugins.slots.memory memory-mamba`

### Storage Backends

| Backend | Status | Use case |
|---------|--------|----------|
| **SQLite** | Production | Default, zero config, single instance |
| **PostgreSQL** | Stub | Multi-instance, pgvector, concurrent access |
| **Redis** | Stub (L2 state only) | Real-time L2 state sync across instances |

## Competitors & Positioning

| | Mem0 | Zep | LangMem | MambaMemory |
|---|---|---|---|---|
| Core model | Graph + vector | Temporal KG | LangChain | SSM state machine |
| Capacity mgmt | Manual | Time-based | Manual | Auto (Ebbinghaus + eviction) |
| Write filtering | No | No | No | Neural gate (99.3% CV, self-evolving) |
| State evolution | No | No | No | Yes (SSM-inspired) |
| Forgetting model | No | Decay | No | Ebbinghaus + spaced repetition |
| Learned gate | No | No | No | Yes (neural net + user profiling) |
| Knowledge graph | Yes | Yes | No | Yes (typed + inference) |
| Recursive compression | No | No | No | Yes (summary chain) |
| Multi-language | Depends | Depends | Depends | Built-in zh/en/ja/ko |
| MCP native | No | No | No | Yes (5 tools + 2 prompts) |
| Self-hosted | Partial | Partial | Yes | Yes (SQLite, zero deps) |

## License

MIT
