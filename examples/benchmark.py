"""MambaMemory Benchmark Suite

Two benchmarks:
  A. Decay Parameter Tuning    — sweep decay_rate × slot_count × eviction_threshold
                                  measure memory retention, precision, slot utilization
  B. Gate Accuracy Evaluation  — labeled dataset of should-store / should-discard
                                  measure precision, recall, F1

Run: GOOGLE_API_KEY=... python examples/benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field

from mamba_memory.core.engine import MambaMemoryEngine
from mamba_memory.core.l2.gate import Gate, _importance_score
from mamba_memory.core.types import (
    EngineConfig,
    GateInput,
    L1Config,
    L2Config,
    L3Config,
    MemorySlot,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


# ╔══════════════════════════════════════════════════════════════╗
# ║  Benchmark A: Decay Parameter Tuning                        ║
# ╚══════════════════════════════════════════════════════════════╝


@dataclass
class DecayTrialResult:
    decay_rate: float
    slot_count: int
    eviction_threshold: float
    # Metrics
    anchor_retained: bool = False        # 标志记忆是否留在 L2
    anchor_recall_score: float = 0.0     # 标志记忆的召回分数
    anchor_in_any_layer: bool = False    # 标志记忆是否在任意层可找到
    l2_active_slots: int = 0
    l2_utilization: float = 0.0          # active / total
    l3_evicted: int = 0                  # 淘汰到 L3 的记录数
    noise_in_l2: int = 0                 # L2 中的低价值内容数
    total_steps: int = 0
    elapsed_ms: float = 0.0


async def run_decay_trial(
    decay_rate: float,
    slot_count: int,
    eviction_threshold: float,
) -> DecayTrialResult:
    """Run a single trial with given parameters."""
    tmp = tempfile.mkdtemp(prefix="bench-decay-")
    config = EngineConfig(
        l1=L1Config(window_size=4, max_compressed_segments=5),
        l2=L2Config(
            slot_count=slot_count,
            base_decay_rate=decay_rate,
            eviction_threshold=eviction_threshold,
            snapshot_interval=999,  # no auto-snapshot
        ),
        l3=L3Config(db_path=f"{tmp}/bench.db"),
        embedding_provider="google",
        compression_model="none",
    )

    engine = MambaMemoryEngine(config)
    await engine.start()
    t0 = time.time()

    result = DecayTrialResult(
        decay_rate=decay_rate,
        slot_count=slot_count,
        eviction_threshold=eviction_threshold,
    )

    # Phase 1: 写入标志记忆 (重要)
    await engine.ingest_explicit(
        "ANCHOR: 生产数据库 PostgreSQL 16, 端口5432, 连接池max=50",
        tags=["PostgreSQL", "database"],
    )

    # Phase 2: 写入其他重要记忆
    important = [
        ("Redis缓存256MB，LRU淘汰策略", ["Redis"]),
        ("Docker Compose 部署，Nginx反向代理端口443", ["Docker", "Nginx"]),
        ("CI/CD用GitHub Actions，自动部署到AWS ECS", ["GitHub Actions", "AWS"]),
        ("监控Prometheus+Grafana，CPU>80%告警", ["Prometheus"]),
    ]
    for content, tags in important:
        await engine.ingest_explicit(content, tags=tags)

    # Phase 3: 大量普通对话（制造衰减压力）
    for i in range(30):
        await engine.ingest(f"第{i}轮日常对话：讨论了模块{i%5}的设计细节，没有关键决定")

    # Phase 4: 再写一些噪声
    noise = ["你好", "好的收到", "嗯嗯", "ok thanks", "明天见"]
    for n in noise:
        await engine.ingest(n)

    elapsed = (time.time() - t0) * 1000

    # Evaluate
    s = engine.status()
    result.l2_active_slots = s.l2_active_slots
    result.l2_utilization = s.l2_active_slots / slot_count
    result.l3_evicted = s.l3_total_records
    result.total_steps = s.l2_step_count
    result.elapsed_ms = elapsed

    # Check anchor retention in L2
    for slot in engine.l2.slots:
        if "ANCHOR" in slot.state or "PostgreSQL" in slot.state:
            result.anchor_retained = True
            break

    # Check anchor recall score
    r = await engine.recall("PostgreSQL数据库连接", limit=3)
    result.anchor_in_any_layer = any(
        "PostgreSQL" in m.content or "ANCHOR" in m.content for m in r.memories
    )
    if r.memories:
        for m in r.memories:
            if "PostgreSQL" in m.content or "ANCHOR" in m.content:
                result.anchor_recall_score = m.score
                break

    # Count noise in L2
    noise_keywords = {"你好", "好的收到", "嗯嗯", "ok thanks", "明天见"}
    for slot in engine.l2.slots:
        if not slot.is_empty:
            if any(nk in slot.state for nk in noise_keywords):
                result.noise_in_l2 += 1

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
    return result


async def benchmark_decay():
    """Sweep decay parameters and find optimal settings."""
    print(f"\n{'█' * 60}")
    print(f"  Benchmark A: Decay Parameter Tuning")
    print(f"{'█' * 60}\n")

    # Parameter grid
    decay_rates = [0.80, 0.85, 0.90, 0.95, 0.98]
    slot_counts = [4, 8, 16]
    eviction_thresholds = [0.03, 0.05, 0.10]

    # Only sweep decay_rate with fixed slot/eviction for first pass
    print(f"  Phase 1: Sweep decay_rate (slots=8, evict=0.05)")
    print(f"  {'rate':>6} │ {'L2 act':>6} │ {'util':>5} │ {'L3':>4} │ {'anchor':>7} │ {'recall':>6} │ {'noise':>5} │ {'ms':>6}")
    print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*6}")

    best_score = -1
    best_params = {}
    all_results: list[DecayTrialResult] = []

    for dr in decay_rates:
        r = await run_decay_trial(dr, 8, 0.05)
        all_results.append(r)

        anchor_mark = PASS if r.anchor_in_any_layer else FAIL
        print(
            f"  {r.decay_rate:>6.2f} │ {r.l2_active_slots:>6} │ {r.l2_utilization:>5.0%} │ "
            f"{r.l3_evicted:>4} │ {anchor_mark:>7} │ {r.anchor_recall_score:>6.3f} │ "
            f"{r.noise_in_l2:>5} │ {r.elapsed_ms:>6.0f}"
        )

        # Composite score: recall quality + low noise + good utilization
        score = (
            r.anchor_recall_score * 3.0           # recall quality (most important)
            + (1.0 if r.anchor_in_any_layer else 0)  # anchor found
            - r.noise_in_l2 * 0.5                 # noise penalty
            + r.l2_utilization * 0.5              # utilization bonus
        )
        if score > best_score:
            best_score = score
            best_params = {"decay_rate": dr, "slot_count": 8, "eviction_threshold": 0.05}

    # Phase 2: Sweep slot_count with best decay_rate
    best_dr = best_params["decay_rate"]
    print(f"\n  Phase 2: Sweep slot_count (decay={best_dr}, evict=0.05)")
    print(f"  {'slots':>6} │ {'L2 act':>6} │ {'util':>5} │ {'L3':>4} │ {'anchor':>7} │ {'recall':>6} │ {'noise':>5}")
    print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*5}")

    for sc in slot_counts:
        r = await run_decay_trial(best_dr, sc, 0.05)
        all_results.append(r)
        anchor_mark = PASS if r.anchor_in_any_layer else FAIL
        print(
            f"  {r.slot_count:>6} │ {r.l2_active_slots:>6} │ {r.l2_utilization:>5.0%} │ "
            f"{r.l3_evicted:>4} │ {anchor_mark:>7} │ {r.anchor_recall_score:>6.3f} │ "
            f"{r.noise_in_l2:>5}"
        )

        score = (
            r.anchor_recall_score * 3.0
            + (1.0 if r.anchor_in_any_layer else 0)
            - r.noise_in_l2 * 0.5
            + r.l2_utilization * 0.5
        )
        if score > best_score:
            best_score = score
            best_params = {"decay_rate": best_dr, "slot_count": sc, "eviction_threshold": 0.05}

    # Phase 3: Sweep eviction threshold
    best_sc = best_params["slot_count"]
    print(f"\n  Phase 3: Sweep eviction_threshold (decay={best_dr}, slots={best_sc})")
    print(f"  {'evict':>6} │ {'L2 act':>6} │ {'util':>5} │ {'L3':>4} │ {'anchor':>7} │ {'recall':>6} │ {'noise':>5}")
    print(f"  {'─'*6}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*4}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*5}")

    for et in eviction_thresholds:
        r = await run_decay_trial(best_dr, best_sc, et)
        all_results.append(r)
        anchor_mark = PASS if r.anchor_in_any_layer else FAIL
        print(
            f"  {r.eviction_threshold:>6.2f} │ {r.l2_active_slots:>6} │ {r.l2_utilization:>5.0%} │ "
            f"{r.l3_evicted:>4} │ {anchor_mark:>7} │ {r.anchor_recall_score:>6.3f} │ "
            f"{r.noise_in_l2:>5}"
        )

        score = (
            r.anchor_recall_score * 3.0
            + (1.0 if r.anchor_in_any_layer else 0)
            - r.noise_in_l2 * 0.5
            + r.l2_utilization * 0.5
        )
        if score > best_score:
            best_score = score
            best_params = {"decay_rate": best_dr, "slot_count": best_sc, "eviction_threshold": et}

    print(f"\n  {BOLD}Optimal parameters:{RESET}")
    print(f"    decay_rate:          {best_params['decay_rate']}")
    print(f"    slot_count:          {best_params['slot_count']}")
    print(f"    eviction_threshold:  {best_params['eviction_threshold']}")
    print(f"    composite_score:     {best_score:.3f}")

    return best_params


# ╔══════════════════════════════════════════════════════════════╗
# ║  Benchmark B: Gate Accuracy Evaluation                      ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class GateTestCase:
    content: str
    should_store: bool
    category: str
    lang: str = "zh"


# Labeled dataset: (content, should_store, category, language)
GATE_DATASET: list[GateTestCase] = [
    # ─── SHOULD STORE (True positives) ───────────────────
    # Decisions (zh)
    GateTestCase("我决定把数据库从MySQL迁移到PostgreSQL 16", True, "decision", "zh"),
    GateTestCase("选择用Redis做缓存，Memcached不再维护了", True, "decision", "zh"),
    GateTestCase("前端框架确定采用React 18", True, "decision", "zh"),
    GateTestCase("部署方案改用Kubernetes替代Docker Swarm", True, "decision", "zh"),
    # Decisions (en)
    GateTestCase("I decided to switch from Flask to FastAPI for the API", True, "decision", "en"),
    GateTestCase("We're going with PostgreSQL instead of MongoDB", True, "decision", "en"),
    GateTestCase("Let's adopt TypeScript for all new frontend code", True, "decision", "en"),
    # Decisions (ja)
    GateTestCase("PostgreSQLを採用することに決定しました", True, "decision", "ja"),
    # Facts / configs
    GateTestCase("服务器IP是192.168.1.100，端口5432", True, "fact", "zh"),
    GateTestCase("Redis配置：maxmemory 256MB, policy allkeys-lru", True, "fact", "zh"),
    GateTestCase("The API gateway runs on port 8080 with SSL", True, "fact", "en"),
    GateTestCase("Database backup runs daily at 3am UTC", True, "fact", "en"),
    GateTestCase("API rate limit: 100 req/min for free tier", True, "fact", "en"),
    GateTestCase("生产环境配置文件在 /etc/app/config.yaml", True, "fact", "zh"),
    GateTestCase("docker compose -f docker-compose.prod.yml up -d", True, "fact", "mixed"),
    GateTestCase("连接串：postgres://user:pass@prod-db:5432/mydb", True, "fact", "mixed"),
    # Preferences
    GateTestCase("我喜欢用VS Code，不喜欢JetBrains的IDE", True, "preference", "zh"),
    GateTestCase("I prefer tabs over spaces, always 4-width", True, "preference", "en"),
    GateTestCase("测试偏好用pytest，不用unittest", True, "preference", "zh"),
    # Corrections
    GateTestCase("不对，端口应该是8443不是8080", True, "correction", "zh"),
    GateTestCase("Actually, the database is on port 5433, not 5432", True, "correction", "en"),
    GateTestCase("错了，密钥轮换周期是60天不是90天", True, "correction", "zh"),
    # Explicit memory
    GateTestCase("记住：每周三凌晨2点执行数据库备份", True, "explicit", "zh"),
    GateTestCase("Remember: API keys must be rotated every 90 days", True, "explicit", "en"),
    GateTestCase("别忘了：生产环境禁止使用root账户", True, "explicit", "zh"),
    GateTestCase("Important: never expose internal IPs in API responses", True, "explicit", "en"),
    # Action items
    GateTestCase("下一步需要配置Prometheus监控告警", True, "action", "zh"),
    GateTestCase("TODO: add rate limiting to the /api/upload endpoint", True, "action", "en"),
    GateTestCase("接下来要把日志从文件切换到ELK Stack", True, "action", "zh"),

    # ─── SHOULD DISCARD (True negatives) ─────────────────
    # Pure greetings
    GateTestCase("你好", False, "greeting", "zh"),
    GateTestCase("Hello!", False, "greeting", "en"),
    GateTestCase("Hi there", False, "greeting", "en"),
    GateTestCase("こんにちは", False, "greeting", "ja"),
    GateTestCase("早上好", False, "greeting", "zh"),
    GateTestCase("Good morning", False, "greeting", "en"),
    # Acknowledgments
    GateTestCase("好的", False, "ack", "zh"),
    GateTestCase("ok", False, "ack", "en"),
    GateTestCase("嗯嗯", False, "ack", "zh"),
    GateTestCase("sure", False, "ack", "en"),
    GateTestCase("收到", False, "ack", "zh"),
    GateTestCase("got it", False, "ack", "en"),
    GateTestCase("thanks", False, "ack", "en"),
    GateTestCase("谢谢", False, "ack", "zh"),
    # Small talk
    GateTestCase("今天天气真好", False, "smalltalk", "zh"),
    GateTestCase("The weather is nice today", False, "smalltalk", "en"),
    GateTestCase("周末有什么计划？", False, "smalltalk", "zh"),
    GateTestCase("哈哈哈", False, "smalltalk", "zh"),
    # Farewells
    GateTestCase("再见", False, "farewell", "zh"),
    GateTestCase("bye", False, "farewell", "en"),
    GateTestCase("晚安", False, "farewell", "zh"),
    GateTestCase("see you", False, "farewell", "en"),
    # Vague / empty
    GateTestCase("嗯", False, "vague", "zh"),
    GateTestCase("hmm", False, "vague", "en"),
    GateTestCase("...", False, "vague", "any"),
    GateTestCase("呵呵", False, "vague", "zh"),
]


def benchmark_gate():
    """Evaluate gate accuracy on labeled dataset."""
    print(f"\n{'█' * 60}")
    print(f"  Benchmark B: Gate Accuracy Evaluation")
    print(f"{'█' * 60}\n")

    gate = Gate()
    empty_slots = [MemorySlot(id=i) for i in range(8)]

    tp = fp = tn = fn = 0
    errors: list[tuple[GateTestCase, float, bool]] = []

    # Per-category stats
    cat_stats: dict[str, dict[str, int]] = {}

    for case in GATE_DATASET:
        score = _importance_score(case.content)
        inp = GateInput(source="turn", content=case.content, entities=[])
        decision = gate.evaluate(inp, empty_slots)

        predicted_store = decision.should_write
        correct = predicted_store == case.should_store

        # Update category stats
        cat = case.category
        if cat not in cat_stats:
            cat_stats[cat] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        if case.should_store and predicted_store:
            tp += 1
            cat_stats[cat]["tp"] += 1
        elif case.should_store and not predicted_store:
            fn += 1
            cat_stats[cat]["fn"] += 1
            errors.append((case, score, predicted_store))
        elif not case.should_store and predicted_store:
            fp += 1
            cat_stats[cat]["fp"] += 1
            errors.append((case, score, predicted_store))
        else:
            tn += 1
            cat_stats[cat]["tn"] += 1

    total = len(GATE_DATASET)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Overall metrics
    print(f"  {BOLD}Overall Metrics{RESET}")
    print(f"  ┌────────────────────┬───────────┐")
    print(f"  │ Total cases        │ {total:>9} │")
    print(f"  │ Accuracy           │ {accuracy:>8.1%} │")
    print(f"  │ Precision          │ {precision:>8.1%} │")
    print(f"  │ Recall             │ {recall:>8.1%} │")
    print(f"  │ F1 Score           │ {f1:>8.3f} │")
    print(f"  ├────────────────────┼───────────┤")
    print(f"  │ True Positives     │ {tp:>9} │")
    print(f"  │ True Negatives     │ {tn:>9} │")
    print(f"  │ False Positives    │ {fp:>9} │")
    print(f"  │ False Negatives    │ {fn:>9} │")
    print(f"  └────────────────────┴───────────┘")

    # Confusion matrix
    print(f"\n  {BOLD}Confusion Matrix{RESET}")
    print(f"                      Predicted")
    print(f"                   STORE    DISCARD")
    print(f"    Actual STORE  │ {tp:>4}  │  {fn:>4}  │")
    print(f"    Actual DISC   │ {fp:>4}  │  {tn:>4}  │")

    # Per-category breakdown
    print(f"\n  {BOLD}Per-Category Accuracy{RESET}")
    print(f"  {'category':<12} │ {'total':>5} │ {'correct':>7} │ {'acc':>6} │ {'errors':>6}")
    print(f"  {'─'*12}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*6}")

    for cat in sorted(cat_stats.keys()):
        st = cat_stats[cat]
        cat_total = sum(st.values())
        cat_correct = st["tp"] + st["tn"]
        cat_acc = cat_correct / cat_total if cat_total > 0 else 0
        cat_errors = st["fp"] + st["fn"]
        mark = PASS if cat_acc >= 0.8 else (FAIL if cat_acc < 0.5 else "⚠")
        print(f"  {mark} {cat:<10} │ {cat_total:>5} │ {cat_correct:>7} │ {cat_acc:>5.0%} │ {cat_errors:>6}")

    # Per-language breakdown
    print(f"\n  {BOLD}Per-Language Accuracy{RESET}")
    lang_stats: dict[str, dict[str, int]] = {}
    for case in GATE_DATASET:
        lang = case.lang
        if lang not in lang_stats:
            lang_stats[lang] = {"correct": 0, "total": 0}
        lang_stats[lang]["total"] += 1
        score = _importance_score(case.content)
        inp = GateInput(source="turn", content=case.content, entities=[])
        decision = gate.evaluate(inp, empty_slots)
        if decision.should_write == case.should_store:
            lang_stats[lang]["correct"] += 1

    for lang in sorted(lang_stats.keys()):
        ls = lang_stats[lang]
        acc = ls["correct"] / ls["total"]
        mark = PASS if acc >= 0.8 else FAIL
        print(f"  {mark} {lang:<6} {ls['correct']}/{ls['total']} = {acc:.0%}")

    # Error analysis
    if errors:
        print(f"\n  {BOLD}Error Analysis (misclassified){RESET}")
        for case, score, predicted in errors:
            label = "STORE" if case.should_store else "DISCARD"
            pred = "STORE" if predicted else "DISCARD"
            print(
                f"  {FAIL} [{case.category:<10}] [{case.lang}] "
                f"label={label} pred={pred} score={score:.3f} "
                f"│ {case.content[:50]}"
            )

    # Importance score distribution
    print(f"\n  {BOLD}Importance Score Distribution{RESET}")
    store_scores = [_importance_score(c.content) for c in GATE_DATASET if c.should_store]
    discard_scores = [_importance_score(c.content) for c in GATE_DATASET if not c.should_store]

    if store_scores:
        print(f"  Should-STORE:   min={min(store_scores):.3f}  avg={sum(store_scores)/len(store_scores):.3f}  max={max(store_scores):.3f}")
    if discard_scores:
        print(f"  Should-DISCARD: min={min(discard_scores):.3f}  avg={sum(discard_scores)/len(discard_scores):.3f}  max={max(discard_scores):.3f}")

    # Separation gap
    if store_scores and discard_scores:
        gap = min(store_scores) - max(discard_scores)
        print(f"  Separation gap:  {gap:+.3f} {'(good: classes separated)' if gap > 0 else '(overlap: classes not cleanly separable)'}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ╔══════════════════════════════════════════════════════════════╗
# ║  Benchmark C: Recall Weight Sensitivity                     ║
# ╚══════════════════════════════════════════════════════════════╝


async def benchmark_recall_weights():
    """Test different recall weight configurations on precision."""
    print(f"\n{'█' * 60}")
    print(f"  Benchmark C: Recall Weight Sensitivity")
    print(f"{'█' * 60}\n")

    weight_configs = [
        {"name": "semantic-heavy", "ws": 0.7, "wa": 0.1, "wr": 0.1, "wi": 0.1},
        {"name": "balanced",       "ws": 0.4, "wa": 0.3, "wr": 0.2, "wi": 0.1},
        {"name": "current",        "ws": 0.6, "wa": 0.15, "wr": 0.15, "wi": 0.1},
        {"name": "activation-heavy","ws": 0.3, "wa": 0.4, "wr": 0.2, "wi": 0.1},
        {"name": "recency-heavy",  "ws": 0.3, "wa": 0.1, "wr": 0.5, "wi": 0.1},
    ]

    # Test data
    memories = [
        ("MySQL端口3306，只读副本3307", ["MySQL"]),
        ("Redis Sentinel: 3哨兵, quorum=2", ["Redis"]),
        ("Nginx SSL证书, 强制HTTPS", ["Nginx"]),
        ("Docker镜像推送Harbor仓库", ["Docker"]),
        ("Kubernetes: 3 master, 5 worker", ["Kubernetes"]),
        ("Prometheus告警: CPU>80%邮件, >95%短信", ["Prometheus"]),
        ("Git: main保护, feature需2个reviewer", ["Git"]),
        ("API限流: 普通100次/min, VIP 1000次/min", ["API"]),
    ]

    queries = [
        ("MySQL端口", "MySQL"),
        ("Redis哨兵配置", "Redis"),
        ("HTTPS证书", "Nginx"),
        ("Docker镜像", "Docker"),
        ("K8s集群规模", "Kubernetes"),
        ("CPU告警", "Prometheus"),
        ("代码审查", "Git"),
        ("接口限流", "API"),
    ]

    print(f"  {'config':<18} │ {'precision':>9} │ {'avg_score':>9} │ {'hits':>5}")
    print(f"  {'─'*18}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*5}")

    for wc in weight_configs:
        tmp = tempfile.mkdtemp(prefix="bench-w-")
        config = EngineConfig(
            l1=L1Config(window_size=3, max_compressed_segments=5),
            l2=L2Config(
                slot_count=16,
                base_decay_rate=0.95,
                weight_semantic=wc["ws"],
                weight_activation=wc["wa"],
                weight_recency=wc["wr"],
                weight_importance=wc["wi"],
            ),
            l3=L3Config(db_path=f"{tmp}/bench.db"),
            embedding_provider="google",
            compression_model="none",
        )
        engine = MambaMemoryEngine(config)
        await engine.start()

        for content, tags in memories:
            await engine.ingest_explicit(content, tags=tags)

        hits = 0
        total_score = 0.0
        for query, expected in queries:
            r = await engine.recall(query, limit=1)
            if r.memories and expected in r.memories[0].content:
                hits += 1
                total_score += r.memories[0].score

        precision = hits / len(queries)
        avg_score = total_score / hits if hits > 0 else 0

        mark = PASS if precision >= 0.6 else FAIL
        print(f"  {mark} {wc['name']:<16} │ {precision:>8.0%} │ {avg_score:>9.3f} │ {hits:>3}/{len(queries)}")

        await engine.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main                                                       ║
# ╚══════════════════════════════════════════════════════════════╝


async def main():
    print(f"\n{'█' * 60}")
    print(f"  MambaMemory Benchmark Suite")
    print(f"{'█' * 60}")

    t0 = time.time()

    # B runs first (no API calls, instant)
    gate_metrics = benchmark_gate()

    # A: decay tuning (needs API for embeddings)
    optimal_params = await benchmark_decay()

    # C: recall weight sensitivity
    await benchmark_recall_weights()

    elapsed = time.time() - t0

    # Final summary
    print(f"\n{'█' * 60}")
    print(f"  Final Summary")
    print(f"{'█' * 60}")
    print(f"\n  Gate accuracy:       {gate_metrics['accuracy']:.1%}")
    print(f"  Gate F1:             {gate_metrics['f1']:.3f}")
    print(f"  Optimal decay_rate:  {optimal_params.get('decay_rate', 'N/A')}")
    print(f"  Optimal slot_count:  {optimal_params.get('slot_count', 'N/A')}")
    print(f"  Optimal evict_thr:   {optimal_params.get('eviction_threshold', 'N/A')}")
    print(f"  Total time:          {elapsed:.0f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
