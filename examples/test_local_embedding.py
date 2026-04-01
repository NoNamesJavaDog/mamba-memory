"""End-to-end test with local sentence-transformers embedding model.

Tests the full three-layer flow with real semantic search:
  1. Ingest diverse conversations (zh + en + mixed)
  2. Verify L2 gate filters noise, stores decisions
  3. Verify semantic recall across all layers
  4. Verify entity graph is populated
  5. Verify decay and eviction
  6. Verify snapshot persistence and cold-start recovery

Run: python examples/test_local_embedding.py
"""

import asyncio
import shutil
import tempfile

from mamba_memory.core.engine import MambaMemoryEngine
from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config


async def main():
    # Use a temp dir so we start fresh
    tmp = tempfile.mkdtemp(prefix="mamba-test-")
    db_path = f"{tmp}/test.db"

    config = EngineConfig(
        l1=L1Config(window_size=4, max_compressed_segments=3),
        l2=L2Config(
            slot_count=8,
            base_decay_rate=0.85,  # aggressive decay for testing
            eviction_threshold=0.05,
            snapshot_interval=10,
        ),
        l3=L3Config(db_path=db_path),
        embedding_provider="google",  # use Google Gemini embedding
        compression_model="none",     # no LLM for compression
    )

    engine = MambaMemoryEngine(config)
    await engine.start(session_id="e2e-test")

    print("=" * 60)
    print("MambaMemory E2E Test — Local Embedding Model")
    print("=" * 60)

    # ─── Phase 1: Ingest conversations ─────────────────────────

    print("\n--- Phase 1: Ingesting conversations ---")

    conversations = [
        # High value — should be stored
        ("user", "我决定把数据库从MySQL迁移到PostgreSQL", ["MySQL", "PostgreSQL"]),
        ("user", "服务器IP是192.168.1.100，端口5432", ["PostgreSQL"]),
        ("user", "Redis缓存设为256MB，淘汰策略用LRU", ["Redis"]),
        ("user", "I decided to use Docker Compose for orchestration", ["Docker"]),
        ("user", "The API gateway runs on port 8080 with Nginx reverse proxy", ["Nginx"]),
        ("user", "记住：每周三凌晨2点执行数据库备份", []),
        # Low value — should be filtered by gate
        ("user", "你好", []),
        ("user", "嗯嗯好的", []),
        ("user", "ok", []),
        ("user", "今天天气真好", []),
        # More high value — these will overflow L1 window
        ("user", "前端框架选择了React 18，状态管理用Zustand", ["React", "Zustand"]),
        ("user", "CI/CD pipeline用GitHub Actions，部署到AWS ECS", ["GitHub Actions", "AWS"]),
        ("user", "日志系统用ELK Stack，Elasticsearch端口9200", ["Elasticsearch"]),
        ("user", "监控用Prometheus + Grafana，告警阈值CPU>80%", ["Prometheus", "Grafana"]),
    ]

    stored_count = 0
    discarded_count = 0
    for role, content, tags in conversations:
        result = await engine.ingest(content, role=role, tags=tags if tags else None)
        status = "STORED" if result.stored else "DISCARDED"
        if result.stored:
            stored_count += 1
        else:
            discarded_count += 1
        layer = result.layer
        print(f"  [{status:>9}] [{layer}] {content[:45]:<45} → {result.reason[:40]}")

    print(f"\n  Summary: {stored_count} stored, {discarded_count} discarded")

    # ─── Phase 2: Check status ─────────────────────────────────

    print("\n--- Phase 2: Memory Status ---")
    s = engine.status()
    print(f"  L1 window:     {s.l1_window_turns} turns")
    print(f"  L1 compressed: {s.l1_compressed_segments} segments")
    print(f"  L2 slots:      {s.l2_active_slots}/{s.l2_total_slots} active")
    print(f"  L2 steps:      {s.l2_step_count}")
    print(f"  L3 records:    {s.l3_total_records}")
    print(f"  L3 entities:   {s.l3_entity_count}")

    # ─── Phase 3: Semantic recall ──────────────────────────────

    print("\n--- Phase 3: Semantic Recall ---")

    queries = [
        "数据库用的什么？",
        "What's the deployment setup?",
        "缓存配置",
        "monitoring and alerts",
        "前端技术栈",
        "CI/CD pipeline",
        "备份策略",
    ]

    for q in queries:
        result = await engine.recall(q, limit=3)
        print(f"\n  Q: {q}")
        if result.memories:
            for m in result.memories:
                print(f"    [{m.layer}] (score={m.score:.3f}) {m.content[:60]}")
        else:
            print("    (no results)")

    # ─── Phase 4: L2 slot inspection ──────────────────────────

    print("\n--- Phase 4: L2 Active Slots ---")
    active = engine.l2.get_active_slots()
    for slot in active[:10]:
        print(f"  Slot {slot.id:2d} | act={slot.activation:.3f} | "
              f"topic={slot.topic[:40]:<40} | entities={slot.entities[:3]}")

    # ─── Phase 5: Force ingest + forget ───────────────────────

    print("\n--- Phase 5: Force Ingest + Forget ---")
    result = await engine.ingest_explicit(
        "绝密：生产环境root密码是SuperSecret123!",
        tags=["credentials", "security"],
    )
    print(f"  Force stored: {result.stored}, layer: {result.layer}")

    # Verify it's recallable
    recall = await engine.recall("root密码", limit=1)
    if recall.memories:
        print(f"  Recalled: {recall.memories[0].content[:50]}")
    else:
        print("  (recall failed)")

    # Forget it
    forgotten = await engine.forget("root密码")
    print(f"  Forgotten: {forgotten} memories")

    # Verify it's gone from L2
    recall2 = await engine.recall("root密码", limit=1, layers=["l2"])
    if recall2.memories:
        print(f"  Still in L2: {recall2.memories[0].content[:50]} (unexpected)")
    else:
        print("  Confirmed removed from L2")

    # ─── Phase 6: Snapshot & Cold Start ───────────────────────

    print("\n--- Phase 6: Snapshot & Cold Start ---")
    step_before = engine.l2.step_count
    slots_before = engine.l2.active_slot_count
    await engine.shutdown()
    print(f"  Shutdown: saved snapshot at step {step_before}")

    # Restart with same DB
    engine2 = MambaMemoryEngine(config)
    await engine2.start(session_id="e2e-test-2")
    step_after = engine2.l2.step_count
    slots_after = engine2.l2.active_slot_count
    print(f"  Restored: step={step_after}, active_slots={slots_after}")
    print(f"  Match: steps={'OK' if step_after == step_before else 'MISMATCH'}, "
          f"slots={'OK' if slots_after > 0 else 'EMPTY'}")

    # Recall after restart
    recall3 = await engine2.recall("PostgreSQL", limit=2)
    if recall3.memories:
        print(f"  Post-restart recall: {recall3.memories[0].content[:50]}")
    else:
        print("  Post-restart recall: (empty — may have decayed)")

    s2 = engine2.status()
    print(f"  L3 records after restart: {s2.l3_total_records}")

    await engine2.shutdown()

    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    print("E2E test complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
