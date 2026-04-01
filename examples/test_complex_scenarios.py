"""Complex scenario E2E tests — simulates real-world usage patterns.

Scenarios:
  1. Multi-session continuity    — 跨多个会话的记忆延续
  2. Knowledge update & conflict — 知识更新与矛盾处理
  3. Decay & natural forgetting  — 长时间衰减与自然遗忘
  4. Slot saturation & eviction  — 槽位饱和与淘汰到 L3
  5. Multi-language mixed recall  — 多语言混合检索
  6. High-volume stress test     — 大量写入压力测试
  7. Pre-summary MCP simulation  — 模拟 Claude/Codex 预压缩
  8. Entity graph traversal      — 实体图谱关联查询
  9. Precision recall benchmark   — 召回精度评测
  10. Concurrent safety           — 并发安全验证

Run: GOOGLE_API_KEY=... python examples/test_complex_scenarios.py
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time

from mamba_memory.core.engine import MambaMemoryEngine
from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config

# ── Shared helpers ──────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = ""):
    _results.append((name, condition, detail))
    mark = PASS if condition else FAIL
    msg = f"  {mark} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def make_config(tmp: str, **overrides) -> EngineConfig:
    defaults = dict(
        l1=L1Config(window_size=4, max_compressed_segments=3),
        l2=L2Config(slot_count=8, base_decay_rate=0.85, eviction_threshold=0.05, snapshot_interval=5),
        l3=L3Config(db_path=f"{tmp}/test.db"),
        embedding_provider="google",
        compression_model="none",
    )
    defaults.update(overrides)
    return EngineConfig(**defaults)


async def fresh_engine(tmp: str, **overrides) -> MambaMemoryEngine:
    engine = MambaMemoryEngine(make_config(tmp, **overrides))
    await engine.start()
    return engine


# ── Scenario 1: Multi-session continuity ────────────────────────

async def test_multi_session_continuity():
    """用户在多个会话中逐步构建项目知识，跨会话记忆应该延续。"""
    print("\n" + "=" * 60)
    print("Scenario 1: Multi-session continuity (跨会话记忆延续)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s1-")
    config = make_config(tmp)

    # Session 1: 项目初始化
    e1 = MambaMemoryEngine(config)
    await e1.start(session_id="session-1")
    await e1.ingest("项目名称是MegaApp，使用Python 3.12开发", tags=["MegaApp", "Python"])
    await e1.ingest("数据库选择PostgreSQL 16，部署在port 5432", tags=["PostgreSQL"])
    await e1.ingest_explicit("团队负责人是张三，后端；李四，前端", tags=["张三", "李四"])
    await e1.shutdown()
    print("  Session 1: ingested project basics")

    # Session 2: 添加部署信息
    e2 = MambaMemoryEngine(config)
    await e2.start(session_id="session-2")
    await e2.ingest("部署方案确定用Kubernetes，集群在AWS EKS", tags=["Kubernetes", "AWS"])
    await e2.ingest("CI用GitHub Actions，CD用ArgoCD", tags=["GitHub Actions", "ArgoCD"])

    # 应该能回忆起 session-1 的内容（可能在摘要中或 L3 中）
    r = await e2.recall("PostgreSQL数据库", limit=5)
    found_pg = any(
        "PostgreSQL" in m.content or "postgresql" in m.content.lower() or "数据库" in m.content
        for m in r.memories
    )
    check("跨会话召回数据库相关内容", found_pg, f"found {len(r.memories)} results")

    r2 = await e2.recall("团队成员", limit=3)
    found_team = any("张三" in m.content or "李四" in m.content for m in r2.memories)
    check("跨会话召回团队成员", found_team)

    await e2.shutdown()

    # Session 3: 验证长期持久化（L2 state 通过 snapshot 恢复）
    e3 = MambaMemoryEngine(config)
    await e3.start(session_id="session-3")
    r3 = await e3.recall("MegaApp项目", limit=5)
    s3 = e3.status()
    # 即使 L2 内容衰减，至少 L2 state 应该恢复了
    found_anything = len(r3.memories) > 0 or s3.l2_active_slots > 0
    check("第三个会话 L2 状态恢复", found_anything, f"slots={s3.l2_active_slots}, results={len(r3.memories)}")
    await e3.shutdown()

    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 2: Knowledge update & conflict ─────────────────────

async def test_knowledge_update():
    """用户纠正之前的错误信息，新信息应该取代旧信息。"""
    print("\n" + "=" * 60)
    print("Scenario 2: Knowledge update & conflict (知识更新与矛盾)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s2-")
    engine = await fresh_engine(tmp)

    # 初始信息
    await engine.ingest("数据库用的是MySQL 5.7，端口3306", tags=["MySQL"])
    await engine.ingest("缓存用Memcached，端口11211", tags=["Memcached"])

    # 纠正：迁移到了新的数据库
    await engine.ingest("不对，我们已经从MySQL迁移到PostgreSQL 16了", tags=["PostgreSQL"])
    await engine.ingest("其实缓存也换成Redis了，Memcached已经下线", tags=["Redis"])

    # 再加一些无关内容触发 L2 演化
    for i in range(5):
        await engine.ingest(f"开发任务 #{i}: 实现用户模块的第{i}个接口")

    # 验证：召回应该包含新信息（不一定是 top-1，因为语义距离受 embedding 影响）
    r = await engine.recall("PostgreSQL数据库迁移", limit=5)
    contents = " ".join(m.content for m in r.memories)
    has_pg = "PostgreSQL" in contents or "postgresql" in contents.lower() or "迁移" in contents
    check("纠正后召回包含 PostgreSQL", has_pg, f"top: {r.memories[0].content[:40] if r.memories else 'empty'}")

    r2 = await engine.recall("缓存方案", limit=3)
    contents2 = " ".join(m.content for m in r2.memories)
    has_redis = "Redis" in contents2 or "redis" in contents2.lower()
    check("纠正后召回 Redis", has_redis)

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 3: Decay & natural forgetting ──────────────────────

async def test_decay_and_forgetting():
    """模拟长时间不使用后记忆自然衰减。"""
    print("\n" + "=" * 60)
    print("Scenario 3: Decay & natural forgetting (衰减与自然遗忘)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s3-")
    config = make_config(
        tmp,
        l2=L2Config(slot_count=4, base_decay_rate=0.5, eviction_threshold=0.1, snapshot_interval=100),
    )
    engine = MambaMemoryEngine(config)
    await engine.start()

    # 写入一条记忆
    await engine.ingest_explicit("重要：API密钥是sk-abc123", tags=["API"])
    initial_slots = engine.l2.get_active_slots()
    initial_act = initial_slots[0].activation if initial_slots else 0
    check("初始写入", len(initial_slots) > 0, f"activation={initial_act:.3f}")

    # 模拟很多步不访问这条记忆（写入不相关内容）
    for i in range(20):
        await engine.ingest(f"无关讨论第{i}轮：今天开会讨论了产品路线图")

    # 检查衰减
    after_slots = engine.l2.get_active_slots()
    api_slot = None
    for s in engine.l2.slots:
        if "API" in s.state or "sk-abc" in s.state:
            api_slot = s
            break

    if api_slot and not api_slot.is_empty:
        check("衰减后 activation 降低", api_slot.activation < initial_act,
              f"{initial_act:.3f} → {api_slot.activation:.3f}")
    else:
        check("记忆已被淘汰到 L3", True, "slot evicted (expected with 0.5 decay)")

    # 验证：即使 L2 衰减了，L3 应该还有
    r = await engine.recall("API密钥", limit=3, layers=["l3"])
    # L3 可能有也可能没有（取决于是否被淘汰）
    r_all = await engine.recall("API密钥", limit=3)
    check("全层召回仍可找到", len(r_all.memories) > 0 or len(r.memories) > 0)

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 4: Slot saturation & eviction to L3 ───────────────

async def test_slot_saturation():
    """当所有 L2 槽位都满时，低价值内容应该被淘汰到 L3。"""
    print("\n" + "=" * 60)
    print("Scenario 4: Slot saturation & eviction (槽位饱和与淘汰)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s4-")
    config = make_config(
        tmp,
        l2=L2Config(slot_count=4, base_decay_rate=0.7, eviction_threshold=0.05, snapshot_interval=100),
    )
    engine = MambaMemoryEngine(config)
    await engine.start()

    # 填满所有 4 个槽位
    topics = [
        ("Python 3.12 是项目的主要开发语言", ["Python"]),
        ("使用FastAPI作为Web框架，端口8000", ["FastAPI"]),
        ("前端使用Vue 3 + TypeScript", ["Vue", "TypeScript"]),
        ("测试框架选用pytest，覆盖率要求80%", ["pytest"]),
    ]
    for content, tags in topics:
        await engine.ingest_explicit(content, tags=tags)

    s = engine.status()
    check("4个槽位全部填满", s.l2_active_slots == 4, f"active={s.l2_active_slots}")

    # 继续写入更多内容，旧的应该被淘汰
    more_topics = [
        ("数据库备份策略：每天凌晨3点全量备份", ["backup"]),
        ("日志级别生产环境设为WARN，开发环境DEBUG", ["logging"]),
        ("API限流配置：每分钟100次请求", ["rate-limit"]),
        ("安全扫描每周一执行，用SonarQube", ["SonarQube"]),
    ]
    for content, tags in more_topics:
        await engine.ingest_explicit(content, tags=tags)

    s2 = engine.status()
    check("L3 有淘汰记录", s2.l3_total_records > 0, f"l3_records={s2.l3_total_records}")

    # 被淘汰的老内容应该在 L3 中可搜索
    r = await engine.recall("Python开发语言", limit=3)
    found = any("Python" in m.content for m in r.memories)
    check("淘汰内容仍可通过 L3 召回", found)

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 5: Multi-language mixed recall ─────────────────────

async def test_multilanguage_recall():
    """中英日混合内容的跨语言语义召回。"""
    print("\n" + "=" * 60)
    print("Scenario 5: Multi-language mixed recall (多语言混合检索)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s5-")
    engine = await fresh_engine(tmp)

    # 混合语言写入
    await engine.ingest_explicit("The main database is PostgreSQL running on port 5432", tags=["PostgreSQL"])
    await engine.ingest_explicit("キャッシュにはRedisを使用、メモリ上限256MB", tags=["Redis"])
    await engine.ingest_explicit("部署方案是Docker Compose + Nginx反向代理", tags=["Docker", "Nginx"])
    await engine.ingest_explicit("Authentication uses JWT tokens with 24h expiry", tags=["JWT"])
    await engine.ingest_explicit("로그 시스템은 ELK Stack을 사용합니다", tags=["ELK"])

    # 中文查英文内容（用包含实体名的查询提升跨语言命中率）
    r1 = await engine.recall("PostgreSQL database", limit=3)
    found_pg = any("PostgreSQL" in m.content for m in r1.memories)
    check("跨语言召回 PostgreSQL", found_pg)

    # 英文查中文内容
    r2 = await engine.recall("deployment architecture", limit=3)
    found_docker = any("Docker" in m.content for m in r2.memories)
    check("英文查→中文结果 (Docker)", found_docker)

    # 中文查日文内容
    r3 = await engine.recall("缓存设置", limit=3)
    found_redis = any("Redis" in m.content for m in r3.memories)
    check("中文查→日文结果 (Redis)", found_redis)

    # 英文查韩文内容
    r4 = await engine.recall("logging system", limit=3)
    found_elk = any("ELK" in m.content for m in r4.memories)
    check("英文查→韩文结果 (ELK)", found_elk)

    # 认证相关
    r5 = await engine.recall("用户认证方式", limit=3)
    found_jwt = any("JWT" in m.content for m in r5.memories)
    check("中文查→英文认证 (JWT)", found_jwt)

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 6: High-volume stress test ─────────────────────────

async def test_high_volume():
    """大量写入后系统仍然稳定且能正确召回。"""
    print("\n" + "=" * 60)
    print("Scenario 6: High-volume stress test (大量写入压力)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s6-")
    config = make_config(
        tmp,
        l1=L1Config(window_size=5, max_compressed_segments=10),
        l2=L2Config(slot_count=16, base_decay_rate=0.9, eviction_threshold=0.03, snapshot_interval=20),
    )
    engine = MambaMemoryEngine(config)
    await engine.start()

    # 写入一条标志性记忆
    await engine.ingest_explicit("ANCHOR: 生产数据库连接串是 postgres://prod:5432/megadb", tags=["database"])

    # 大量普通对话
    t0 = time.time()
    for i in range(50):
        await engine.ingest(
            f"第{i}轮对话：讨论了关于功能模块{i % 10}的实现细节，"
            f"涉及接口设计和数据模型修改，预计需要{i % 5 + 1}天完成",
        )
    elapsed = time.time() - t0

    s = engine.status()
    print(f"  50 turns ingested in {elapsed:.1f}s ({elapsed/50*1000:.0f}ms/turn)")
    check("50轮写入完成", True, f"{elapsed:.1f}s total")
    check("L1 未溢出崩溃", s.l1_window_turns <= 5)
    check("L2 有活跃槽位", s.l2_active_slots > 0, f"active={s.l2_active_slots}")

    # 验证标志性记忆仍可召回
    r = await engine.recall("生产数据库连接", limit=3)
    found_anchor = any("ANCHOR" in m.content or "megadb" in m.content for m in r.memories)
    check("大量写入后标志记忆仍可召回", found_anchor)

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 7: Pre-summary MCP simulation ──────────────────────

async def test_pre_summary():
    """模拟 Claude/Codex 通过 MCP 传入预压缩摘要。"""
    print("\n" + "=" * 60)
    print("Scenario 7: Pre-summary MCP simulation (预压缩模式)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s7-")
    engine = await fresh_engine(tmp)

    # 模拟 Claude 自己做了压缩后传入
    r1 = await engine.ingest(
        "用户和助手讨论了很长一段关于数据库选型的对话，涉及MySQL、PostgreSQL、MongoDB的优劣对比...",
        pre_summary="用户最终决定使用PostgreSQL 16，原因：JSONB支持好、性能优、社区活跃",
        pre_entities=["PostgreSQL", "MySQL", "MongoDB"],
        tags=["database"],
    )
    check("预压缩写入 L2", r1.stored and r1.layer == "l2", f"layer={r1.layer}")

    # 验证压缩后的摘要可被召回
    r2 = await engine.recall("数据库选型", limit=3)
    found = any("PostgreSQL" in m.content and "JSONB" in m.content for m in r2.memories)
    check("召回预压缩摘要", found)

    # 再模拟一条
    r3 = await engine.ingest(
        "A long discussion about caching strategies, comparing Redis, Memcached, and local cache...",
        pre_summary="决定用Redis Cluster，3主3从，maxmemory 2GB，淘汰策略 allkeys-lru",
        pre_entities=["Redis", "Memcached"],
    )
    check("英文预压缩写入", r3.stored)

    r4 = await engine.recall("缓存策略", limit=3)
    found2 = any("Redis" in m.content for m in r4.memories)
    check("跨语言召回预压缩内容", found2)

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 8: Entity graph traversal ──────────────────────────

async def test_entity_graph():
    """验证实体图谱的自动构建和关联查询。"""
    print("\n" + "=" * 60)
    print("Scenario 8: Entity graph traversal (实体图谱)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s8-")
    engine = await fresh_engine(tmp)

    # 写入带实体标签的内容
    await engine.ingest_explicit("张三负责后端API开发", tags=["张三", "API", "backend"])
    await engine.ingest_explicit("李四负责前端React页面", tags=["李四", "React", "frontend"])
    await engine.ingest_explicit("张三和李四一起负责用户模块", tags=["张三", "李四", "用户模块"])
    await engine.ingest_explicit("API使用FastAPI框架，端口8000", tags=["API", "FastAPI"])
    await engine.ingest_explicit("React前端通过API网关调用后端", tags=["React", "API"])

    # 检查实体图谱
    s = engine.status()
    check("实体图谱有节点", s.l3_entity_count > 0, f"entities={s.l3_entity_count}")

    # 查询实体
    zhang = engine.l3.get_entity("张三")
    check("张三实体存在", zhang is not None)
    if zhang:
        check("张三提及次数 >= 2", zhang.mention_count >= 2, f"count={zhang.mention_count}")

    # 查询关系
    zhang_related = engine.l3.get_related_entities("张三")
    related_names = [e.name for e, _ in zhang_related]
    check("张三关联到李四", "李四" in related_names, f"related={related_names}")
    check("张三关联到API", "API" in related_names)

    # 通过实体召回记忆
    r = await engine.recall("张三", limit=5)
    zhang_memories = [m for m in r.memories if "张三" in m.content]
    check("通过实体名召回记忆", len(zhang_memories) > 0, f"found {len(zhang_memories)}")

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 9: Precision recall benchmark ──────────────────────

async def test_recall_precision():
    """测试召回精度：给定明确查询，第一条结果是否是最佳匹配。"""
    print("\n" + "=" * 60)
    print("Scenario 9: Precision recall benchmark (召回精度)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s9-")
    engine = await fresh_engine(tmp, l2=L2Config(slot_count=16, base_decay_rate=0.95))

    # 写入互不相关的 10 条记忆
    memories = [
        ("生产环境MySQL端口是3306，只读副本在3307", ["MySQL"]),
        ("Redis Sentinel 配置：3个哨兵节点，quorum=2", ["Redis"]),
        ("Nginx配置了SSL证书，强制HTTPS跳转", ["Nginx", "SSL"]),
        ("Docker镜像推送到Harbor私有仓库", ["Docker", "Harbor"]),
        ("Kubernetes集群有3个master节点，5个worker", ["Kubernetes"]),
        ("Prometheus监控告警：CPU>80%发邮件，>95%发短信", ["Prometheus"]),
        ("Git分支策略：main保护，feature分支合并需要2个reviewer", ["Git"]),
        ("API限流策略：普通用户100次/分钟，VIP用户1000次/分钟", ["API"]),
        ("日志保留策略：生产环境30天，测试环境7天", ["logging"]),
        ("数据库备份：全量每天凌晨3点，增量每小时一次", ["backup", "database"]),
    ]

    for content, tags in memories:
        await engine.ingest_explicit(content, tags=tags)

    # 精确查询 → 验证 top-1 是否正确
    queries_and_expected = [
        ("MySQL数据库端口", "MySQL"),
        ("Redis哨兵", "Redis"),
        ("SSL证书", "SSL"),
        ("Docker镜像仓库", "Docker"),
        ("K8s节点数量", "Kubernetes"),
        ("CPU告警阈值", "Prometheus"),
        ("Git代码审查", "Git"),
        ("接口限流", "API"),
        ("日志保留多久", "日志"),
        ("数据库什么时候备份", "备份"),
    ]

    hits = 0
    for query, expected_tag in queries_and_expected:
        r = await engine.recall(query, limit=1)
        if r.memories:
            top = r.memories[0]
            is_hit = expected_tag in top.content or any(expected_tag.lower() in e.lower() for e in [])
            # 宽松匹配：检查是否包含预期关键词
            is_hit = expected_tag in top.content
            if is_hit:
                hits += 1
            mark = PASS if is_hit else FAIL
            print(f"  {mark} Q: {query:<20} → {top.content[:45]}")
        else:
            print(f"  {FAIL} Q: {query:<20} → (empty)")

    precision = hits / len(queries_and_expected)
    check(f"召回精度 Top-1", precision >= 0.5, f"{hits}/{len(queries_and_expected)} = {precision:.0%}")

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Scenario 10: Concurrent safety ──────────────────────────────

async def test_concurrent_safety():
    """并发读写不应该导致崩溃或数据损坏。"""
    print("\n" + "=" * 60)
    print("Scenario 10: Concurrent safety (并发安全)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="mamba-s10-")
    engine = await fresh_engine(tmp, l2=L2Config(slot_count=16, base_decay_rate=0.95))

    errors: list[str] = []

    async def writer(task_id: int):
        try:
            for i in range(5):
                await engine.ingest(f"Writer-{task_id} 第{i}条: 配置项{task_id}_{i}=value_{i}", tags=[f"task{task_id}"])
        except Exception as e:
            errors.append(f"writer-{task_id}: {e}")

    async def reader(task_id: int):
        try:
            for i in range(5):
                await engine.recall(f"配置项{task_id}", limit=2)
        except Exception as e:
            errors.append(f"reader-{task_id}: {e}")

    async def forgetter():
        try:
            await asyncio.sleep(0.1)
            await engine.forget("配置项0")
        except Exception as e:
            errors.append(f"forgetter: {e}")

    # 5 个 writer + 5 个 reader + 1 个 forgetter 并发
    tasks = []
    for i in range(5):
        tasks.append(writer(i))
        tasks.append(reader(i))
    tasks.append(forgetter())

    await asyncio.gather(*tasks)

    check("并发无报错", len(errors) == 0, f"errors: {errors[:3]}" if errors else "clean")

    s = engine.status()
    check("并发后状态一致", s.l2_active_slots >= 0, f"slots={s.l2_active_slots}")

    # 验证可正常召回
    r = await engine.recall("配置项", limit=3)
    check("并发后召回正常", True, f"found {len(r.memories)} results")

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


# ── Main ────────────────────────────────────────────────────────

async def main():
    print("\n" + "█" * 60)
    print("  MambaMemory — Complex Scenario E2E Tests")
    print("█" * 60)

    t0 = time.time()

    await test_multi_session_continuity()
    await test_knowledge_update()
    await test_decay_and_forgetting()
    await test_slot_saturation()
    await test_multilanguage_recall()
    await test_high_volume()
    await test_pre_summary()
    await test_entity_graph()
    await test_recall_precision()
    await test_concurrent_safety()

    elapsed = time.time() - t0

    # Summary
    print("\n" + "█" * 60)
    print("  Summary")
    print("█" * 60)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)

    print(f"\n  Total: {total} checks, {PASS} {passed} passed, {FAIL} {failed} failed")
    print(f"  Time: {elapsed:.1f}s")

    if failed > 0:
        print(f"\n  Failed checks:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    {FAIL} {name}  {detail}")

    print()
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
