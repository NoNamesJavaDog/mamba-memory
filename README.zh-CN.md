# MambaMemory

**AI Agent 认知记忆引擎** — 不是又一个向量数据库，而是一个会思考的记忆系统。

受 [Mamba3](https://arxiv.org/abs/2312.00752)（选择性状态空间模型）启发，MambaMemory 将 SSM 核心思想应用于 AI Agent 记忆管理：选择性门控、状态衰减、固定容量演化、输入驱动召回。

```
输入 → [L1 会话层] → 溢出 → [L2 状态层 / SSM] → 淘汰 → [L3 持久层]
        最近几轮     门控     认知槽位          归档    SQLite + HNSW
        完整保真     过滤     衰减与演化        永久    向量检索
```

## 为什么不直接用向量数据库？

| | 向量数据库 (Chroma, Qdrant) | MambaMemory |
|---|---|---|
| **模型** | 被动仓库：存进去，搜出来 | 主动大脑：过滤、演化、衰减、召回 |
| **容量** | 无限增长，用户自己清理 | 固定槽位，自动压缩淘汰 |
| **写入** | 全量写入 | 选择性门控过滤噪声（准确率 94.5%） |
| **检索** | 静态相似度匹配 | 语义主导 + 时间感知衰减 + 间隔重复 |
| **旧记忆** | 永远存在直到手动删除 | 艾宾浩斯遗忘曲线 + 间隔重复 |
| **多语言** | 取决于 embedding | 内置中英日韩文本分析 + 任意 embedding 模型 |
| **扩展** | 单一后端 | 可插拔：SQLite / PostgreSQL / Redis |

MambaMemory 不是为百万文档 RAG 设计的。它专为 **单 Agent 个人记忆** 打造 — 记住重要的，忘掉不重要的，随对话不断演化。

## 安装

```bash
pip install mamba-memory

# 使用 Google Gemini 嵌入向量（推荐）
pip install mamba-memory[google]

# 使用 OpenAI 嵌入向量
pip install mamba-memory[openai]

# 使用 MCP + HTTP 服务
pip install mamba-memory[server]

# 全部安装
pip install mamba-memory[all]
```

需要 Python 3.11+。

## 快速上手

```python
import asyncio
from mamba_memory.sdk import MambaMemory

async def main():
    async with MambaMemory() as mem:
        await mem.ingest("我决定把数据库切换到PostgreSQL，端口5432")
        await mem.ingest("你好！")  # 门控丢弃
        await mem.ingest("记住：API密钥每90天必须轮换", force=True)

        result = await mem.recall("数据库配置")
        for m in result.memories:
            print(f"[{m.layer}] (score={m.score:.2f}) {m.content}")

asyncio.run(main())
```

### MCP 模式（Claude Code / Codex / OpenClaw）

不需要 API key — LLM 客户端**本身就是**压缩器：

```python
memory_ingest({
    "content": "关于数据库迁移的长段对话...",
    "summary": "用户决定使用PostgreSQL 16，原因：JSONB + 性能",
    "entities": ["PostgreSQL", "MySQL"],
})
```

## 架构

### 三层设计

```
┌─────────────────────────────────────────────────────────────────┐
│  L1  会话层 — 最近 K 轮 + 递归摘要链                             │
│  延迟: <1ms  |  容量: 固定窗口  |  存储: 内存                     │
└──────────────────────┬──────────────────────────────────────────┘
                  Gate ↕ 规则引擎 + 学习型分类器（混合）
┌──────────────────────┴──────────────────────────────────────────┐
│  L2  状态层 (SSM 核心) — 64 个认知槽位 + 艾宾浩斯衰减            │
│  延迟: <10ms  |  容量: 固定维度  |  存储: 内存                    │
└──────────────────────┬──────────────────────────────────────────┘
             Sync/Query ↕ 持久化 / 检索 / 知识图谱
┌──────────────────────┴──────────────────────────────────────────┐
│  L3  持久层 — SQLite/PostgreSQL + HNSW + 类型化知识图谱          │
│  延迟: <200ms  |  容量: 无上限  |  存储: 磁盘                     │
└─────────────────────────────────────────────────────────────────┘
```

### 核心机制

**选择性门控（混合模式）**

1. **规则引擎**（始终生效）— 5 维加权评分：意图(0.25) + 结构化数据(0.25) + 显式记忆(0.20) + 行动/任务(0.15) + 信息密度(0.15)
2. **学习型分类器**（可选）— 15 维特征 + 逻辑回归，在标注数据上训练。启用后用其置信度替代规则引擎的重要性评分，同时规则引擎仍处理槽位分配。

**时间感知衰减（艾宾浩斯遗忘曲线）**

```
步数衰减:  activation *= base_decay_rate              (每次交互)
时间衰减:  activation *= 2^(-Δt / 有效半衰期)          (墙钟时间)
复习效应:  有效半衰期 *= (1 + 0.5 * 被召回次数)
```

被召回 5 次的记忆半衰期是未召回的 **3.5 倍**。实现了间隔重复——常用的记忆越用越持久，不用的自然淡化。

**递归摘要链** — L1 压缩段溢出时，不逐个推给 L2，而是先合并为"超级摘要"：

```
对话轮次 → 摘要段 → [溢出] → 合并(N段) → 超级摘要 → L2 门控
```

跨更长时间范围生成更密集的摘要。

**知识图谱推理** — 实体自动提取并建立 10 种类型化关系（uses、depends_on、replaces、deployed_on、connects_to 等）。支持多跳邻域扩展、路径查找、间接关系推理。

### 召回排序

| 信号 | 权重 | 来源 |
|------|------|------|
| 语义相似度 | **0.60** | Embedding 余弦距离 |
| 活跃度 | 0.15 | 衰减历史（含时间衰减 + 复习效应） |
| 近因 | 0.15 | 距上次访问的步数 |
| 重要性 | 0.10 | 初始门控评分 |

被召回的槽位获得活跃度提升 + 复习计数递增，形成间隔重复正反馈循环。

## 服务模式

### MCP Server

```bash
mamba-memory serve --mcp
```

工具：`memory_ingest`、`memory_recall`、`memory_forget`、`memory_status`、`memory_compact`

### HTTP API

```bash
mamba-memory serve --http --port 8420
```

认证：设置 `MAMBA_MEMORY_API_KEY` 启用 Bearer token 认证。状态面板：`http://localhost:8420/ui`

### CLI

```bash
mamba-memory init                     # 交互式初始化
mamba-memory serve --mcp              # MCP 服务
mamba-memory serve --http             # HTTP API
mamba-memory status --detail slots    # 查看认知槽位
mamba-memory compact --layer all      # 强制压缩
mamba-memory export -o memories.json  # 导出全部记忆
```

### Docker

```bash
docker compose up -d
```

## 配置

```python
config = EngineConfig(
    l2=L2Config(
        slot_count=64,
        base_decay_rate=0.98,           # 步数衰减（benchmark 最优）
        time_decay_enabled=True,        # 艾宾浩斯时间衰减
        time_decay_halflife=3600.0,     # 基础半衰期 1 小时
    ),
    embedding_provider="google",
    compression_model="none",
    namespace="default",                # 多 agent 隔离
)
```

支持 YAML 配置文件（`~/.mamba-memory/config.yaml`）和环境变量覆盖。

## Benchmark 结果

| 指标 | 规则引擎 | 学习型 Gate |
|------|---------|------------|
| 准确率 | **94.5%** | **90.9%** |
| 精确率 | **100%** | — |
| F1 | **0.945** | — |

学习型 Gate 能正确分类规则引擎漏判的 case（如 "Database backup runs daily at 3am" → conf=0.85）。混合模式结合两者优势。

最优参数：`decay_rate=0.98`，`semantic_weight=0.60`。

## 项目结构

```
mamba_memory/
├── core/
│   ├── types.py              # 数据模型
│   ├── embedding.py          # Embedding (Google/OpenAI/本地/Dummy)
│   ├── engine.py             # 主引擎
│   ├── text.py               # 文本分析
│   ├── llm.py                # LLM 后端 (Google/OpenAI/Anthropic)
│   ├── l1/session.py         # 会话层 + 递归摘要链
│   ├── l2/
│   │   ├── gate.py           # 混合门控（规则 + 学习）
│   │   ├── learned_gate.py   # 15 维逻辑回归分类器
│   │   ├── evolver.py        # 双重衰减（步数 + 艾宾浩斯）
│   │   ├── recaller.py       # 语义加权召回 + 复习追踪
│   │   └── state.py          # 状态管理器
│   └── l3/
│       ├── store.py          # SQLite + HNSW + 实体图谱 + 迁移
│       ├── knowledge_graph.py # 类型化关系 + 多跳推理
│       └── backend.py        # 可插拔后端 (PostgreSQL/Redis)
├── server/
│   ├── mcp/server.py         # MCP Server
│   └── http/                 # FastAPI + 认证 + Web UI
├── sdk/client.py             # Python SDK
├── config.py                 # YAML 配置
└── cli.py                    # CLI
```

**5,600+ 行源码，29 个模块，71 单元测试 + 10 MCP 集成测试。**

## 竞品对比

| | Mem0 | Zep | LangMem | MambaMemory |
|---|---|---|---|---|
| 核心模型 | 图 + 向量 | 时序知识图谱 | LangChain | SSM 状态机 |
| 容量管理 | 手动 | 基于时间 | 手动 | 自动（艾宾浩斯 + 淘汰） |
| 写入过滤 | 无 | 无 | 无 | 混合门控（F1=0.945） |
| 遗忘模型 | 无 | 衰减 | 无 | 艾宾浩斯 + 间隔重复 |
| 学习型门控 | 无 | 无 | 无 | 有（逻辑回归） |
| 知识图谱 | 有 | 有 | 无 | 有（类型化 + 推理） |
| 递归压缩 | 无 | 无 | 无 | 有（摘要链） |
| 多语言 | 取决于模型 | 取决于模型 | 取决于模型 | 内置中英日韩 |
| MCP 原生 | 无 | 无 | 无 | 有 |
| 可自托管 | 部分 | 部分 | 是 | 是（SQLite，零依赖） |

## 许可证

MIT
