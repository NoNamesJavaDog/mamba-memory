"""Basic usage example of MambaMemory.

Run: python examples/basic_usage.py
"""

import asyncio

from mamba_memory.sdk import MambaMemory


async def main():
    # Create an in-process memory engine (uses dummy embeddings by default)
    async with MambaMemory(db_path="/tmp/mamba-memory-example.db") as mem:
        print("=== MambaMemory Basic Example ===\n")

        # 1. Ingest some conversations
        print("--- Ingesting conversations ---")
        conversations = [
            "我叫张三，是后端工程师",
            "我们的项目使用Python 3.12和FastAPI",
            "数据库选择了PostgreSQL，部署在port 5432",
            "我决定用Docker Compose管理开发环境",
            "Redis用作缓存层，配置了128MB内存限制",
            "今天天气不错",  # 这条应该被门控过滤掉
            "你好",  # 这条也应该被过滤
            "记住：生产环境的API密钥需要每90天轮换一次",  # 显式记忆请求
        ]

        for msg in conversations:
            result = await mem.ingest(msg)
            status = "STORED" if result.stored else "DISCARDED"
            print(f"  [{status}] {msg[:40]}... → {result.reason[:50]}")

        # 2. Check status
        print("\n--- Memory Status ---")
        status = mem.status()
        print(f"  L1 窗口轮次: {status.l1_window_turns}")
        print(f"  L1 压缩段数: {status.l1_compressed_segments}")
        print(f"  L2 活跃槽位: {status.l2_active_slots}/{status.l2_total_slots}")
        print(f"  L2 步数: {status.l2_step_count}")
        print(f"  L3 记录数: {status.l3_total_records}")

        # 3. Recall
        print("\n--- Recalling memories ---")

        queries = [
            "数据库用的什么？",
            "部署方式",
            "团队成员",
        ]

        for q in queries:
            print(f"\n  Q: {q}")
            result = await mem.recall(q, limit=3)
            if result.memories:
                for m in result.memories:
                    print(f"    [{m.layer}] (score={m.score:.2f}) {m.content[:60]}")
            else:
                print("    (no memories found)")

        # 4. Explicit memory
        print("\n--- Explicit memory ---")
        result = await mem.ingest(
            "下次部署前必须先跑集成测试",
            tags=["deployment", "testing"],
            force=True,
        )
        print(f"  Force stored: {result.stored}, layer: {result.layer}")

        # 5. Forget
        print("\n--- Forgetting ---")
        count = await mem.forget("API密钥")
        print(f"  Forgotten {count} memories about API keys")

        print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
