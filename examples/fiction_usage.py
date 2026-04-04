"""Fiction writing usage example — demonstrates MambaMemory for novel writing.

Covers three genres:
  1. Xianxia/Fantasy — cultivation, sects, artifacts
  2. Romance/Urban — relationships, families, careers
  3. Sci-fi — starships, civilizations, technology

Run: python examples/fiction_usage.py
"""

import asyncio
import shutil
import tempfile

from mamba_memory.presets.fiction import (
    create_fiction_engine,
    fiction_importance_score,
)


async def main():
    tmp = tempfile.mkdtemp(prefix="fiction-")
    db_path = f"{tmp}/fiction.db"
    engine = create_fiction_engine(db_path=db_path, embedding_provider="dummy")
    await engine.start()

    print("=" * 60)
    print("  MambaMemory Fiction Writing Demo")
    print("=" * 60)

    # ─── Gate scoring demo ─────────────────────────────────
    print("\n--- Gate Scoring (fiction mode) ---\n")
    test_cases = [
        ("林月是月影门掌门的独女，性格冷傲但内心善良，精通寒冰剑法", True),
        ("张三突然背叛了天剑宗，投靠了魔教", True),
        ("昆仑山位于大陆西北，是天下第一仙山", True),
        ("修炼体系分为九个境界：练气、筑基、金丹、元婴、化神、合体、渡劫、大乘、飞升", True),
        ("林月爱上了陈风，但陈风是她的师兄", True),
        ("写作要求：第三人称全知视角，节奏紧凑", True),
        ("Captain Nova命令飞船进入曲率航行", True),
        ("陆氏集团的CEO陆远舟暗恋女主角三年了", True),
        ("你好", False),
        ("嗯嗯", False),
        ("ok", False),
        ("今天天气真好", False),
    ]

    for content, expected_store in test_cases:
        score = fiction_importance_score(content)
        mark = "✓" if (score > 0.1) == expected_store else "✗"
        label = "STORE" if score > 0.1 else "DISC"
        print(f"  {mark} {score:.3f} [{label:>5}] {content[:50]}")

    # ─── Xianxia / Fantasy ────────────────────────────────
    print("\n--- Genre: Xianxia/Fantasy ---\n")

    xianxia_data = [
        ("林月是月影门掌门独女，冰灵根天赋，精通寒冰剑法，性格冷傲", ["林月", "月影门"]),
        ("陈风是天剑宗外门弟子，杂灵根，但悟性极高，善于以弱胜强", ["陈风", "天剑宗"]),
        ("月影门位于昆仑山北麓，以冰系功法闻名天下，门下弟子三千", ["月影门", "昆仑山"]),
        ("修炼境界：练气→筑基→金丹→元婴→化神→合体→渡劫→大乘→飞升", []),
        ("玄冰剑是月影门镇派之宝，万年寒冰所铸，剑身自带冰属性攻击", ["玄冰剑", "月影门"]),
        ("陈风在秘境中意外获得上古传承，修为突破至金丹期", ["陈风"]),
        ("林月和陈风在宗门大比中相识，从对手变成了盟友", ["林月", "陈风"]),
        ("魔教突袭月影门，掌门重伤，林月被迫提前接任掌门之位", ["林月", "月影门", "魔教"]),
    ]

    for content, tags in xianxia_data:
        result = await engine.ingest(content, tags=tags)
        status = "STORED" if result.stored else "SKIP"
        print(f"  [{status:>6}] {content[:45]}")

    # ─── Romance / Urban ──────────────────────────────────
    print("\n--- Genre: Romance/Urban ---\n")

    romance_data = [
        ("苏晚是知名律师，毕业于北大法学院，性格独立要强，不信爱情", ["苏晚"]),
        ("陆远舟是陆氏集团CEO，外表冷酷但内心温柔，暗恋苏晚三年", ["陆远舟", "陆氏集团"]),
        ("苏晚的闺蜜林小雨是时尚杂志编辑，性格开朗，是苏晚唯一的倾诉对象", ["林小雨", "苏晚"]),
        ("陆远舟在一次商业纠纷中委托苏晚做代理律师，两人重逢", ["陆远舟", "苏晚"]),
        ("苏晚发现陆远舟就是十年前在雨中为她撑伞的少年", ["苏晚", "陆远舟"]),
        ("陆家老太太反对陆远舟和苏晚在一起，因为苏晚是孤儿出身", ["陆远舟", "苏晚"]),
    ]

    for content, tags in romance_data:
        result = await engine.ingest(content, tags=tags)
        status = "STORED" if result.stored else "SKIP"
        print(f"  [{status:>6}] {content[:45]}")

    # ─── Sci-fi ───────────────────────────────────────────
    print("\n--- Genre: Sci-fi ---\n")

    scifi_data = [
        ("Captain Nova是星际联邦第七舰队旗舰'曙光号'的舰长，曾参加过三次星际战争", ["Nova", "曙光号"]),
        ("曙光号是联邦最新的无畏级战舰，装备了量子跃迁引擎和暗物质护盾", ["曙光号"]),
        ("人类文明已达到二级文明，能够利用恒星能源，但尚未掌握虫洞技术", []),
        ("AI副官'Echo'的智能等级为S级，具有自我意识但服从三大定律", ["Echo", "曙光号"]),
        ("虫族在仙女座星系发起入侵，联邦在前线损失了三个舰队", ["虫族"]),
        ("Nova发现Echo隐瞒了一段关于虫族起源的关键情报", ["Nova", "Echo", "虫族"]),
    ]

    for content, tags in scifi_data:
        result = await engine.ingest(content, tags=tags)
        status = "STORED" if result.stored else "SKIP"
        print(f"  [{status:>6}] {content[:45]}")

    # ─── Recall test ──────────────────────────────────────
    print("\n--- Recall Test ---\n")

    queries = [
        "林月的武功和背景",
        "陈风的修为",
        "苏晚和陆远舟的关系",
        "曙光号的武器装备",
        "谁背叛了谁",
        "月影门在哪里",
    ]

    for q in queries:
        result = await engine.recall(q, limit=2)
        print(f"  Q: {q}")
        if result.memories:
            for m in result.memories:
                print(f"    [{m.layer}] {m.content[:50]}")
        else:
            print("    (no results)")
        print()

    # ─── Status ───────────────────────────────────────────
    print("--- Memory Status ---\n")
    s = engine.status()
    print(f"  L1 window:     {s.l1_window_turns} turns")
    print(f"  L2 slots:      {s.l2_active_slots}/{s.l2_total_slots}")
    print(f"  L3 records:    {s.l3_total_records}")
    print(f"  L3 entities:   {s.l3_entity_count}")

    await engine.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n  Done.")


if __name__ == "__main__":
    asyncio.run(main())
