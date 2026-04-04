"""Fiction writing preset — adapts MambaMemory for novel/story writing.

Supports: xianxia/fantasy, romance/urban, sci-fi, and general fiction.

Changes from default (technical) mode:
  1. Gate: fiction-aware importance signals (character, plot, world-building, relationship)
  2. Decay: much slower — characters don't fade between chapters
  3. Entity types: character, location, faction, artifact, event, concept
  4. Relation types: loves, hates, master_of, member_of, killed, betrayed... (14 types)
  5. Config: 128 slots, 500 tokens/slot, 0.995 decay rate

Usage:
    from mamba_memory.presets.fiction import create_fiction_engine

    engine = create_fiction_engine(db_path="~/.mamba-memory/my-novel.db")
    await engine.start()

    await engine.ingest(
        "林月是月影门掌门的独女，性格冷傲但内心善良，精通寒冰剑法",
        tags=["林月", "月影门"],
    )
    result = await engine.recall("林月的武功")
"""

from __future__ import annotations

import re
from typing import Any

from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config


# ╔══════════════════════════════════════════════════════════════╗
# ║  Fiction Config                                              ║
# ╚══════════════════════════════════════════════════════════════╝

def create_fiction_config(
    db_path: str = "~/.mamba-memory/fiction.db",
    **overrides: Any,
) -> EngineConfig:
    """Create an EngineConfig optimized for fiction writing."""
    return EngineConfig(
        l1=L1Config(
            window_size=10,
            max_compressed_segments=30,
        ),
        l2=L2Config(
            slot_count=128,
            slot_max_tokens=500,
            base_decay_rate=0.995,
            eviction_threshold=0.02,
            snapshot_interval=20,
            time_decay_enabled=True,
            time_decay_halflife=86400.0,
            activation_boost=1.2,
        ),
        l3=L3Config(db_path=db_path),
        embedding_provider=overrides.get("embedding_provider", "auto"),
        compression_model=overrides.get("compression_model", "none"),
        namespace=overrides.get("namespace", "fiction"),
    )


def create_fiction_engine(db_path: str = "~/.mamba-memory/fiction.db", **kwargs):
    """Create a MambaMemoryEngine with fiction-optimized settings."""
    from mamba_memory.core.engine import MambaMemoryEngine
    config = create_fiction_config(db_path, **kwargs)
    return MambaMemoryEngine(config)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Fiction Gate — importance scoring for novel content          ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Character signals (all fiction types) ---
_CHARACTER_PATTERNS = re.compile(
    r"(名叫|叫做|人称|绰号|外号|化名|真名|本名|"
    r"是个|是一个|是一位|是一名|"
    r"身高|年龄|长相|容貌|外貌|肤色|发色|眼睛|"
    r"性格|特点|特征|习惯|口头禅|"
    r"背景|身世|出身|来历|来自|"
    # Xianxia/fantasy specific
    r"修为|修炼|境界|突破|渡劫|元婴|金丹|筑基|练气|化神|"
    r"武功|剑法|刀法|拳法|掌法|身法|轻功|内功|心法|功法|"
    r"灵根|血脉|天赋|资质|体质|"
    r"法宝|神器|灵器|仙器|丹药|灵药|"
    # Sci-fi specific
    r"改造|植入|义体|基因|克隆|AI等级|"
    r"舰船|型号|武器系统|防御等级|"
    # Romance/urban specific
    r"职业|公司|职位|学历|家庭|"
    # English
    r"named\s|known\s+as|nicknamed|alias|"
    r"personality|appearance|background|"
    r"specializes\s+in|ability|skill|power|"
    r"cultivation\s+level|rank|class)",
    re.IGNORECASE,
)

# --- Plot event signals ---
_PLOT_PATTERNS = re.compile(
    r"(发生了|突然|终于|原来|没想到|竟然|却|不料|万万没想到|"
    r"被杀|死了|复活|苏醒|觉醒|暴露|"
    r"背叛|投降|结盟|宣战|决裂|和好|"
    r"发现了|揭露|真相是|秘密是|"
    r"失踪|被抓|逃脱|获救|重伤|痊愈|"
    r"继承|登基|退位|被废|"
    r"比赛|比武|大战|决战|围攻|突围|"
    # Chapter/timeline markers
    r"第.{1,4}章|chapter\s+\d|"
    r"三年后|多年后|次日|翌日|当晚|"
    r"years?\s+later|the\s+next\s+day|that\s+night|"
    # English plot words
    r"happened|suddenly|finally|revealed|discovered|"
    r"killed|died|betrayed|allied|declared\s+war|"
    r"turns\s+out|plot\s+twist|secret\s+is|"
    r"escaped|captured|rescued|awakened)",
    re.IGNORECASE,
)

# --- Relationship signals ---
_RELATIONSHIP_PATTERNS = re.compile(
    r"(爱上了?|喜欢上|暗恋|表白|结婚|离婚|分手|"
    r"仇恨|仇人|世仇|血仇|"
    r"师父|师母|师兄|师姐|师弟|师妹|徒弟|弟子|"
    r"兄弟|姐妹|父亲|母亲|儿子|女儿|爷爷|奶奶|"
    r"盟友|同伴|队友|搭档|战友|"
    r"对手|宿敌|情敌|死对头|"
    r"上司|下属|同事|青梅竹马|"
    # English
    r"fell\s+in\s+love|loves|hates|"
    r"master|apprentice|disciple|"
    r"ally|companion|rival|nemesis|enemy|"
    r"married|divorced|siblings|parent|"
    r"childhood\s+friend|partner|boss|subordinate)",
    re.IGNORECASE,
)

# --- World-building signals ---
_WORLD_PATTERNS = re.compile(
    r"(位于|坐落|地处|疆域|领地|版图|"
    r"门派|宗门|势力|帮派|教派|家族|"
    r"王国|帝国|皇朝|联邦|星球|星系|殖民地|"
    r"修炼体系|功法等级|境界划分|"
    r"天地法则|禁忌|规则|律法|"
    r"历史|传说|起源|纪元|上古|远古|"
    r"大陆|海域|秘境|禁地|副本|"
    # Sci-fi specific
    r"超光速|曲率|虫洞|跃迁|"
    r"文明等级|科技树|能源|"
    # English
    r"located\s+(in|at)|territory|"
    r"faction|kingdom|empire|clan|guild|sect|"
    r"magic\s+system|cultivation|power\s+system|"
    r"planet|galaxy|colony|civilization|"
    r"history|legend|origin|era|epoch|ancient)",
    re.IGNORECASE,
)

# --- Style / POV / writing rules ---
_STYLE_PATTERNS = re.compile(
    r"(视角|人称|叙事|风格|语气|基调|节奏|"
    r"第一人称|第三人称|全知视角|限制视角|"
    r"写作要求|文风|笔法|"
    r"POV|point\s+of\s+view|"
    r"tone|style|narrative|voice|pacing|"
    r"writing\s+rule|chapter\s+structure)",
    re.IGNORECASE,
)

# --- Noise filter (same as technical, applies to fiction too) ---
_FICTION_FILLER = re.compile(
    r"^(你好|嗯|好的|ok|okay|sure|yes|no|嗯嗯|哈哈|呵呵|"
    r"谢谢|感谢|不客气|thanks|thank\s+you|"
    r"hi|hello|hey|good\s+morning|good\s+night|"
    r"bye|see\s+you|再见|回见|晚安|"
    r"こんにちは|ありがとう|さようなら|"
    r"👍|🎉|❤️|👌|😂)$",
    re.IGNORECASE,
)


def fiction_importance_score(content: str) -> float:
    """Compute importance score for fiction content (0–1).

    Dimensions:
      character(0.25) + plot(0.20) + relationship(0.20) +
      world(0.15) + style(0.05) + length(0.05) + names(0.10)
    """
    stripped = content.strip()
    if len(stripped) < 3:
        return 0.0
    if _FICTION_FILLER.match(stripped):
        return 0.0

    scores: dict[str, float] = {}

    # Character signals
    char_hits = len(_CHARACTER_PATTERNS.findall(content))
    scores["character"] = min(char_hits / 2.0, 1.0)

    # Plot signals
    plot_hits = len(_PLOT_PATTERNS.findall(content))
    scores["plot"] = min(plot_hits / 2.0, 1.0)

    # Relationship signals
    rel_hits = len(_RELATIONSHIP_PATTERNS.findall(content))
    scores["relationship"] = min(rel_hits / 1.5, 1.0)

    # World-building signals
    world_hits = len(_WORLD_PATTERNS.findall(content))
    scores["world"] = min(world_hits / 2.0, 1.0)

    # Style signals
    style_hits = len(_STYLE_PATTERNS.findall(content))
    scores["style"] = min(style_hits / 1.0, 1.0)

    # Length signal
    if len(content) > 50:
        scores["length"] = 0.3
    elif len(content) > 20:
        scores["length"] = 0.15
    else:
        scores["length"] = 0.0

    # Named entity density
    # CJK names: 2-4 chars that look like names or faction/location names
    cjk_org = len(re.findall(r"[\u4e00-\u9fff]{2,4}(?:门|派|宗|帮|盟|国|城|山|谷|殿|阁|星|号)", content))
    cjk_people = len(re.findall(r"(?:叫|是|找|问|和|与|跟|给|被|把)\s*([\u4e00-\u9fff]{2,4})", content))
    en_names = len(re.findall(r"[A-Z][a-z]{2,}", content))
    scores["names"] = min((cjk_org + cjk_people + en_names) / 3.0, 1.0)

    weights = {
        "character": 0.25,
        "plot": 0.20,
        "relationship": 0.20,
        "world": 0.15,
        "style": 0.05,
        "length": 0.05,
        "names": 0.10,
    }

    total = sum(scores.get(k, 0) * weights[k] for k in weights)
    return min(max(total, 0.0), 1.0)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Fiction Entity & Relationship Types                         ║
# ╚══════════════════════════════════════════════════════════════╝

ENTITY_TYPES = {
    "character":  "A person or being in the story",
    "location":   "A place (city, mountain, realm, planet, building)",
    "faction":    "An organization (sect, kingdom, guild, family, corporation)",
    "artifact":   "An important object (weapon, treasure, document, ship)",
    "event":      "A significant plot event or incident",
    "concept":    "A world rule, magic system, technology, or abstract concept",
}

RELATION_TYPES = {
    "loves":        "Romantic feelings or affection",
    "hates":        "Enmity, hatred, or vendetta",
    "master_of":    "Teacher/master to student",
    "disciple_of":  "Student/apprentice to teacher",
    "parent_of":    "Parent to child",
    "sibling_of":   "Siblings",
    "ally_of":      "Alliance, friendship, partnership",
    "rival_of":     "Rivalry or competition",
    "member_of":    "Belongs to a faction/organization",
    "located_in":   "Physical location relationship",
    "possesses":    "Owns an artifact, ability, or resource",
    "killed":       "Killed (plot event)",
    "betrayed":     "Betrayal",
    "successor_of": "Succession or inheritance",
}

# Relation extraction patterns (Chinese + English)
RELATION_EXTRACT_PATTERNS = [
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:爱上了?|喜欢上了?|暗恋)\s*([\u4e00-\u9fff]{2,4})"), "loves"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:仇恨|恨|讨厌|看不起)\s*([\u4e00-\u9fff]{2,4})"), "hates"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:的师父是|拜)\s*([\u4e00-\u9fff]{2,4})"), "disciple_of"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:收|为徒|弟子是)\s*([\u4e00-\u9fff]{2,4})"), "master_of"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:是|属于|加入了?)\s*([\u4e00-\u9fff]{2,6}(?:门|派|宗|帮|盟|国|会))"), "member_of"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:位于|在|坐落于)\s*([\u4e00-\u9fff]{2,8})"), "located_in"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:杀了|杀死了?|击杀了?)\s*([\u4e00-\u9fff]{2,4})"), "killed"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:背叛了?|出卖了?)\s*([\u4e00-\u9fff]{2,4})"), "betrayed"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:和|与)\s*([\u4e00-\u9fff]{2,4})\s*(?:结盟|联手|合作|是盟友)"), "ally_of"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:和|与)\s*([\u4e00-\u9fff]{2,4})\s*(?:是对手|是宿敌|为敌)"), "rival_of"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:拥有|持有|得到了?)\s*([\u4e00-\u9fff]{2,6})"), "possesses"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:是)\s*([\u4e00-\u9fff]{2,4})\s*(?:的父亲|的母亲|的爸|的妈)"), "parent_of"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:和|与)\s*([\u4e00-\u9fff]{2,4})\s*(?:是兄弟|是姐妹|是兄妹)"), "sibling_of"),
    # English
    (re.compile(r"(\w+)\s+(?:loves?|fell\s+in\s+love\s+with)\s+(\w+)", re.I), "loves"),
    (re.compile(r"(\w+)\s+(?:hates?|despises?)\s+(\w+)", re.I), "hates"),
    (re.compile(r"(\w+)\s+(?:killed|murdered|slew)\s+(\w+)", re.I), "killed"),
    (re.compile(r"(\w+)\s+(?:betrayed|double-crossed)\s+(\w+)", re.I), "betrayed"),
    (re.compile(r"(\w+)\s+(?:is|was)\s+(\w+)'s\s+(?:master|teacher)", re.I), "master_of"),
    (re.compile(r"(\w+)\s+(?:is|was)\s+a\s+member\s+of\s+(?:the\s+)?(\w+)", re.I), "member_of"),
    (re.compile(r"(\w+)\s+(?:allied|teamed\s+up)\s+with\s+(\w+)", re.I), "ally_of"),
]
