"""Legal preset — adapts MambaMemory for law/litigation scenarios.

Core principle: deadlines are critical, statutes and precedents must be precise.

Usage:
    from mamba_memory.presets.legal import create_legal_engine
    engine = create_legal_engine()
    await engine.start()
    await engine.ingest("根据民法典第1165条，侵权人应当承担赔偿责任", tags=["民法典"])
"""

from __future__ import annotations
import re
from typing import Any
from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config


def create_legal_config(db_path: str = "~/.mamba-memory/legal.db", **overrides: Any) -> EngineConfig:
    return EngineConfig(
        l1=L1Config(window_size=8, max_compressed_segments=25),
        l2=L2Config(
            slot_count=128, slot_max_tokens=500, base_decay_rate=0.995,
            eviction_threshold=0.02, snapshot_interval=15,
            time_decay_enabled=True, time_decay_halflife=259200.0, activation_boost=1.3,
        ),
        l3=L3Config(db_path=db_path),
        embedding_provider=overrides.get("embedding_provider", "auto"),
        compression_model=overrides.get("compression_model", "none"),
        namespace=overrides.get("namespace", "legal"),
    )

def create_legal_engine(db_path: str = "~/.mamba-memory/legal.db", **kwargs):
    from mamba_memory.core.engine import MambaMemoryEngine
    return MambaMemoryEngine(create_legal_config(db_path, **kwargs))


_STATUTE = re.compile(
    r"(法条|法律|法规|条款|第.{1,6}条|"
    r"民法典|刑法|行政法|合同法|劳动法|"
    r"Article|Section|statute|regulation|act|"
    r"USC|CFR|amendment|provision|"
    r"司法解释|最高法|最高检)", re.I)

_CASE_LAW = re.compile(
    r"(判例|判决|裁定|裁决|案号|"
    r"[\(（]\d{4}[\)）].*号|"
    r"ruling|precedent|verdict|judgment|"
    r"case\s+no|docket|opinion|"
    r"上诉|一审|二审|再审|终审)", re.I)

_CONTRACT = re.compile(
    r"(合同|协议|条款|契约|"
    r"甲方|乙方|丙方|签约|违约|"
    r"contract|agreement|clause|term|"
    r"party|breach|termination|"
    r"有效期|生效|解除|附件|"
    r"NDA|SLA|MOU|LOI)", re.I)

_DEADLINE = re.compile(
    r"(时效|期限|截止|到期|"
    r"deadline|due\s+date|filing\s+date|"
    r"statute\s+of\s+limitations|"
    r"答辩期|举证期限|上诉期|"
    r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}|"
    r"within\s+\d+\s+days)", re.I)

_EVIDENCE = re.compile(
    r"(证据|证人|证物|物证|书证|"
    r"鉴定|笔录|录音|录像|"
    r"evidence|witness|exhibit|testimony|"
    r"deposition|affidavit|subpoena|"
    r"举证|质证|采信)", re.I)

_PARTIES = re.compile(
    r"(原告|被告|第三人|当事人|"
    r"代理人|律师|法官|"
    r"plaintiff|defendant|appellant|respondent|"
    r"counsel|attorney|judge|"
    r"申请人|被申请人|上诉人|被上诉人)", re.I)

_FILLER = re.compile(
    r"^(你好|嗯|好的|ok|okay|sure|thanks|谢谢|bye|再见|"
    r"hi|hello|嗯嗯|哈哈|呵呵|👍|🎉)$", re.I)


def legal_importance_score(content: str) -> float:
    s = content.strip()
    if len(s) < 3 or _FILLER.match(s):
        return 0.0
    scores = {
        "statute": min(len(_STATUTE.findall(content)) / 2.0, 1.0),
        "case_law": min(len(_CASE_LAW.findall(content)) / 2.0, 1.0),
        "contract": min(len(_CONTRACT.findall(content)) / 2.0, 1.0),
        "deadline": min(len(_DEADLINE.findall(content)) / 1.5, 1.0),
        "evidence": min(len(_EVIDENCE.findall(content)) / 2.0, 1.0),
        "parties": min(len(_PARTIES.findall(content)) / 2.0, 1.0),
    }
    weights = {"statute": 0.20, "case_law": 0.20, "contract": 0.20,
               "deadline": 0.15, "evidence": 0.15, "parties": 0.10}
    return min(max(sum(scores[k] * weights[k] for k in weights), 0.0), 1.0)


ENTITY_TYPES = {
    "case": "A legal case or lawsuit", "statute": "A law, regulation, or legal provision",
    "contract": "A contract or agreement", "party": "A legal party (plaintiff, defendant)",
    "evidence": "Evidence or exhibit", "court": "A court or tribunal",
}

RELATION_TYPES = {
    "represents": "Attorney represents party", "cites": "Document cites statute/precedent",
    "contradicts": "Evidence contradicts claim", "filed_against": "Case filed against party",
    "ruled_in_favor": "Court ruled in favor of party", "bound_by": "Party bound by contract",
    "amends": "New law amends old law", "supersedes": "New ruling supersedes old one",
}

RELATION_EXTRACT_PATTERNS = [
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:代理|代表)\s*([\u4e00-\u9fff]{2,4})"), "represents"),
    (re.compile(r"(?:根据|依据|引用)\s*([\u4e00-\u9fff]{2,8}第.{1,6}条)"), "cites"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:起诉|控告)\s*([\u4e00-\u9fff]{2,4})"), "filed_against"),
    (re.compile(r"(\w+)\s+(?:represents?|counsel\s+for)\s+(\w+)", re.I), "represents"),
    (re.compile(r"(?:citing|pursuant\s+to|under)\s+(.{3,30})", re.I), "cites"),
    (re.compile(r"(\w+)\s+(?:filed\s+against|sued)\s+(\w+)", re.I), "filed_against"),
    (re.compile(r"(?:ruled|decided)\s+(?:in\s+favor\s+of|for)\s+(\w+)", re.I), "ruled_in_favor"),
]
