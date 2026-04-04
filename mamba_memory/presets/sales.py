"""Sales preset — adapts MambaMemory for CRM/sales pipeline scenarios.

Core principle: stale leads lose value FAST. Aggressive decay is intentional.

Usage:
    from mamba_memory.presets.sales import create_sales_engine
    engine = create_sales_engine()
    await engine.start()
    await engine.ingest("客户张总对企业版报价50万，下周二前给答复", tags=["张总"])
"""

from __future__ import annotations
import re
from typing import Any
from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config


def create_sales_config(db_path: str = "~/.mamba-memory/sales.db", **overrides: Any) -> EngineConfig:
    return EngineConfig(
        l1=L1Config(window_size=6, max_compressed_segments=15),
        l2=L2Config(
            slot_count=64, slot_max_tokens=300, base_decay_rate=0.95,
            eviction_threshold=0.08, snapshot_interval=30,
            time_decay_enabled=True, time_decay_halflife=14400.0, activation_boost=1.5,
        ),
        l3=L3Config(db_path=db_path),
        embedding_provider=overrides.get("embedding_provider", "auto"),
        compression_model=overrides.get("compression_model", "none"),
        namespace=overrides.get("namespace", "sales"),
    )

def create_sales_engine(db_path: str = "~/.mamba-memory/sales.db", **kwargs):
    from mamba_memory.core.engine import MambaMemoryEngine
    return MambaMemoryEngine(create_sales_config(db_path, **kwargs))


_CUSTOMER = re.compile(
    r"(客户|公司|联系人|负责人|决策人|"
    r"customer|client|contact|account|lead|prospect|"
    r"总|经理|主管|director|VP|CEO|CTO|"
    r"采购|需求方|甲方)", re.I)

_DEAL = re.compile(
    r"(报价|合同|订单|成交|签约|"
    r"金额|万|万元|百万|"
    r"[\$€¥]\s*[\d,.]+|"
    r"quote|deal|order|close|contract|"
    r"revenue|ARR|MRR|ACV|TCV|"
    r"折扣|优惠|discount)", re.I)

_PIPELINE = re.compile(
    r"(跟进|阶段|漏斗|进展|状态|"
    r"pipeline|stage|funnel|"
    r"demo|演示|提案|proposal|negotiation|"
    r"POC|试用|试点|pilot|"
    r"初次接触|需求确认|方案|谈判|签约)", re.I)

_COMPETITOR = re.compile(
    r"(竞品|竞争对手|对比|替代|"
    r"competitor|vs|versus|alternative|"
    r"switch\s+from|migrate\s+from|replace|"
    r"优势|劣势|差异|比较)", re.I)

_ACTION = re.compile(
    r"(下一步|待办|回访|约见|"
    r"next\s+step|follow\s+up|call\s+back|schedule|"
    r"meeting|会议|拜访|电话|邮件|"
    r"deadline|截止|之前|before|by\s+\w+day|"
    r"明天|下周|本周|today|tomorrow|this\s+week)", re.I)

_FILLER = re.compile(
    r"^(你好|嗯|好的|ok|okay|sure|thanks|谢谢|bye|再见|"
    r"hi|hello|嗯嗯|哈哈|呵呵|👍|🎉)$", re.I)


def sales_importance_score(content: str) -> float:
    s = content.strip()
    if len(s) < 3 or _FILLER.match(s):
        return 0.0
    scores = {
        "customer": min(len(_CUSTOMER.findall(content)) / 2.0, 1.0),
        "deal": min(len(_DEAL.findall(content)) / 2.0, 1.0),
        "pipeline": min(len(_PIPELINE.findall(content)) / 2.0, 1.0),
        "competitor": min(len(_COMPETITOR.findall(content)) / 1.5, 1.0),
        "action": min(len(_ACTION.findall(content)) / 2.0, 1.0),
    }
    weights = {"customer": 0.25, "deal": 0.25, "pipeline": 0.20,
               "competitor": 0.15, "action": 0.15}
    return min(max(sum(scores[k] * weights[k] for k in weights), 0.0), 1.0)


ENTITY_TYPES = {
    "customer": "A customer or prospect company",
    "deal": "A sales deal or opportunity",
    "product": "A product or service being sold",
    "competitor": "A competing product or company",
    "contact": "A person at the customer organization",
}

RELATION_TYPES = {
    "contacts": "Salesperson contacts customer",
    "quotes": "Deal includes a quote for product",
    "competes_with": "Product competes with competitor",
    "referred_by": "Customer referred by another",
    "upsold_to": "Customer upsold to higher tier",
    "churned_from": "Customer churned from product",
}

RELATION_EXTRACT_PATTERNS = [
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:联系了?|拜访了?|约了)\s*([\u4e00-\u9fff]{2,6})"), "contacts"),
    (re.compile(r"(?:给|向)\s*([\u4e00-\u9fff]{2,6})\s*(?:报价|报了)"), "quotes"),
    (re.compile(r"(\w+)\s*(?:竞品|竞争|对比|vs)\s*(\w+)"), "competes_with"),
    (re.compile(r"(\w+)\s+(?:contacted|visited|called)\s+(\w+)", re.I), "contacts"),
    (re.compile(r"(\w+)\s+(?:quoted|proposed)\s+(?:to\s+)?(\w+)", re.I), "quotes"),
    (re.compile(r"(\w+)\s+(?:competes?\s+with|vs)\s+(\w+)", re.I), "competes_with"),
]
