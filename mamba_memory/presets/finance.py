"""Finance preset — adapts MambaMemory for accounting/financial scenarios.

Core principle: numbers must be precise, compliance records must persist.

Usage:
    from mamba_memory.presets.finance import create_finance_engine
    engine = create_finance_engine()
    await engine.start()
    await engine.ingest("Q3营收2.3亿，同比增长15%", tags=["营收"])
"""

from __future__ import annotations
import re
from typing import Any
from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config


def create_finance_config(db_path: str = "~/.mamba-memory/finance.db", **overrides: Any) -> EngineConfig:
    return EngineConfig(
        l1=L1Config(window_size=8, max_compressed_segments=20),
        l2=L2Config(
            slot_count=128, slot_max_tokens=400, base_decay_rate=0.99,
            eviction_threshold=0.03, snapshot_interval=15,
            time_decay_enabled=True, time_decay_halflife=86400.0, activation_boost=1.3,
        ),
        l3=L3Config(db_path=db_path),
        embedding_provider=overrides.get("embedding_provider", "auto"),
        compression_model=overrides.get("compression_model", "none"),
        namespace=overrides.get("namespace", "finance"),
    )

def create_finance_engine(db_path: str = "~/.mamba-memory/finance.db", **kwargs):
    from mamba_memory.core.engine import MambaMemoryEngine
    return MambaMemoryEngine(create_finance_config(db_path, **kwargs))


_AMOUNTS = re.compile(
    r"(金额|万元|百万|千万|亿|营收|利润|亏损|成本|预算|"
    r"[\$€¥£]\s*[\d,.]+|[\d,.]+\s*(?:万|亿|元|美元|dollar)|"
    r"\d+\.?\d*\s*%|"
    r"revenue|profit|loss|budget|cost|margin|EBITDA|"
    r"同比|环比|增长|下降|YoY|QoQ|MoM)", re.I)

_ACCOUNTS = re.compile(
    r"(账户|账号|科目|余额|借方|贷方|"
    r"account|ledger|balance|debit|credit|"
    r"应收|应付|现金流|资产|负债|权益|"
    r"receivable|payable|cash\s+flow|asset|liability|equity)", re.I)

_COMPLIANCE = re.compile(
    r"(合规|审计|法规|税率|税务|申报|"
    r"regulation|audit|compliance|tax|filing|"
    r"GAAP|IFRS|SOX|反洗钱|AML|KYC|"
    r"监管|处罚|罚款|penalty|fine)", re.I)

_TRANSACTIONS = re.compile(
    r"(交易|转账|支付|收入|发票|收据|汇款|"
    r"transaction|payment|invoice|receipt|transfer|"
    r"结算|清算|settlement|clearing)", re.I)

_REPORTING = re.compile(
    r"(报表|财报|季报|年报|月报|"
    r"report|quarterly|annual|P&L|"
    r"balance\s+sheet|income\s+statement|cash\s+flow\s+statement|"
    r"资产负债表|利润表|现金流量表)", re.I)

_RISK = re.compile(
    r"(风险|风控|敞口|对冲|波动|"
    r"risk|exposure|hedge|volatility|VaR|"
    r"信用风险|市场风险|操作风险|流动性|"
    r"credit\s+risk|market\s+risk|liquidity)", re.I)

_FILLER = re.compile(
    r"^(你好|嗯|好的|ok|okay|sure|thanks|谢谢|bye|再见|"
    r"hi|hello|嗯嗯|哈哈|呵呵|👍|🎉)$", re.I)


def finance_importance_score(content: str) -> float:
    s = content.strip()
    if len(s) < 3 or _FILLER.match(s):
        return 0.0
    scores = {
        "amounts": min(len(_AMOUNTS.findall(content)) / 2.0, 1.0),
        "accounts": min(len(_ACCOUNTS.findall(content)) / 2.0, 1.0),
        "compliance": min(len(_COMPLIANCE.findall(content)) / 1.5, 1.0),
        "transactions": min(len(_TRANSACTIONS.findall(content)) / 2.0, 1.0),
        "reporting": min(len(_REPORTING.findall(content)) / 2.0, 1.0),
        "risk": min(len(_RISK.findall(content)) / 2.0, 1.0),
    }
    weights = {"amounts": 0.25, "accounts": 0.20, "compliance": 0.20,
               "transactions": 0.15, "reporting": 0.10, "risk": 0.10}
    return min(max(sum(scores[k] * weights[k] for k in weights), 0.0), 1.0)


ENTITY_TYPES = {
    "account": "A financial account or ledger",
    "transaction": "A financial transaction",
    "regulation": "A regulatory requirement or standard",
    "report": "A financial report or statement",
    "entity_org": "A company, fund, or financial entity",
    "product": "A financial product or instrument",
}

RELATION_TYPES = {
    "owns": "Entity owns account/asset", "audits": "Auditor audits entity",
    "complies_with": "Entity complies with regulation",
    "transfers_to": "Account transfers to another account",
    "reports_to": "Entity reports to regulator", "invoiced_by": "Entity invoiced by vendor",
}

RELATION_EXTRACT_PATTERNS = [
    (re.compile(r"([\u4e00-\u9fff]{2,8})\s*(?:转账|汇款|支付)\s*(?:给|到)\s*([\u4e00-\u9fff]{2,8})"), "transfers_to"),
    (re.compile(r"([\u4e00-\u9fff]{2,8})\s*(?:审计|审查)\s*([\u4e00-\u9fff]{2,8})"), "audits"),
    (re.compile(r"([\u4e00-\u9fff]{2,8})\s*(?:遵守|符合)\s*([\u4e00-\u9fff]{2,8})"), "complies_with"),
    (re.compile(r"(\w+)\s+(?:transferred?\s+to|paid)\s+(\w+)", re.I), "transfers_to"),
    (re.compile(r"(\w+)\s+(?:audited|reviewed)\s+(\w+)", re.I), "audits"),
    (re.compile(r"(\w+)\s+(?:complies?\s+with|subject\s+to)\s+(\w+)", re.I), "complies_with"),
]
