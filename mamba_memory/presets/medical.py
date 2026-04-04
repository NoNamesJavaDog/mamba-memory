"""Medical preset — adapts MambaMemory for clinical/healthcare scenarios.

Core principle: NEVER forget critical info (allergies, chronic conditions).

Usage:
    from mamba_memory.presets.medical import create_medical_engine
    engine = create_medical_engine()
    await engine.start()
    await engine.ingest("患者对青霉素严重过敏", tags=["过敏"])
"""

from __future__ import annotations
import re
from typing import Any
from mamba_memory.core.types import EngineConfig, L1Config, L2Config, L3Config


def create_medical_config(db_path: str = "~/.mamba-memory/medical.db", **overrides: Any) -> EngineConfig:
    return EngineConfig(
        l1=L1Config(window_size=10, max_compressed_segments=30),
        l2=L2Config(
            slot_count=256, slot_max_tokens=400, base_decay_rate=0.999,
            eviction_threshold=0.01, snapshot_interval=10,
            time_decay_enabled=True, time_decay_halflife=604800.0, activation_boost=1.2,
        ),
        l3=L3Config(db_path=db_path),
        embedding_provider=overrides.get("embedding_provider", "auto"),
        compression_model=overrides.get("compression_model", "none"),
        namespace=overrides.get("namespace", "medical"),
    )

def create_medical_engine(db_path: str = "~/.mamba-memory/medical.db", **kwargs):
    from mamba_memory.core.engine import MambaMemoryEngine
    return MambaMemoryEngine(create_medical_config(db_path, **kwargs))


_DIAGNOSIS = re.compile(
    r"(诊断|确诊|疑似|病理|症状|病变|阳性|阴性|"
    r"diagnosed|confirmed|suspected|pathology|symptom|positive|negative|"
    r"糖尿病|高血压|冠心病|肿瘤|癌|炎症|感染|骨折|"
    r"diabetes|hypertension|cancer|tumor|infection|fracture|pneumonia|"
    r"综合征|syndrome|disease|disorder|condition)", re.I)

_MEDICATION = re.compile(
    r"(药物|处方|剂量|用药|口服|注射|静脉|肌注|"
    r"mg|ml|μg|片|粒|支|瓶|袋|"
    r"prescribed|dosage|tablet|capsule|injection|IV|"
    r"抗生素|胰岛素|降压药|止痛药|"
    r"antibiotic|insulin|aspirin|metformin|amoxicillin|"
    r"每日|每次|bid|tid|qd|prn)", re.I)

_ALLERGY = re.compile(
    r"(过敏|禁忌|不良反应|副作用|"
    r"allergic|allergy|contraindication|contraindicated|adverse|"
    r"anaphylaxis|rash|swelling|禁用|慎用)", re.I)

_VITALS = re.compile(
    r"(血压|心率|体温|血糖|血氧|呼吸|脉搏|"
    r"BP|HR|SpO2|mmHg|bpm|℃|°C|"
    r"白细胞|血红蛋白|血小板|肌酐|转氨酶|"
    r"WBC|RBC|HGB|PLT|ALT|AST|BUN|eGFR|"
    r"lab\s+result|vital\s+sign|检验|化验)", re.I)

_PROCEDURE = re.compile(
    r"(手术|切除|移植|穿刺|活检|引流|"
    r"CT|MRI|X光|B超|心电图|胃镜|肠镜|"
    r"surgery|biopsy|transplant|endoscopy|"
    r"ECG|EKG|ultrasound|radiograph)", re.I)

_HISTORY = re.compile(
    r"(病史|既往|家族史|遗传|慢性|"
    r"history|chronic|previous|familial|hereditary|"
    r"复发|长期|多年)", re.I)

_FILLER = re.compile(
    r"^(你好|嗯|好的|ok|okay|sure|thanks|谢谢|bye|再见|"
    r"hi|hello|嗯嗯|哈哈|呵呵|👍|🎉)$", re.I)


def medical_importance_score(content: str) -> float:
    s = content.strip()
    if len(s) < 3 or _FILLER.match(s):
        return 0.0
    scores = {
        "diagnosis": min(len(_DIAGNOSIS.findall(content)) / 2.0, 1.0),
        "medication": min(len(_MEDICATION.findall(content)) / 2.0, 1.0),
        "allergy": min(len(_ALLERGY.findall(content)) / 1.5, 1.0),
        "vitals": min(len(_VITALS.findall(content)) / 2.0, 1.0),
        "procedure": min(len(_PROCEDURE.findall(content)) / 2.0, 1.0),
        "history": min(len(_HISTORY.findall(content)) / 2.0, 1.0),
    }
    weights = {"diagnosis": 0.25, "medication": 0.20, "allergy": 0.20,
               "vitals": 0.15, "procedure": 0.10, "history": 0.10}
    return min(max(sum(scores[k] * weights[k] for k in weights), 0.0), 1.0)


ENTITY_TYPES = {
    "patient": "A patient", "symptom": "A symptom or complaint",
    "medication": "A drug or treatment", "diagnosis": "A clinical diagnosis",
    "procedure": "A medical procedure or test", "body_part": "An anatomical location",
}

RELATION_TYPES = {
    "diagnosed_with": "Patient has diagnosis", "allergic_to": "Patient is allergic to",
    "prescribed": "Patient is prescribed medication", "referred_to": "Referred to specialist",
    "contraindicated_with": "Drug contraindicated with condition",
    "caused_by": "Symptom caused by condition", "treated_by": "Condition treated by procedure",
    "has_symptom": "Patient presents symptom",
}

RELATION_EXTRACT_PATTERNS = [
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:确诊|诊断为|患有)\s*([\u4e00-\u9fff]{2,8})"), "diagnosed_with"),
    (re.compile(r"([\u4e00-\u9fff]{2,4})\s*(?:对|过敏)\s*([\u4e00-\u9fff]{2,6})\s*(?:过敏)"), "allergic_to"),
    (re.compile(r"(?:给|为)\s*([\u4e00-\u9fff]{2,4})\s*(?:开|处方)\s*([\u4e00-\u9fff]{2,6})"), "prescribed"),
    (re.compile(r"(\w+)\s+(?:diagnosed\s+with|has)\s+(\w[\w\s]{2,20})", re.I), "diagnosed_with"),
    (re.compile(r"(\w+)\s+(?:allergic\s+to)\s+(\w+)", re.I), "allergic_to"),
    (re.compile(r"(?:prescribed|started)\s+(\w+)\s+(?:for|to)\s+(\w+)", re.I), "prescribed"),
]
