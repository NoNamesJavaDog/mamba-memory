"""Selective Gate — the core of the Mamba3-inspired architecture.

Corresponds to Mamba3's Δ(x) selection mechanism: gate behavior is
input-dependent, not fixed. Each piece of incoming content is evaluated
for novelty, importance, and relevance to decide whether/how it enters
the state layer.
"""

from __future__ import annotations

import re

import numpy as np

from mamba_memory.core.types import (
    GateDecision,
    GateInput,
    L2Config,
    MemorySlot,
    WriteMode,
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# -- Importance signal detectors ---------------------------------------------
#
# Multi-dimensional weighted scoring with language-agnostic patterns.
# Each dimension returns a float (0–1), then dimensions are combined
# with learned-ish weights. Much richer than the original 5-bool average.

# --- Dimension 1: Intent signals (decisions, preferences, corrections) ---
# Covers: zh, en, ja, ko, and mixed

_DECISION_PATTERNS = re.compile(
    r"(决定|选择|确定|采用|改用|换成|用了|选用|启用|禁用|"       # zh
    r"決定|選択|採用|変更|"                                      # ja
    r"결정|선택|변경|"                                            # ko
    r"adopt|choose|decide|go\s+with|settle\s+on|pick|opt\s+for|"  # en
    r"switch\s+to|migrate\s+to|move\s+to|upgrade\s+to|downgrade)",
    re.IGNORECASE,
)
_PREFERENCE_PATTERNS = re.compile(
    r"(喜欢|偏好|不要|不喜欢|讨厌|倾向|习惯用|推荐|建议用|"       # zh
    r"好き|嫌い|おすすめ|"                                        # ja
    r"prefer|like\s+using|dislike|don't\s+want|avoid|hate|love\s+using|"
    r"recommend|suggest\s+using|rather\s+use|fan\s+of)",
    re.IGNORECASE,
)
_CORRECTION_PATTERNS = re.compile(
    r"(不对|错了|其实|改成|应该是|搞错|纠正|更正|实际上|"          # zh
    r"違う|間違|実は|修正|"                                       # ja
    r"actually|wrong|correct\s+that|instead|no[,.]|"
    r"wait[,.]|let\s+me\s+correct|I\s+was\s+wrong|scratch\s+that|"
    r"sorry[,.]?\s+I\s+meant|not\s+.{1,15}\s+but\s+)",
    re.IGNORECASE,
)

# --- Dimension 2: Structured data presence ---

# Note: avoid \b around CJK — it doesn't match CJK-to-ASCII boundaries.
_FACT_PATTERNS = re.compile(
    r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"                     # IP addresses
    r"|(?:port|端口)\s*[:=：是为]?\s*\d{2,5}"                     # Ports
    r"|[vV]ersion\s*[:=]?\s*[\d.]+"                               # Version strings
    r"|[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+"                         # Any URI scheme (http, postgres, redis, etc.)
    r"|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"         # Emails
    r"|(?:密码|password|token|api[_-]?key|secret)\s*[:=：]"      # Credentials
    r"|\.(?:ya?ml|json|toml|conf|env|ini|cfg)"                   # Config files
    r"|\d+\s*(?:MB|GB|TB|KB|ms|req|次|秒|分钟|小时|天)(?:/\w+)?" # Quantities (100 req/min, 256MB, etc.)
    r"|\d+\s*(?:am|pm|AM|PM)"                                    # Times (3am, 2pm)
    r"|(?:every|每)\s*(?:\d+\s*)?(?:day|hour|minute|week|month"  # Schedules
    r"|天|小时|分钟|周|月)"
    r"|(?:daily|weekly|monthly|hourly|每天|每周|每月|每小时)"      # Schedule keywords
    r"|(?:backup|备份|cron|schedule|定时|轮换|rotate)"             # Ops keywords
    r"|(?:limit|限流|throttle|quota|配额)\s*[:：]?\s*\d+"         # Rate limits
    r"|(?:sudo|npm|pip|docker|kubectl|git)\s+\w+)",              # Commands
    re.IGNORECASE,
)

# --- Dimension 3: Explicit memory request ---

_EXPLICIT_MEMORY_PATTERNS = re.compile(
    r"(记住|别忘|千万别忘|务必记|牢记|"                           # zh
    r"覚えて|忘れないで|"                                         # ja
    r"remember|don't\s+forget|keep\s+in\s+mind|note\s+that|"
    r"important:|make\s+sure\s+to|always\s+remember|never\s+forget|"
    r"write\s+this\s+down|for\s+future\s+reference)",
    re.IGNORECASE,
)

# --- Dimension 4: Action / task signals ---

_ACTION_PATTERNS = re.compile(
    r"(TODO|FIXME|HACK|XXX|WARN|"                                # code markers
    r"下一步|待办|接下来要|之后需要|别忘了做|计划是|"                # zh tasks
    r"next\s+step|action\s+item|follow\s+up|deadline|by\s+\w+day|"
    r"need\s+to|have\s+to|must|should|make\s+sure|don't\s+forget\s+to)",
    re.IGNORECASE,
)

# --- Dimension 5: Noise / filler detection (negative signal) ---

_FILLER_PATTERNS = re.compile(
    r"^(你好|嗯|好的|ok|okay|sure|yes|no|嗯嗯|哈哈|呵呵|"
    r"谢谢|感谢|不客气|thanks|thank\s+you|"
    r"hi|hello|hey|good\s+morning|good\s+night|"
    r"こんにちは|ありがとう|はい|いいえ|"
    r"bye|see\s+you|再见|回见|晚安)$",
    re.IGNORECASE,
)

_GREETING_LIKE = re.compile(
    r"^.{0,12}(你好|hi|hello|hey|嗨|早|good\s).{0,12}$",
    re.IGNORECASE,
)


def _importance_score(content: str) -> float:
    """Compute a multi-dimensional importance score (0–1).

    Dimensions and weights:
      - Intent signals  (decision/preference/correction): 0.25
      - Structured data (IPs, ports, URLs, configs):      0.25
      - Explicit memory request:                          0.20
      - Action/task signals:                              0.15
      - Information density:                              0.10
      - Noise penalty (filler/greeting):                  -0.15

    Returns a float in [0, 1].
    """
    # Short-circuit: pure filler/greeting → 0
    stripped = content.strip()
    if len(stripped) < 3:
        return 0.0
    if _FILLER_PATTERNS.match(stripped) or _GREETING_LIKE.match(stripped):
        return 0.0

    scores: dict[str, float] = {}

    # Intent signals — graded by how many types match (not just binary)
    intent_hits = sum([
        bool(_DECISION_PATTERNS.search(content)),
        bool(_PREFERENCE_PATTERNS.search(content)),
        bool(_CORRECTION_PATTERNS.search(content)),
    ])
    scores["intent"] = min(intent_hits / 1.5, 1.0)  # 1 type → 0.67, 2+ → 1.0

    # Structured data — count distinct matches
    fact_matches = _FACT_PATTERNS.findall(content)
    scores["structured"] = min(len(fact_matches) / 2.0, 1.0)

    # Explicit memory request
    scores["explicit"] = 1.0 if _EXPLICIT_MEMORY_PATTERNS.search(content) else 0.0

    # Action/task signals
    scores["action"] = 1.0 if _ACTION_PATTERNS.search(content) else 0.0

    # Information density (from text module)
    from mamba_memory.core.text import information_density
    scores["density"] = information_density(content)

    # Weighted combination
    weights = {
        "intent": 0.25,
        "structured": 0.25,
        "explicit": 0.20,
        "action": 0.15,
        "density": 0.15,
    }

    total = sum(scores[k] * weights[k] for k in weights)
    return min(max(total, 0.0), 1.0)


class Gate:
    """Selective gate that decides how L1 overflow enters L2.

    Three evaluation paths (in priority order):
    1. Learned classifier (if trained) — feature-based logistic regression
    2. Rule engine (default, <10ms) — embeddings + heuristics
    3. Hybrid: learned gate sets importance, rule engine handles slot allocation

    The learned gate doesn't replace the rule engine entirely — it replaces
    the importance scoring. Slot allocation, novelty detection, and write mode
    decisions still use the rule-based logic.
    """

    def __init__(self, config: L2Config | None = None) -> None:
        self.config = config or L2Config()
        self._learned_gate = None

    def set_learned_gate(self, learned_gate) -> None:
        """Attach a trained LearnedGate for hybrid evaluation."""
        self._learned_gate = learned_gate

    def evaluate(self, inp: GateInput, slots: list[MemorySlot]) -> GateDecision:
        """Evaluate whether and how to write *inp* into the state layer."""

        # Always compute rule-based importance
        rule_importance = _importance_score(inp.content)

        if self._learned_gate is not None and self._learned_gate.trained:
            # Ensemble: weighted combination of rule engine + neural gate
            neural_conf, neural_store = self._learned_gate.predict(inp.content)
            neural_importance = neural_conf if neural_store else neural_conf * 0.3

            # Agreement check → confidence boost / disagreement → cautious
            rule_store = rule_importance >= 0.20
            if rule_store == neural_store:
                # Both agree → use neural (it's more accurate) with confidence boost
                importance = neural_importance
            elif neural_conf > 0.9:
                # Neural is very confident, rule disagrees → trust neural
                importance = neural_importance
            elif neural_conf < 0.3 and rule_importance > 0.3:
                # Neural says discard (confident), rule says store → trust neural
                # (neural catches deferral/negation that rules miss)
                importance = neural_importance
            else:
                # Ambiguous disagreement → average both (cautious)
                importance = 0.5 * rule_importance + 0.5 * neural_importance
        else:
            importance = rule_importance

        # Compute similarity against all non-empty slots that have embeddings
        similarities: list[tuple[int, float]] = []
        if inp.embedding is not None:
            for slot in slots:
                if slot.is_empty or slot.embedding is None:
                    continue
                sim = cosine_similarity(inp.embedding, slot.embedding)
                similarities.append((slot.id, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)

        top_match = similarities[0] if similarities else None
        is_novel = top_match is None or top_match[1] < 0.7

        # --- Decision logic ---

        # Explicit memory request → always store
        if bool(_EXPLICIT_MEMORY_PATTERNS.search(inp.content)):
            return self._decide_store(
                inp, slots, similarities, importance,
                force_new=is_novel,
                reason="explicit memory request",
            )

        # Low importance + not novel → discard
        if importance < 0.2 and not is_novel:
            return GateDecision(
                should_write=False,
                reason=f"low importance ({importance:.2f}), not novel",
            )

        # Novel + important → create new slot
        if is_novel and importance >= 0.25:
            return self._decide_store(
                inp, slots, similarities, importance,
                force_new=True,
                reason=f"novel topic, importance={importance:.2f}",
            )

        # High similarity to existing slot → update
        if top_match and top_match[1] >= 0.7:
            co_acts = [sid for sid, _ in similarities[1:4]]
            write_mode = WriteMode.MERGE if top_match[1] > 0.9 else WriteMode.UPDATE

            # Correction signal → overwrite instead
            if bool(_CORRECTION_PATTERNS.search(inp.content)):
                write_mode = WriteMode.OVERWRITE

            return GateDecision(
                should_write=True,
                target_slot=top_match[0],
                write_mode=write_mode,
                gate_strength=max(importance, 0.3),
                co_activations=co_acts,
                reason=f"updating slot {top_match[0]} (sim={top_match[1]:.2f})",
            )

        # Novel but low importance → still store if moderately important
        if is_novel and importance >= 0.20:
            return self._decide_store(
                inp, slots, similarities, importance,
                force_new=True,
                reason=f"novel, moderate importance={importance:.2f}",
            )

        # Ambiguous → discard (conservative default)
        top_sim = top_match[1] if top_match else 0.0
        return GateDecision(
            should_write=False,
            reason=f"ambiguous: importance={importance:.2f}, top_sim={top_sim:.2f}",
        )

    def _decide_store(
        self,
        inp: GateInput,
        slots: list[MemorySlot],
        similarities: list[tuple[int, float]],
        importance: float,
        *,
        force_new: bool,
        reason: str,
    ) -> GateDecision:
        """Build a store decision, finding or allocating a target slot."""
        co_acts = [sid for sid, _ in similarities[:3]]

        if not force_new and similarities:
            target = similarities[0][0]
        else:
            target = self._find_free_slot(slots)

        # Saturation: all slots are important, reject low-value writes
        if target == -1:
            return GateDecision(
                should_write=False,
                reason=f"system saturated: all slots have high activation, "
                f"rejecting content with importance={importance:.2f}",
            )

        return GateDecision(
            should_write=True,
            target_slot=target,
            write_mode=WriteMode.OVERWRITE if force_new else WriteMode.UPDATE,
            gate_strength=max(importance, 0.4),
            co_activations=co_acts,
            reason=reason,
        )

    def _find_free_slot(self, slots: list[MemorySlot]) -> int:
        """Find an empty or lowest-activation slot to allocate.

        Returns -1 if all slots are saturated (all activations above 0.6),
        signaling that the gate should reject this write.
        """
        # Prefer empty slots
        for slot in slots:
            if slot.is_empty:
                return slot.id

        # Find the weakest slot
        weakest = min(slots, key=lambda s: s.activation)

        # Saturation guard: if even the weakest slot has high activation,
        # signal that the system is saturated
        saturation_threshold = 0.6
        if weakest.activation > saturation_threshold:
            return -1  # all slots are important, reject the write

        return weakest.id
