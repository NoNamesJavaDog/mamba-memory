"""Learned Gate v2 — self-evolving neural classifier with semantic + context awareness.

Architecture:
  Input features (37-dim) → Hidden layer (16 neurons, ReLU) → Output (sigmoid)

Four types of features:
  1. Rule signals     (15-dim) — same regex features as v1
  2. Semantic embed   (8-dim)  — compressed embedding captures meaning
  3. Context window   (8-dim)  — what's being discussed right now
  4. User profile     (6-dim)  — what topics this user cares about

Three learning modes:
  1. Batch training       — cold start on labeled data
  2. Online learning      — real-time single-sample SGD from corrections
  3. Implicit feedback    — recalled=positive, evicted-without-recall=negative

Why two-layer NN instead of logistic regression:
  - Can learn feature interactions (e.g., "short + has number" is a config value)
  - ReLU activation captures non-linear decision boundaries
  - Still tiny: 72×16 + 16×1 = 1168 parameters, trains in <1ms per sample
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

from mamba_memory.core.l2.gate import (
    _ACTION_PATTERNS,
    _CORRECTION_PATTERNS,
    _DECISION_PATTERNS,
    _EXPLICIT_MEMORY_PATTERNS,
    _FACT_PATTERNS,
    _FILLER_PATTERNS,
    _PREFERENCE_PATTERNS,
    _importance_score,
)
from mamba_memory.core.text import _STOP_WORDS, _is_cjk, information_density, tokenize

# ---------------------------------------------------------------------------
# Feature extraction (37 dimensions total)
# ---------------------------------------------------------------------------

N_RULE_FEATURES = 15
N_NEGATION_FEATURES = 4
N_QUESTION_FEATURES = 3  # NEW: question detection
N_OFFTOPIC_FEATURES = 3  # NEW: off-topic / non-technical
N_COMMAND_FEATURES = 5
N_CODE_FEATURES = 4
N_SEMANTIC_FEATURES = 24
N_CONTEXT_FEATURES = 8
N_PROFILE_FEATURES = 6
N_FEATURES = (N_RULE_FEATURES + N_NEGATION_FEATURES + N_QUESTION_FEATURES
              + N_OFFTOPIC_FEATURES + N_COMMAND_FEATURES + N_CODE_FEATURES
              + N_SEMANTIC_FEATURES + N_CONTEXT_FEATURES + N_PROFILE_FEATURES)
# = 15 + 4 + 3 + 3 + 5 + 4 + 24 + 8 + 6 = 72


def extract_rule_features(content: str) -> np.ndarray:
    """Original 15 rule-based features."""
    features = np.zeros(N_RULE_FEATURES, dtype=np.float32)
    if not content.strip():
        return features

    features[0] = 1.0 if _DECISION_PATTERNS.search(content) else 0.0
    features[1] = 1.0 if _PREFERENCE_PATTERNS.search(content) else 0.0
    features[2] = 1.0 if _CORRECTION_PATTERNS.search(content) else 0.0
    features[3] = 1.0 if _FACT_PATTERNS.search(content) else 0.0
    features[4] = 1.0 if _EXPLICIT_MEMORY_PATTERNS.search(content) else 0.0
    features[5] = information_density(content)
    features[6] = math.log1p(len(content)) / 10.0
    cjk_count = sum(1 for c in content if _is_cjk(c))
    features[7] = cjk_count / max(len(content), 1)
    features[8] = min(len(re.findall(r"\d+", content)) / 5.0, 1.0)
    features[9] = min(len(re.findall(r"[A-Z][a-zA-Z]+", content)) / 5.0, 1.0)
    punct_count = sum(1 for c in content if not c.isalnum() and not c.isspace())
    features[10] = punct_count / max(len(content), 1)
    words = re.findall(r"\w+", content.lower())
    if words:
        stop_count = sum(1 for w in words if w in _STOP_WORDS)
        features[11] = stop_count / len(words)
    tokens = tokenize(content)
    if tokens:
        features[12] = len(set(tokens)) / len(tokens)
    features[13] = 1.0 if _ACTION_PATTERNS.search(content) else 0.0
    features[14] = 1.0 if _FILLER_PATTERNS.match(content.strip()) else 0.0

    return features


# -- Negation / deferral detection
# Catches: "到时候再决定", "以后再说", "let me think about it", "maybe later"
# These contain action/decision words but the actual intent is to DEFER, not act.

_DEFERRAL_PATTERNS = re.compile(
    r"(到时候|以后|之后|下次|改天|等等|待|回头|later|maybe|perhaps|"
    r"not\s+sure\s+yet|let\s+me\s+think|think\s+about\s+it|"
    r"haven't\s+decided|not\s+decided|undecided|pending|TBD|"
    r"再说|再看|再想|还没|尚未|暂时不|先不|不急|"
    r"someday|eventually|not\s+now|overkill\s+for\s+now|"
    r"revisit|look\s+into\s+.{0,10}later|next\s+quarter|"
    r"看看要不要|考虑.*以后|等.*再|いつか|かもしれません)",
    re.IGNORECASE,
)

_NEGATION_PATTERNS = re.compile(
    r"(不是|并非|没有|未|别|不要|不用|无需|不必|"
    r"not|no|never|don't|doesn't|didn't|won't|shouldn't|"
    r"isn't|aren't|wasn't|can't|cannot|without)",
    re.IGNORECASE,
)

_UNCERTAINTY_PATTERNS = re.compile(
    r"(可能|也许|大概|或许|不确定|不太清楚|不知道|"
    r"maybe|might|possibly|probably|not\s+sure|uncertain|"
    r"I\s+guess|I\s+think\s+so|suppose|wonder)",
    re.IGNORECASE,
)


def extract_negation_features(content: str) -> np.ndarray:
    """Detect negation, deferral, and uncertainty (4 features).

    Key insight: "到时候再决定" contains "决定" (decision keyword)
    but "到时候再" flips the meaning to "defer". Without this feature,
    the gate sees "决定" and stores it. With this feature, the deferral
    signal counteracts the decision signal.
    """
    features = np.zeros(N_NEGATION_FEATURES, dtype=np.float32)
    if not content.strip():
        return features

    # 0: Deferral signal (strongest — "do it later" = don't store now)
    features[0] = 1.0 if _DEFERRAL_PATTERNS.search(content) else 0.0

    # 1: Negation present
    features[1] = 1.0 if _NEGATION_PATTERNS.search(content) else 0.0

    # 2: Uncertainty / hedging
    features[2] = 1.0 if _UNCERTAINTY_PATTERNS.search(content) else 0.0

    # 3: Combined poison: deferral + decision word = definitely don't store
    has_decision = bool(_DECISION_PATTERNS.search(content))
    has_deferral = bool(_DEFERRAL_PATTERNS.search(content))
    features[3] = 1.0 if (has_decision and has_deferral) else 0.0

    return features


# -- Question detection
# Must catch: "这个怎么部署的", "数据库密码在哪里", "Is the CI pipeline still broken?"
# Key: question words can appear ANYWHERE in Chinese, not just at the start.

_QUESTION_END = re.compile(
    r"[？?]$|\?$|吗$|呢$|吧$|么$",  # Ends with question mark or particle
)

_QUESTION_WORDS_ANYWHERE = re.compile(
    r"(什么|怎么|为什么|哪里|哪个|哪些|谁|几个|多少|是否|"
    r"有没有|能不能|可不可以|在哪|是啥|咋|啥时候|怎样|如何)",  # zh anywhere
    re.IGNORECASE,
)

_QUESTION_WORDS_EN_START = re.compile(
    r"^(what|how|why|when|where|who|which|whose|whom|"
    r"is\s+there|are\s+there|is\s+the|are\s+the|was\s+the|"
    r"can\s+we|can\s+I|do\s+we|does|did|will|should|could|would|"
    r"have\s+you|has\s+the|is\s+it|are\s+you)\b",
    re.IGNORECASE,
)

_QUESTION_WORDS_JAKO = re.compile(
    r"(何|どう|いつ|どこ|なぜ|誰|どれ|どの|ですか|ますか|"  # ja
    r"뭐|어떻게|왜|언제|어디|누가|인가요|입니까|나요)",       # ko
)


def extract_question_features(content: str) -> np.ndarray:
    """Detect if content is a question asking for info (not providing it)."""
    features = np.zeros(N_QUESTION_FEATURES, dtype=np.float32)
    stripped = content.strip()

    # 0: Has ANY question signal (anywhere in text, not just start)
    has_q = (
        bool(_QUESTION_END.search(stripped))
        or bool(_QUESTION_WORDS_ANYWHERE.search(stripped))
        or bool(_QUESTION_WORDS_EN_START.search(stripped))
        or bool(_QUESTION_WORDS_JAKO.search(stripped))
    )
    features[0] = 1.0 if has_q else 0.0

    # 1: Question with NO answer embedded (no "=", ":", numbers after colon)
    has_answer_signal = bool(re.search(r"[:=：]\s*\S+|→|->|端口\s*\d|port\s*\d", stripped, re.I))
    features[1] = 1.0 if has_q and not has_answer_signal else 0.0

    # 2: Short question (< 25 chars, likely asking not explaining)
    features[2] = 1.0 if has_q and len(stripped) < 25 else 0.0

    return features


# -- Off-topic / non-technical detection
_OFFTOPIC_PATTERNS = re.compile(
    r"(电影|电视剧|追剧|综艺|比赛|球赛|世界杯|奥运|"
    r"餐厅|火锅|咖啡|奶茶|美食|健身|爬山|旅游|"
    r"猫|狗|宠物|打球|游戏|手机|键盘|"
    r"movie|film|show|game|match|cup|"
    r"restaurant|pizza|coffee|gym|workout|travel|vacation|"
    r"cat|dog|pet|keyboard|phone|weather|traffic|"
    r"weekend|holiday|birthday|party)",
    re.IGNORECASE,
)

_TECH_PATTERNS = re.compile(
    r"(server|database|API|port|deploy|config|docker|kubernetes|k8s|"
    r"redis|postgres|mysql|nginx|aws|gcp|azure|git|CI|CD|"
    r"服务器|数据库|接口|端口|部署|配置|集群|监控|日志|"
    r"缓存|索引|查询|备份|证书|密钥|权限|"
    r"function|class|module|import|export|return|async|"
    r"error|bug|fix|test|build|release|version|"
    r"http|https|ssl|tls|tcp|udp|dns|ssh|"
    r"npm|pip|docker|kubectl|terraform|ansible)",
    re.IGNORECASE,
)


def extract_offtopic_features(content: str) -> np.ndarray:
    """Detect non-technical/off-topic content."""
    features = np.zeros(N_OFFTOPIC_FEATURES, dtype=np.float32)

    # 0: Has off-topic keywords
    features[0] = 1.0 if _OFFTOPIC_PATTERNS.search(content) else 0.0

    # 1: Has NO technical keywords (strong non-tech signal)
    features[1] = 0.0 if _TECH_PATTERNS.search(content) else 1.0

    # 2: Off-topic + no tech = definitely off-topic
    features[2] = 1.0 if features[0] > 0 and features[1] > 0 else 0.0

    return features


# -- Command detection patterns (catches "terraform plan", "psql -h", etc.)
_COMMAND_BINS = [
    # Binary names at start of string
    re.compile(r"^(sudo\s+)?(docker|kubectl|helm|terraform|ansible|vagrant|"
               r"npm|npx|yarn|pnpm|bun|pip|poetry|uv|"
               r"git|ssh|scp|rsync|curl|wget|"
               r"psql|mysql|redis-cli|mongo|"
               r"systemctl|journalctl|nginx|certbot|"
               r"python|node|java|go|cargo|make|cmake)\s", re.I),
    # Flags pattern: word followed by -- or - flags
    re.compile(r"\s-{1,2}[a-zA-Z][\w-]*", re.I),
    # Pipe/redirect
    re.compile(r"[|><]"),
    # File args with extensions
    re.compile(r"\S+\.(ya?ml|json|toml|conf|sh|py|ts|js|sql|tf|tfvars)\b", re.I),
    # Environment variable assignment
    re.compile(r"[A-Z_]{2,}=\S+"),
]

_PATH_PATTERNS = [
    re.compile(r"[~/.][\w./\\-]{3,}"),              # File paths
    re.compile(r"\b\w+://[^\s]+"),                    # Any URI
    re.compile(r"[a-zA-Z0-9._-]+\.(ya?ml|json|toml|conf|env|ini|cfg|sql|tf)\b", re.I),  # Config files
    re.compile(r"(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s", re.I),  # SQL
]


def extract_command_features(content: str) -> np.ndarray:
    """Detect if content is a command/script (5 features).

    Catches things like:
      'terraform plan -var-file=prod.tfvars'
      'psql -h prod-db -U admin -d myapp'
      'docker compose up -d --build'
    """
    features = np.zeros(N_COMMAND_FEATURES, dtype=np.float32)
    if not content.strip():
        return features

    # 0: Starts with known command binary
    features[0] = 1.0 if _COMMAND_BINS[0].search(content) else 0.0

    # 1: Has CLI flags (-x, --flag)
    flag_count = len(_COMMAND_BINS[1].findall(content))
    features[1] = min(flag_count / 3.0, 1.0)

    # 2: Has pipes/redirects
    features[2] = 1.0 if _COMMAND_BINS[2].search(content) else 0.0

    # 3: Has file args with extensions
    features[3] = 1.0 if _COMMAND_BINS[3].search(content) else 0.0

    # 4: Has env var assignment
    features[4] = 1.0 if _COMMAND_BINS[4].search(content) else 0.0

    return features


def extract_code_features(content: str) -> np.ndarray:
    """Detect paths, configs, code patterns (4 features)."""
    features = np.zeros(N_CODE_FEATURES, dtype=np.float32)
    if not content.strip():
        return features

    # 0: Contains file paths
    features[0] = 1.0 if _PATH_PATTERNS[0].search(content) else 0.0

    # 1: Contains URIs (any scheme)
    features[1] = 1.0 if _PATH_PATTERNS[1].search(content) else 0.0

    # 2: Contains config file references
    features[2] = 1.0 if _PATH_PATTERNS[2].search(content) else 0.0

    # 3: Contains SQL keywords
    features[3] = 1.0 if _PATH_PATTERNS[3].search(content) else 0.0

    return features


def compress_embedding(embedding: list[float] | None, target_dim: int = 24) -> np.ndarray:
    """Compress a high-dim embedding to fixed size via chunked averaging.

    768-dim embedding → split into 8 chunks of 96 → mean each → 8-dim vector.
    Captures the semantic "shape" without storing the full vector.
    """
    result = np.zeros(target_dim, dtype=np.float32)
    if embedding is None or len(embedding) == 0:
        return result

    arr = np.array(embedding, dtype=np.float32)
    chunk_size = max(1, len(arr) // target_dim)
    for i in range(target_dim):
        start = i * chunk_size
        end = min(start + chunk_size, len(arr))
        if start < len(arr):
            result[i] = float(np.mean(arr[start:end]))

    # Normalize to [-1, 1]
    norm = np.linalg.norm(result)
    if norm > 0:
        result /= norm

    return result


def extract_context_features(recent_topics: list[str], content: str) -> np.ndarray:
    """Features derived from recent conversation context.

    Captures: is this content related to what we've been discussing?
    If yes → more likely worth storing (continuation of important topic).
    If no → might be a topic shift (could be important or not).
    """
    features = np.zeros(N_CONTEXT_FEATURES, dtype=np.float32)
    if not recent_topics:
        return features

    content_tokens = set(tokenize(content.lower()))
    if not content_tokens:
        return features

    # 0: Overlap with last topic
    if recent_topics:
        last_tokens = set(tokenize(recent_topics[-1].lower()))
        if last_tokens and content_tokens:
            features[0] = len(content_tokens & last_tokens) / max(len(content_tokens), 1)

    # 1: Average overlap with last 3 topics
    overlaps = []
    for topic in recent_topics[-3:]:
        topic_tokens = set(tokenize(topic.lower()))
        if topic_tokens and content_tokens:
            overlaps.append(len(content_tokens & topic_tokens) / max(len(content_tokens), 1))
    features[1] = sum(overlaps) / max(len(overlaps), 1)

    # 2: Is this a topic shift? (low overlap with all recent)
    features[2] = 1.0 if features[1] < 0.1 else 0.0

    # 3: Topic depth (how many recent messages on same topic)
    same_topic_count = sum(1 for o in overlaps if o > 0.3)
    features[3] = min(same_topic_count / 3.0, 1.0)

    # 4: Conversation length signal
    features[4] = min(len(recent_topics) / 20.0, 1.0)

    # 5-7: Token frequency in recent context (are we seeing repeated entities?)
    all_recent_tokens: list[str] = []
    for t in recent_topics[-5:]:
        all_recent_tokens.extend(tokenize(t.lower()))
    freq = Counter(all_recent_tokens)
    repeated = [t for t in content_tokens if freq.get(t, 0) >= 2]
    features[5] = len(repeated) / max(len(content_tokens), 1)
    features[6] = min(len(freq) / 50.0, 1.0)  # vocabulary diversity
    features[7] = 1.0 if any(freq.get(t, 0) >= 3 for t in content_tokens) else 0.0

    return features


class UserProfile:
    """Tracks what topics this user cares about.

    Built incrementally from recalled topics — if a user keeps asking
    about "database" and "Redis", those topics get higher affinity scores.
    """

    def __init__(self) -> None:
        self.topic_affinity: Counter = Counter()
        self.total_recalls: int = 0

    def record_recall(self, content: str) -> None:
        """Record that the user recalled content about these tokens."""
        tokens = tokenize(content.lower())
        for t in set(tokens):
            self.topic_affinity[t] += 1
        self.total_recalls += 1

    def extract_features(self, content: str) -> np.ndarray:
        """How much does this content match the user's interests?"""
        features = np.zeros(N_PROFILE_FEATURES, dtype=np.float32)

        content_tokens = set(tokenize(content.lower()))
        if not content_tokens or self.total_recalls == 0:
            return features

        # 0: Max affinity of any token in content
        affinities = [self.topic_affinity.get(t, 0) for t in content_tokens]
        if affinities:
            features[0] = min(max(affinities) / 10.0, 1.0)

        # 1: Average affinity
        features[1] = min(sum(affinities) / max(len(affinities), 1) / 5.0, 1.0)

        # 2: Fraction of content tokens that are "known interests"
        known = sum(1 for t in content_tokens if self.topic_affinity.get(t, 0) >= 2)
        features[2] = known / max(len(content_tokens), 1)

        # 3: Is this in the user's top-10 topics?
        top_10 = set(t for t, _ in self.topic_affinity.most_common(10))
        features[3] = len(content_tokens & top_10) / max(len(content_tokens), 1)

        # 4: User engagement level (total recalls normalized)
        features[4] = min(self.total_recalls / 100.0, 1.0)

        # 5: Topic concentration (does user focus on few topics or many?)
        if len(self.topic_affinity) > 0:
            top_5_sum = sum(c for _, c in self.topic_affinity.most_common(5))
            total_sum = sum(self.topic_affinity.values())
            features[5] = top_5_sum / max(total_sum, 1)

        return features

    def to_dict(self) -> dict:
        return {
            "topic_affinity": dict(self.topic_affinity.most_common(100)),
            "total_recalls": self.total_recalls,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserProfile:
        profile = cls()
        profile.topic_affinity = Counter(data.get("topic_affinity", {}))
        profile.total_recalls = data.get("total_recalls", 0)
        return profile


# ---------------------------------------------------------------------------
# Two-layer neural network
# ---------------------------------------------------------------------------

class TwoLayerNet:
    """Minimal 2-layer neural network: Input → Hidden(ReLU) → Output(sigmoid).

    37 input → 16 hidden → 1 output = 608 parameters.
    Pure numpy, no framework needed.
    """

    def __init__(self, input_dim: int = N_FEATURES, hidden_dim: int = 16) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier initialization
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim).astype(np.float32) * scale2
        self.b2 = np.float32(0.0)

    def forward(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Forward pass. Returns (output_prob, hidden_activations)."""
        # Hidden layer: ReLU
        h = x @ self.W1 + self.b1
        h_relu = np.maximum(h, 0)  # ReLU

        # Output: sigmoid
        logit = float(h_relu @ self.W2 + self.b2)
        prob = _sigmoid_scalar(logit)

        return prob, h_relu

    def backward_step(self, x: np.ndarray, target: float, lr: float = 0.01) -> float:
        """One step of backpropagation. Returns loss."""
        # Forward
        h = x @ self.W1 + self.b1
        h_relu = np.maximum(h, 0)
        logit = float(h_relu @ self.W2 + self.b2)
        prob = _sigmoid_scalar(logit)

        # Loss
        eps = 1e-7
        loss = -(target * math.log(prob + eps) + (1 - target) * math.log(1 - prob + eps))

        # Output gradient
        d_logit = prob - target  # scalar

        # Gradient for W2, b2
        d_W2 = h_relu * d_logit
        d_b2 = d_logit

        # Backprop through ReLU
        d_h = self.W2 * d_logit  # (hidden_dim,)
        d_h[h <= 0] = 0  # ReLU mask

        # Gradient for W1, b1
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h

        # Update with L2 regularization
        reg = 0.001
        self.W2 -= lr * (d_W2 + reg * self.W2)
        self.b2 -= lr * d_b2
        self.W1 -= lr * (d_W1 + reg * self.W1)
        self.b1 -= lr * (d_b1 + reg * self.b1)

        return loss

    def to_dict(self) -> dict:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": float(self.b2),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TwoLayerNet:
        net = cls(data["input_dim"], data["hidden_dim"])
        net.W1 = np.array(data["W1"], dtype=np.float32)
        net.b1 = np.array(data["b1"], dtype=np.float32)
        net.W2 = np.array(data["W2"], dtype=np.float32)
        net.b2 = np.float32(data["b2"])
        return net


# ---------------------------------------------------------------------------
# LearnedGate v2 — main class
# ---------------------------------------------------------------------------

class LearnedGate:
    """Self-evolving neural gate with semantic + context + user profile awareness.

    Upgrades from v1:
    - 37 features (was 15): +semantic embedding +context window +user profile
    - 2-layer neural network (was logistic regression): captures non-linear patterns
    - User profile: tracks what topics the user actually cares about
    - Context awareness: evaluates content in relation to recent conversation
    """

    def __init__(self, online_lr: float = 0.02) -> None:
        self._net: TwoLayerNet | None = None
        self.threshold: float = 0.5
        self.trained = False

        self._online_lr = online_lr
        self._profile = UserProfile()
        self._recent_topics: deque[str] = deque(maxlen=20)

        self.stats = {
            "batch_samples": 0,
            "online_updates": 0,
            "implicit_positive": 0,
            "implicit_negative": 0,
            "corrections": 0,
        }

        self._replay_buffer: deque[dict] = deque(maxlen=500)

    # -- Feature extraction (full 37-dim) ------------------------------------

    def _extract_full_features(
        self,
        content: str,
        embedding: list[float] | None = None,
    ) -> np.ndarray:
        """Extract complete 72-dim feature vector."""
        rule = extract_rule_features(content)          # 15
        negation = extract_negation_features(content)  # 4
        question = extract_question_features(content)  # 3
        offtopic = extract_offtopic_features(content)  # 3
        command = extract_command_features(content)    # 5
        code = extract_code_features(content)          # 4
        semantic = compress_embedding(embedding)       # 24
        context = extract_context_features(list(self._recent_topics), content)  # 8
        profile = self._profile.extract_features(content)  # 6

        return np.concatenate([rule, negation, question, offtopic, command, code, semantic, context, profile])

    # -- Batch training ------------------------------------------------------

    def train(
        self,
        data: list[tuple[str, bool]],
        lr: float = 0.05,
        epochs: int = 300,
        embeddings: list[list[float] | None] | None = None,
    ) -> dict:
        """Batch train on labeled data.

        Args:
            data: List of (content, should_store) tuples
            embeddings: Optional embeddings for each content (same length as data)
        """
        if not data:
            return {"error": "no data"}

        embs = embeddings or [None] * len(data)

        X = np.array([
            self._extract_full_features(content, emb)
            for (content, _), emb in zip(data, embs)
        ])
        y = np.array([1.0 if label else 0.0 for _, label in data])

        # Initialize network if needed (preserve existing on re-train)
        if self._net is None:
            self._net = TwoLayerNet(input_dim=X.shape[1])

        total_loss = 0.0
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(len(data))
            epoch_loss = 0.0
            for i in idx:
                loss = self._net.backward_step(X[i], y[i], lr=lr)
                epoch_loss += loss
            total_loss = epoch_loss / len(data)

        # Evaluate
        correct = 0
        for i in range(len(data)):
            prob, _ = self._net.forward(X[i])
            if (prob >= self.threshold) == y[i].astype(bool):
                correct += 1
        accuracy = correct / len(data)

        self.trained = True
        self.stats["batch_samples"] += len(data)

        return {
            "accuracy": accuracy,
            "samples": len(data),
            "epochs": epochs,
            "loss": float(total_loss),
        }

    # -- Online learning -----------------------------------------------------

    def learn_online(self, content: str, should_store: bool, embedding: list[float] | None = None) -> None:
        """Single-sample online update."""
        if self._net is None:
            self._net = TwoLayerNet()

        features = self._extract_full_features(content, embedding)
        target = 1.0 if should_store else 0.0

        self._net.backward_step(features, target, lr=self._online_lr)
        self.trained = True
        self.stats["online_updates"] += 1

    def record_prediction(self, content: str, stored: bool, slot_id: int | None = None,
                          embedding: list[float] | None = None) -> None:
        """Record a prediction for later implicit feedback."""
        self._replay_buffer.append({
            "features": self._extract_full_features(content, embedding).tolist(),
            "content_preview": content[:100],
            "stored": stored,
            "slot_id": slot_id,
            "timestamp": time.time(),
            "recalled": False,
        })
        # Track as recent topic for context awareness
        if len(content) > 5:
            self._recent_topics.append(content)

    def feedback_recalled(self, slot_id: int) -> None:
        """Implicit positive: stored memory was recalled = good decision."""
        for entry in self._replay_buffer:
            if entry["slot_id"] == slot_id and entry["stored"] and not entry["recalled"]:
                entry["recalled"] = True
                features = np.array(entry["features"], dtype=np.float32)
                if self._net is None:
                    self._net = TwoLayerNet()
                self._net.backward_step(features, 1.0, lr=self._online_lr)
                self.stats["implicit_positive"] += 1

                # Update user profile
                self._profile.record_recall(entry["content_preview"])
                break

    def feedback_evicted(self, slot_id: int, was_ever_recalled: bool) -> None:
        """Implicit feedback from eviction."""
        if was_ever_recalled:
            return

        for entry in self._replay_buffer:
            if entry["slot_id"] == slot_id and entry["stored"] and not entry["recalled"]:
                features = np.array(entry["features"], dtype=np.float32)
                if self._net is None:
                    self._net = TwoLayerNet()
                # Weaker learning signal for indirect evidence
                self._net.backward_step(features, 0.0, lr=self._online_lr * 0.3)
                self.stats["implicit_negative"] += 1
                break

    def correct(self, content: str, should_have_stored: bool, embedding: list[float] | None = None) -> None:
        """Explicit user correction — strongest signal."""
        self.learn_online(content, should_have_stored, embedding)
        self.stats["corrections"] += 1

    # -- Prediction -----------------------------------------------------------

    def predict(self, content: str, embedding: list[float] | None = None) -> tuple[float, bool]:
        """Predict whether content should be stored."""
        if not self.trained or self._net is None:
            score = _importance_score(content)
            return score, score >= 0.20

        features = self._extract_full_features(content, embedding)
        prob, _ = self._net.forward(features)
        return prob, prob >= self.threshold

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model + profile + stats."""
        if self._net is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "net": self._net.to_dict(),
            "threshold": self.threshold,
            "stats": self.stats,
            "profile": self._profile.to_dict(),
            "recent_topics": list(self._recent_topics),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str | Path) -> bool:
        """Load model + profile + stats."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))

            if data.get("version", 1) >= 2:
                self._net = TwoLayerNet.from_dict(data["net"])
            else:
                # v1 format: logistic regression weights → can't directly convert
                # Start fresh with neural net
                return False

            self.threshold = data.get("threshold", 0.5)
            self.stats = data.get("stats", self.stats)
            self._profile = UserProfile.from_dict(data.get("profile", {}))
            self._recent_topics = deque(data.get("recent_topics", []), maxlen=20)
            self.trained = True
            return True
        except Exception:
            return False


# Backward compatibility: keep extract_features as a public function
def extract_features(content: str) -> np.ndarray:
    """Extract 15-dim rule features (backward compatible with v1 callers)."""
    return extract_rule_features(content)


def _sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))
