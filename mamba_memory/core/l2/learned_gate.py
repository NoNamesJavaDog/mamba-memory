"""Learned Gate — self-evolving classifier that improves from usage.

Three learning modes:

1. **Batch training** — Train on a labeled dataset (cold start)
2. **Online learning** — Update weights from individual feedback samples
   in real-time as the user corrects the gate's decisions
3. **Implicit feedback** — Memories that get recalled frequently are
   retroactively labeled as "good stores"; memories that decay to
   eviction without ever being recalled are labeled as "bad stores"

The classifier uses 15 hand-crafted features and logistic regression.
No GPU, no external ML library — pure numpy, trains in <10ms.

Evolution flow:
  1. Gate makes a store/discard decision
  2. User uses the system (recalls some memories, ignores others)
  3. Recalled memories → positive feedback signal
  4. Evicted-without-recall memories → negative feedback signal
  5. Weights update incrementally (online SGD)
  6. Gate gets better at predicting what the user actually needs
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import deque
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

# Feature dimension
N_FEATURES = 15


def extract_features(content: str) -> np.ndarray:
    """Extract a fixed-size feature vector from text content."""
    features = np.zeros(N_FEATURES, dtype=np.float32)

    if not content.strip():
        return features

    # 0-4: Regex signal flags
    features[0] = 1.0 if _DECISION_PATTERNS.search(content) else 0.0
    features[1] = 1.0 if _PREFERENCE_PATTERNS.search(content) else 0.0
    features[2] = 1.0 if _CORRECTION_PATTERNS.search(content) else 0.0
    features[3] = 1.0 if _FACT_PATTERNS.search(content) else 0.0
    features[4] = 1.0 if _EXPLICIT_MEMORY_PATTERNS.search(content) else 0.0

    # 5: Information density
    features[5] = information_density(content)

    # 6: Log-scaled text length
    features[6] = math.log1p(len(content)) / 10.0

    # 7: CJK character ratio
    cjk_count = sum(1 for c in content if _is_cjk(c))
    features[7] = cjk_count / max(len(content), 1)

    # 8: Numeric token count (normalized)
    num_count = len(re.findall(r"\d+", content))
    features[8] = min(num_count / 5.0, 1.0)

    # 9: Uppercase word count (normalized)
    upper_count = len(re.findall(r"[A-Z][a-zA-Z]+", content))
    features[9] = min(upper_count / 5.0, 1.0)

    # 10: Punctuation density
    punct_count = sum(1 for c in content if not c.isalnum() and not c.isspace())
    features[10] = punct_count / max(len(content), 1)

    # 11: Stop word ratio
    words = re.findall(r"\w+", content.lower())
    if words:
        stop_count = sum(1 for w in words if w in _STOP_WORDS)
        features[11] = stop_count / len(words)
    else:
        features[11] = 0.0

    # 12: Unique token ratio (lexical diversity)
    tokens = tokenize(content)
    if tokens:
        features[12] = len(set(tokens)) / len(tokens)
    else:
        features[12] = 0.0

    # 13: Action signal
    features[13] = 1.0 if _ACTION_PATTERNS.search(content) else 0.0

    # 14: Filler/greeting (negative signal)
    features[14] = 1.0 if _FILLER_PATTERNS.match(content.strip()) else 0.0

    return features


class LearnedGate:
    """Self-evolving logistic regression classifier for gate decisions.

    Learns from three sources:
    1. Explicit labeled data (batch training / cold start)
    2. User corrections ("this should have been stored/discarded")
    3. Implicit feedback (recalled = good, evicted-without-recall = bad)

    The model evolves continuously via online SGD — each feedback sample
    updates the weights immediately without full retraining.
    """

    def __init__(self, online_lr: float = 0.05) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.threshold: float = 0.5
        self.trained = False

        # Online learning config
        self._online_lr = online_lr
        self._reg_lambda = 0.01

        # Stats for monitoring evolution
        self.stats = {
            "batch_samples": 0,
            "online_updates": 0,
            "implicit_positive": 0,
            "implicit_negative": 0,
            "corrections": 0,
        }

        # Replay buffer — stores recent predictions for implicit feedback
        # (content_features, predicted_store, slot_id, timestamp)
        self._replay_buffer: deque[dict] = deque(maxlen=500)

    # -- Batch training (cold start) -----------------------------------------

    def train(self, data: list[tuple[str, bool]], lr: float = 0.1, epochs: int = 200) -> dict:
        """Batch train on labeled data. Used for cold start.

        Does NOT reset weights if already trained — continues from current state.
        This means you can batch-train, then online-learn, then batch-train again
        on new data without losing previous knowledge.
        """
        if not data:
            return {"error": "no data"}

        X = np.array([extract_features(content) for content, _ in data])
        y = np.array([1.0 if label else 0.0 for _, label in data])

        n_samples, n_features = X.shape

        # Initialize weights only if not yet trained (preserve existing knowledge)
        if self.weights is None:
            self.weights = np.zeros(n_features, dtype=np.float32)
            self.bias = 0.0

        for epoch in range(epochs):
            logits = X @ self.weights + self.bias
            preds = _sigmoid(logits)

            eps = 1e-7
            loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            loss += 0.5 * self._reg_lambda * np.sum(self.weights ** 2)

            errors = preds - y
            grad_w = (X.T @ errors) / n_samples + self._reg_lambda * self.weights
            grad_b = np.mean(errors)

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

        final_preds = _sigmoid(X @ self.weights + self.bias) >= self.threshold
        accuracy = np.mean(final_preds == y.astype(bool))

        self.trained = True
        self.stats["batch_samples"] += n_samples

        return {
            "accuracy": float(accuracy),
            "samples": n_samples,
            "epochs": epochs,
            "loss": float(loss),
            "total_training_samples": self.stats["batch_samples"],
        }

    # -- Online learning (real-time evolution) --------------------------------

    def learn_online(self, content: str, should_store: bool) -> None:
        """Update weights from a single feedback sample (online SGD).

        Call this when:
        - User explicitly corrects a decision ("no, remember this" / "forget that")
        - System detects implicit feedback (recalled = good, evicted = bad)

        Each call does ONE gradient step — lightweight, <0.1ms.
        """
        if self.weights is None:
            self.weights = np.zeros(N_FEATURES, dtype=np.float32)

        features = extract_features(content)
        target = 1.0 if should_store else 0.0

        logit = float(features @ self.weights + self.bias)
        pred = _sigmoid_scalar(logit)

        error = pred - target
        self.weights -= self._online_lr * (error * features + self._reg_lambda * self.weights)
        self.bias -= self._online_lr * error

        self.trained = True
        self.stats["online_updates"] += 1

    def record_prediction(self, content: str, stored: bool, slot_id: int | None = None) -> None:
        """Record a gate prediction for later implicit feedback.

        Called after every gate decision. When we later observe that
        a stored memory was recalled (positive) or evicted without
        recall (negative), we use this to generate training signal.
        """
        self._replay_buffer.append({
            "features": extract_features(content).tolist(),
            "content_hash": hash(content),
            "stored": stored,
            "slot_id": slot_id,
            "timestamp": time.time(),
            "recalled": False,
            "evicted_without_recall": False,
        })

    def feedback_recalled(self, slot_id: int) -> None:
        """Implicit positive feedback: a stored memory was recalled.

        This means the gate made a GOOD decision to store it.
        Strengthens the weights toward storing similar content.
        """
        for entry in self._replay_buffer:
            if entry["slot_id"] == slot_id and entry["stored"] and not entry["recalled"]:
                entry["recalled"] = True
                features = np.array(entry["features"], dtype=np.float32)
                self._online_step(features, target=1.0)
                self.stats["implicit_positive"] += 1
                break

    def feedback_evicted(self, slot_id: int, was_ever_recalled: bool) -> None:
        """Implicit feedback from eviction.

        - If the slot was recalled at least once before eviction → neutral
          (it served its purpose, natural lifecycle)
        - If the slot was NEVER recalled → negative feedback
          (gate stored something useless, should learn not to)
        """
        if was_ever_recalled:
            return  # Normal lifecycle, no negative signal

        for entry in self._replay_buffer:
            if entry["slot_id"] == slot_id and entry["stored"] and not entry["recalled"]:
                entry["evicted_without_recall"] = True
                features = np.array(entry["features"], dtype=np.float32)
                # Weaker signal than explicit correction (0.3 weight)
                self._online_step(features, target=0.0, lr_scale=0.3)
                self.stats["implicit_negative"] += 1
                break

    def correct(self, content: str, should_have_stored: bool) -> None:
        """Explicit user correction — strongest learning signal.

        Call when user says "you should have remembered this" or
        "why did you store that garbage".
        """
        self.learn_online(content, should_have_stored)
        self.stats["corrections"] += 1

    # -- Prediction -----------------------------------------------------------

    def predict(self, content: str) -> tuple[float, bool]:
        """Predict whether content should be stored.

        Returns (confidence, should_store).
        """
        if not self.trained or self.weights is None:
            score = _importance_score(content)
            return score, score >= 0.20

        features = extract_features(content)
        logit = float(features @ self.weights + self.bias)
        prob = _sigmoid_scalar(logit)
        return prob, prob >= self.threshold

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights + stats to JSON file."""
        if self.weights is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "stats": self.stats,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str | Path) -> bool:
        """Load model weights + stats from JSON file."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.weights = np.array(data["weights"], dtype=np.float32)
            self.bias = data["bias"]
            self.threshold = data.get("threshold", 0.5)
            self.stats = data.get("stats", self.stats)
            self.trained = True
            return True
        except Exception:
            return False

    # -- Internal -------------------------------------------------------------

    def _online_step(self, features: np.ndarray, target: float, lr_scale: float = 1.0) -> None:
        """Single online SGD step."""
        if self.weights is None:
            self.weights = np.zeros(N_FEATURES, dtype=np.float32)

        logit = float(features @ self.weights + self.bias)
        pred = _sigmoid_scalar(logit)
        error = pred - target
        lr = self._online_lr * lr_scale

        self.weights -= lr * (error * features + self._reg_lambda * self.weights)
        self.bias -= lr * error
        self.trained = True


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))
