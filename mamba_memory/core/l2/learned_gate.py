"""Learned Gate — lightweight ML classifier trained on labeled data.

Enhances the rule-based gate with a trained feature-based classifier.
Uses hand-crafted features (no deep learning, no GPU needed) that
capture the same signals as the rule engine but with learned weights.

Training data format: list of (content, should_store: bool) tuples.

Features (15-dimensional):
  - 5 regex signal flags (decision, preference, correction, fact, explicit)
  - information density score
  - text length (log-scaled)
  - CJK character ratio
  - numeric token count
  - uppercase word count
  - punctuation density
  - stop word ratio
  - unique token ratio
  - action signal flag
  - filler/greeting flag (negative)
"""

from __future__ import annotations

import json
import math
import os
import re
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
    """Logistic regression classifier for gate decisions.

    Trained on labeled (content, should_store) pairs.
    Falls back to rule-based scoring if not trained.

    The model is intentionally simple:
    - No external ML library needed (pure numpy)
    - 15 hand-crafted features
    - Logistic regression with L2 regularization
    - Trains in <10ms on 100 samples
    """

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.threshold: float = 0.5
        self.trained = False

    def train(self, data: list[tuple[str, bool]], lr: float = 0.1, epochs: int = 200) -> dict:
        """Train the classifier on labeled data.

        Args:
            data: List of (content, should_store) tuples
            lr: Learning rate
            epochs: Training epochs

        Returns:
            Training metrics dict
        """
        if not data:
            return {"error": "no data"}

        X = np.array([extract_features(content) for content, _ in data])
        y = np.array([1.0 if label else 0.0 for _, label in data])

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

        # Mini-batch gradient descent with L2 regularization
        reg_lambda = 0.01
        for epoch in range(epochs):
            # Forward pass
            logits = X @ self.weights + self.bias
            preds = _sigmoid(logits)

            # Loss (binary cross-entropy + L2)
            eps = 1e-7
            loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            loss += 0.5 * reg_lambda * np.sum(self.weights ** 2)

            # Gradients
            errors = preds - y
            grad_w = (X.T @ errors) / n_samples + reg_lambda * self.weights
            grad_b = np.mean(errors)

            # Update
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

        # Evaluate
        final_preds = _sigmoid(X @ self.weights + self.bias) >= self.threshold
        accuracy = np.mean(final_preds == y.astype(bool))

        self.trained = True

        return {
            "accuracy": float(accuracy),
            "samples": n_samples,
            "epochs": epochs,
            "loss": float(loss),
        }

    def predict(self, content: str) -> tuple[float, bool]:
        """Predict whether content should be stored.

        Returns (confidence, should_store).
        """
        if not self.trained or self.weights is None:
            # Fallback to rule-based
            score = _importance_score(content)
            return score, score >= 0.20

        features = extract_features(content)
        logit = float(features @ self.weights + self.bias)
        prob = _sigmoid_scalar(logit)
        return prob, prob >= self.threshold

    def save(self, path: str | Path) -> None:
        """Save model weights to JSON file."""
        if self.weights is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str | Path) -> bool:
        """Load model weights from JSON file. Returns True if successful."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.weights = np.array(data["weights"], dtype=np.float32)
            self.bias = data["bias"]
            self.threshold = data.get("threshold", 0.5)
            self.trained = True
            return True
        except Exception:
            return False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))
