"""Text analysis utilities — tokenization, relevance scoring, entity extraction.

Provides language-agnostic text processing for both L1 (compression, recall)
and L2 (gate importance scoring). No external NLP library required.

Key design choices:
  - CJK: character-level ngrams (bigram/trigram), no word segmentation needed
  - Alphabetic: whitespace tokenization + lowercasing
  - Mixed: handles CJK + Latin mixed text naturally
  - Stop words: lightweight built-in lists for zh/en/ja/ko
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


# ---------------------------------------------------------------------------
# Unicode range helpers
# ---------------------------------------------------------------------------

def _is_cjk(ch: str) -> bool:
    """Check if a character is CJK (Chinese/Japanese/Korean ideograph)."""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)       # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)    # Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # Extension B
        or (0xF900 <= cp <= 0xFAFF)    # Compatibility
        or (0x2F800 <= cp <= 0x2FA1F)  # Supplement
        or (0x3040 <= cp <= 0x30FF)    # Hiragana + Katakana
        or (0xAC00 <= cp <= 0xD7AF)    # Korean Hangul
    )


def _is_punctuation(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S") or ch in "？！。，、；：""''（）【】《》"


# ---------------------------------------------------------------------------
# Stop words (lightweight, covers zh/en/ja common function words)
# ---------------------------------------------------------------------------

_STOP_WORDS_ZH = frozenset(
    "的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 那 她 他 它 们 吧 吗 呢 啊 呀 哦 嗯 么 什么 怎么 为什么 哪 谁 "
    "可以 这个 那个 还 从 但 但是 因为 所以 如果 虽然 而 或 把 被 给 让 对 "
    "已经 正在 将 又 再 更 最 太 非常".split()
)

_STOP_WORDS_EN = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "about between through during before after above below and but or nor not "
    "so yet both either neither each every all any few more most other some such "
    "no only own same than too very just don doesn didn won wouldn it its i me my "
    "we our you your he him his she her they them their this that these those "
    "what which who whom how when where why".split()
)

_STOP_WORDS = _STOP_WORDS_ZH | _STOP_WORDS_EN


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize text into meaningful tokens.

    Strategy:
    - CJK characters → individual characters (each is a semantic unit)
    - Alphabetic/numeric runs → lowercased words
    - Punctuation and whitespace → discarded
    - Stop words → filtered out

    Returns deduplicated tokens preserving order.
    """
    tokens: list[str] = []
    buf: list[str] = []

    def flush_buf():
        if buf:
            word = "".join(buf).lower()
            if word and word not in _STOP_WORDS and len(word) > 1:
                tokens.append(word)
            buf.clear()

    for ch in text:
        if _is_cjk(ch):
            flush_buf()
            if ch not in _STOP_WORDS_ZH:
                tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            flush_buf()

    flush_buf()
    return tokens


def extract_ngrams(text: str, n: int = 2) -> list[str]:
    """Extract character-level ngrams from CJK portions + word ngrams from Latin.

    For "数据库选择了PostgreSQL":
      bigrams: ["数据", "据库", "库选", "选择"]  (skip stop chars)
      words: ["postgresql"]
    """
    tokens = tokenize(text)
    ngrams: list[str] = []

    # CJK bigrams/trigrams from raw text
    cjk_chars = [ch for ch in text if _is_cjk(ch) and ch not in _STOP_WORDS_ZH]
    for i in range(len(cjk_chars) - n + 1):
        ngrams.append("".join(cjk_chars[i : i + n]))

    # Word-level tokens (already extracted)
    ngrams.extend(t for t in tokens if not _is_cjk(t[0]) if len(t) > 1)

    return ngrams


# ---------------------------------------------------------------------------
# Text relevance scoring
# ---------------------------------------------------------------------------

def text_relevance(query: str, text: str) -> float:
    """Compute relevance score between query and text (0–1).

    Multi-signal scoring:
    1. Exact substring match (highest signal)
    2. Bigram overlap (CJK-friendly, O(n))
    3. Token overlap (word-level)
    4. Weighted combination

    Much faster and more accurate than the old O(n^2) approach.
    """
    if not query or not text:
        return 0.0

    query_lower = query.lower()
    text_lower = text.lower()

    scores: list[float] = []

    # Signal 1: exact substring match (bidirectional)
    if query_lower in text_lower:
        # Query fully contained in text — strong match
        coverage = len(query_lower) / max(len(text_lower), 1)
        scores.append(0.7 + 0.3 * min(coverage * 5, 1.0))
    elif text_lower in query_lower:
        scores.append(0.6)

    # Signal 2: bigram overlap (fast, CJK-friendly)
    q_bigrams = set(extract_ngrams(query_lower, 2))
    t_bigrams = set(extract_ngrams(text_lower, 2))
    if q_bigrams and t_bigrams:
        overlap = len(q_bigrams & t_bigrams)
        bigram_score = overlap / len(q_bigrams)
        scores.append(bigram_score * 0.7)

    # Signal 3: token overlap
    q_tokens = set(tokenize(query_lower))
    t_tokens = set(tokenize(text_lower))
    if q_tokens and t_tokens:
        overlap = len(q_tokens & t_tokens)
        token_score = overlap / len(q_tokens)
        scores.append(token_score * 0.6)

    return max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Structured information extraction (for L1 fallback compression)
# ---------------------------------------------------------------------------

# Patterns for extracting structured data from conversations
# Note: avoid \b around CJK — it doesn't match CJK-to-ASCII boundaries.
# Use lookaround or drop \b for patterns that appear in mixed CJK/Latin text.
_PATTERNS = {
    "ip_address": re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"),
    "port": re.compile(r"(?:port|端口)\s*[:=：是为]?\s*(\d{2,5})", re.I),
    "url": re.compile(r"https?://[^\s<>\"']+"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "version": re.compile(r"[vV]?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?"),
    "file_path": re.compile(r"[~/.][\w./\\-]+\.(?:json|yaml|yml|toml|conf|cfg|ini|env|sh|py|ts|js)"),
    "number_with_unit": re.compile(r"\d+\s*(?:MB|GB|TB|KB|ms|秒|分钟|小时|天|个|次|条)", re.I),
}

# Sentence-ending patterns for splitting
_SENTENCE_END = re.compile(r"[。！？.!?\n]+")


def extract_key_facts(text: str) -> list[str]:
    """Extract key factual sentences from text.

    Keeps sentences that contain structured data (IPs, ports, URLs, versions, etc.)
    or decision/preference/correction signals.
    """
    sentences = _SENTENCE_END.split(text)
    facts: list[str] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 4:
            continue

        # Check if sentence contains structured data
        has_structure = any(p.search(sent) for p in _PATTERNS.values())

        # Check if sentence contains decision/action signals
        has_signal = bool(re.search(
            r"(决定|选择|使用|配置|设置|部署|安装|迁移|切换|创建|删除|修改|更新|改成|换成|设为|改为|启用|禁用|"
            r"adopt|choose|decide|use|config|setup|deploy|install|migrate|switch|"
            r"create|delete|modify|update|change|set|enable|disable|"
            r"记住|注意|重要|必须|不要|avoid|remember|important|must|never|always)",
            sent,
            re.I,
        ))

        if has_structure or has_signal:
            facts.append(sent)

    return facts


def extract_entities_simple(text: str) -> list[str]:
    """Extract likely entity names from text without NLP.

    Heuristics:
    1. Capitalized words (English proper nouns / tech names)
    2. CJK named patterns (consecutive CJK chars between known markers)
    3. Technical terms (known patterns)
    """
    entities: list[str] = []

    # Capitalized words / tech names (2+ chars)
    # Use pattern without \b to handle CJK-adjacent Latin words
    cap_words = re.findall(r"([A-Z][a-zA-Z0-9]{1,}(?:[.-][a-zA-Z0-9]+)*)", text)
    for w in cap_words:
        if len(w) >= 2 and w.lower() not in _STOP_WORDS_EN:
            entities.append(w)

    # Known tech patterns (no \b — works in CJK context)
    tech_patterns = re.findall(
        r"(Python|Java|JavaScript|TypeScript|Go|Rust|Ruby|PHP|C\+\+|C#|Swift|Kotlin|"
        r"React|Vue|Angular|Next\.?js|Node\.?js|Django|Flask|FastAPI|Spring|Rails|"
        r"PostgreSQL|MySQL|MongoDB|Redis|SQLite|Elasticsearch|Kafka|RabbitMQ|"
        r"Docker|Kubernetes|K8s|Nginx|Apache|AWS|GCP|Azure|Vercel|"
        r"Git|GitHub|GitLab|Linux|macOS|Windows|iOS|Android)",
        text,
        re.I,
    )
    entities.extend(tech_patterns)

    # CJK person-name-like patterns: 2-4 CJK chars after name markers
    name_matches = re.findall(r"(?:叫|是|找|问|联系|@)\s*([一-龥]{2,4})", text)
    entities.extend(name_matches)

    # Dedup preserving order
    seen: set[str] = set()
    result: list[str] = []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            result.append(e)

    return result


def compress_turns_structured(
    turns: list[tuple[str, str]],
    max_chars: int = 600,
) -> tuple[str, list[str]]:
    """Structured fallback compression for conversation turns.

    Args:
        turns: List of (role, content) tuples
        max_chars: Maximum output characters

    Returns:
        (summary_text, entities_list)

    Strategy:
    1. Extract key facts from all turns
    2. Extract entities
    3. Build a structured summary: facts + context
    4. Truncate to fit budget
    """
    all_text = "\n".join(content for _, content in turns)

    # Extract structured information
    facts = extract_key_facts(all_text)
    entities = extract_entities_simple(all_text)

    if facts:
        # Use extracted facts as the summary
        summary_parts: list[str] = []
        used = 0
        for fact in facts:
            if used + len(fact) > max_chars:
                break
            summary_parts.append(fact)
            used += len(fact) + 2  # +2 for separator

        summary = "; ".join(summary_parts)
    else:
        # No structured facts found — fall back to role-prefixed truncation
        parts: list[str] = []
        used = 0
        for role, content in turns:
            # Take first meaningful sentence from each turn
            first_sent = _SENTENCE_END.split(content)[0].strip()
            if not first_sent:
                continue
            line = f"{role}: {first_sent[:120]}"
            if used + len(line) > max_chars:
                break
            parts.append(line)
            used += len(line) + 3

        summary = " | ".join(parts)

    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."

    return summary, entities


# ---------------------------------------------------------------------------
# Information density scoring (for L2 gate)
# ---------------------------------------------------------------------------

def information_density(text: str) -> float:
    """Score how information-dense a piece of text is (0–1).

    High density: contains facts, numbers, names, configurations
    Low density: greetings, filler, pure emotion

    Used as one signal in the L2 gate importance evaluation.
    """
    if not text:
        return 0.0

    length = len(text)
    tokens = tokenize(text)
    unique_tokens = set(tokens)

    signals: list[float] = []

    # 1. Structured data presence (0 or 1 each, averaged)
    struct_count = sum(1 for p in _PATTERNS.values() if p.search(text))
    signals.append(min(struct_count / 3.0, 1.0))  # cap at 1.0

    # 2. Lexical diversity (unique tokens / total tokens)
    if tokens:
        diversity = len(unique_tokens) / len(tokens)
        signals.append(diversity)
    else:
        signals.append(0.0)

    # 3. Content-to-stopword ratio
    all_words = re.findall(r"\w+", text.lower())
    if all_words:
        content_words = [w for w in all_words if w not in _STOP_WORDS and len(w) > 1]
        signals.append(len(content_words) / len(all_words))
    else:
        signals.append(0.0)

    # 4. Length signal (very short = likely filler, medium = good, very long = verbose)
    if length < 5:
        signals.append(0.0)
    elif length < 15:
        signals.append(0.2)
    elif length < 200:
        signals.append(0.7)
    else:
        signals.append(0.5)  # long text might be verbose

    # 5. Numeric content (numbers carry information)
    num_count = len(re.findall(r"\d+", text))
    signals.append(min(num_count / 3.0, 1.0))

    return sum(signals) / len(signals)
