"""Batch generate training data using Gemini API.

Generates ~2000 labeled samples across 20+ categories and 6 languages,
then merges with existing template data and deduplicates.

Run: GOOGLE_API_KEY=... python scripts/generate_training_data.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from google import genai
from google.genai import types

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "training_data.json"

client = genai.Client(api_key=API_KEY)


def generate_batch(prompt: str, count: int = 50) -> list[dict]:
    """Call Gemini to generate labeled samples."""
    system = (
        "You are a training data generator for an AI memory gate classifier. "
        "Generate diverse, realistic examples. Output ONLY a JSON array, no markdown."
    )

    full_prompt = f"""{prompt}

Generate exactly {count} samples. Output a JSON array:
[
  {{"content": "...", "should_store": true/false, "category": "...", "lang": "..."}},
  ...
]

Rules:
- Each content should be 1-2 sentences, realistic conversational text
- Vary sentence structure, don't repeat patterns
- Mix formal and informal styles
- should_store: true = worth remembering long-term, false = noise/filler
- category: one of the categories mentioned in the prompt
- lang: "zh", "en", "ja", "ko", "fr", "de", "es", "mixed"
"""

    config = types.GenerateContentConfig(
        temperature=0.9,  # High creativity for diversity
        system_instruction=system,
    )

    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config=config,
        )
        text = resp.text.strip()
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text)
    except Exception as e:
        print(f"  Error: {e}")
        return []


def main():
    print("=" * 60)
    print("  MambaMemory Training Data Generator")
    print("=" * 60)

    all_samples: list[dict] = []

    # ── Batch definitions ──────────────────────────────────
    batches = [
        # SHOULD STORE categories
        {
            "prompt": (
                "Generate technical DECISION statements where someone commits to using "
                "a specific tool, framework, database, or approach. "
                "Languages: mix of Chinese, English, Japanese. "
                'Category: "decision". should_store: true.'
            ),
            "count": 80,
        },
        {
            "prompt": (
                "Generate FACTUAL CONFIGURATION statements containing specific values: "
                "IP addresses, ports, memory limits, timeouts, file paths, connection strings, "
                "version numbers, cron schedules, rate limits. "
                "Languages: mix of Chinese, English, mixed. "
                'Category: "fact". should_store: true.'
            ),
            "count": 100,
        },
        {
            "prompt": (
                "Generate PREFERENCE statements where someone expresses what they like/dislike "
                "about tools, coding styles, workflows, IDE settings. "
                "Languages: mix of Chinese, English, Japanese, Korean. "
                'Category: "preference". should_store: true.'
            ),
            "count": 50,
        },
        {
            "prompt": (
                "Generate CORRECTION statements where someone fixes a previous mistake: "
                "wrong port number, wrong version, wrong tool name, wrong config value. "
                "The correction should reference what was wrong and what's right. "
                "Languages: mix of Chinese, English. "
                'Category: "correction". should_store: true.'
            ),
            "count": 50,
        },
        {
            "prompt": (
                "Generate EXPLICIT MEMORY REQUESTS: someone asking the system to remember "
                "something important — passwords rotation policy, deployment rules, "
                "security policies, team conventions. Use words like 'remember', '记住', "
                "'don't forget', '别忘了', '覚えて'. "
                "Languages: Chinese, English, Japanese. "
                'Category: "explicit". should_store: true.'
            ),
            "count": 50,
        },
        {
            "prompt": (
                "Generate ACTION ITEMS and TODO statements: next steps in a project, "
                "tasks to complete, deadlines, follow-ups. "
                "Use words like TODO, FIXME, '下一步', 'need to', 'deadline'. "
                "Languages: mix of Chinese, English. "
                'Category: "action". should_store: true.'
            ),
            "count": 50,
        },
        {
            "prompt": (
                "Generate EXECUTABLE COMMANDS: shell commands, docker commands, kubectl, "
                "git commands, SQL queries, API calls with curl. "
                "These should be copy-paste ready commands, not descriptions. "
                'Category: "command". should_store: true. lang: "en" or "mixed".'
            ),
            "count": 50,
        },
        {
            "prompt": (
                "Generate ERROR LOG snippets and debugging findings: "
                "stack traces, error messages, root cause analysis conclusions, "
                "bug descriptions with reproduction steps. "
                "Languages: English, mixed. "
                'Category: "error_log". should_store: true.'
            ),
            "count": 40,
        },
        {
            "prompt": (
                "Generate MEETING NOTES and decision summaries: "
                "what was discussed and decided in a meeting, who is responsible for what, "
                "deadlines agreed upon. "
                "Languages: Chinese, English. "
                'Category: "meeting_note". should_store: true.'
            ),
            "count": 40,
        },
        {
            "prompt": (
                "Generate ARCHITECTURE DECISIONS: technology choices with rationale, "
                "design patterns adopted, trade-offs discussed. "
                "Languages: Chinese, English. "
                'Category: "architecture". should_store: true.'
            ),
            "count": 40,
        },

        # SHOULD DISCARD categories
        {
            "prompt": (
                "Generate GREETINGS and casual hellos in Chinese, English, Japanese, "
                "Korean, French, German, Spanish. Short, 1-5 words. "
                'Category: "greeting". should_store: false.'
            ),
            "count": 60,
        },
        {
            "prompt": (
                "Generate ACKNOWLEDGMENT responses: ok, sure, got it, 好的, understood, "
                "收到, alright, etc. Short confirmations with no information content. "
                "Languages: Chinese, English, Japanese, Korean, French. "
                'Category: "ack". should_store: false.'
            ),
            "count": 60,
        },
        {
            "prompt": (
                "Generate SMALL TALK and chitchat: weather comments, weekend plans, "
                "how are you, lunch topics, jokes, laughing reactions. "
                "Languages: Chinese, English, Japanese. "
                'Category: "smalltalk". should_store: false.'
            ),
            "count": 60,
        },
        {
            "prompt": (
                "Generate DEFERRAL and VAGUE statements: 'let me think about it', "
                "'到时候再决定', 'maybe later', 'TBD', 'not sure yet'. "
                "IMPORTANT: These may contain decision/action WORDS but the intent is to POSTPONE. "
                "Example: '以后再选择框架' contains '选择' but means 'decide later' = don't store. "
                "Languages: Chinese, English, Japanese. "
                'Category: "vague". should_store: false.'
            ),
            "count": 60,
        },
        {
            "prompt": (
                "Generate EMOTIONAL REACTIONS: excitement, frustration, surprise, "
                "encouragement. Short exclamations with no factual content. "
                "Like '太好了!', 'Amazing!', '糟糕', 'Nice!', '加油'. "
                "Languages: Chinese, English, Japanese, Korean. "
                'Category: "emotion". should_store: false.'
            ),
            "count": 50,
        },
        {
            "prompt": (
                "Generate QUESTIONS that are asking for information, not providing it: "
                "'什么意思?', 'How does this work?', 'Why?', '怎么做?'. "
                "These questions don't contain answers and shouldn't be stored. "
                "Languages: Chinese, English, Japanese. "
                'Category: "question". should_store: false.'
            ),
            "count": 50,
        },
        {
            "prompt": (
                "Generate FAREWELL and goodbye messages in multiple languages. "
                "Short, 1-5 words. '再见', 'bye', 'see you', 'さようなら', etc. "
                'Category: "farewell". should_store: false.'
            ),
            "count": 30,
        },
        {
            "prompt": (
                "Generate NON-TECHNICAL content that an AI assistant should NOT store: "
                "personal opinions about movies/food/sports, gossip, memes, "
                "random observations with no actionable value. "
                "Languages: Chinese, English. "
                'Category: "offtopic". should_store: false.'
            ),
            "count": 40,
        },
    ]

    # ── Run generation ─────────────────────────────────────
    total_target = sum(b["count"] for b in batches)
    print(f"\n  Target: {total_target} samples across {len(batches)} batches\n")

    for i, batch in enumerate(batches):
        count = batch["count"]
        # Split into chunks of 50 to avoid API limits
        chunk_size = 50
        generated = 0

        while generated < count:
            this_chunk = min(chunk_size, count - generated)
            print(f"  Batch {i+1}/{len(batches)}: generating {this_chunk} samples "
                  f"({generated}/{count} done)...", end=" ", flush=True)

            samples = generate_batch(batch["prompt"], this_chunk)
            if samples:
                all_samples.extend(samples)
                generated += len(samples)
                print(f"got {len(samples)}")
            else:
                print("failed, retrying...")
                time.sleep(2)

            time.sleep(1)  # Rate limit

    # ── Deduplicate ────────────────────────────────────────
    print(f"\n  Raw samples: {len(all_samples)}")

    seen: set[str] = set()
    unique: list[dict] = []
    for s in all_samples:
        key = s.get("content", "").strip().lower()
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            unique.append({
                "content": s["content"].strip(),
                "should_store": bool(s.get("should_store", False)),
                "category": s.get("category", "unknown"),
                "lang": s.get("lang", "unknown"),
            })

    print(f"  After dedup: {len(unique)}")

    # ── Stats ──────────────────────────────────────────────
    store = sum(1 for s in unique if s["should_store"])
    discard = sum(1 for s in unique if not s["should_store"])
    cats: dict[str, int] = {}
    langs: dict[str, int] = {}
    for s in unique:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
        langs[s["lang"]] = langs.get(s["lang"], 0) + 1

    print(f"  Store: {store}, Discard: {discard}")
    print(f"  Categories: {len(cats)}")
    for cat in sorted(cats.keys()):
        print(f"    {cat:<15} {cats[cat]:>4}")
    print(f"  Languages: {len(langs)}")
    for lang in sorted(langs.keys()):
        print(f"    {lang:<8} {langs[lang]:>4}")

    # ── Save ───────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(unique, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"  Total: {len(unique)} samples")


if __name__ == "__main__":
    main()
