#!/usr/bin/env python3
"""
Generate and add ~2500 DISCARD samples to balance the training data.
Target: Move from 6380:563 (91:9) to closer to 50:50 balance.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Define comprehensive word pools for each category
GREETING_WORDS = {
    "zh": ["你好", "嗨", "早上好", "晚上好", "下午好", "嗨嗨", "哈喽", "你好啊", "怎么样", "咋样",
           "哟", "喂", "诶", "哈", "呃", "嘿", "嘿嘿", "歪歪", "各位", "诸位",
           "大家", "老兄", "老哥", "伙计", "小哥", "好久不见", "好几天没聊了"],
    "en": ["hi", "hello", "hey", "hey there", "yo", "heya", "what's up", "howdy", "greetings", "good morning",
           "good afternoon", "good evening", "sup", "yo yo", "whaddup", "how it's", "hiya", "hullo", "eyo", "wassup"],
    "ja": ["こんにちは", "やあ", "おはよう", "こんばんは", "どうも", "やあやあ", "よ", "こんちは", "どす", "あ"],
    "fr": ["bonjour", "salut", "coucou", "ça va", "allô", "yo", "allo"],
    "es": ["hola", "qué tal", "buenos días", "hola hola", "ey", "halo"],
    "de": ["guten tag", "hallo", "grüß dich", "wie geht's", "hallo zusammen"],
    "ko": ["안녕", "안녕하세요", "안녕하시나요", "뭐하니", "여봐요"]
}

ACK_WORDS = {
    "zh": ["好的", "嗯嗯", "收到", "了解", "明白了", "知道了", "没问题", "可以", "右", "对对", "同意",
           "行", "成", "OK", "搞定", "妥了", "稳", "OK的", "明了", "晓得", "清楚",
           "理解", "接受", "赞同", "同意", "就这样", "那就这样", "好吧"],
    "en": ["ok", "okay", "sure", "got it", "understood", "sounds good", "makes sense", "fair enough", "i see", "cool",
           "yep", "yup", "absolutely", "for sure", "definitely", "right", "uh huh", "kk", "copy that", "roger that",
           "all good", "no problem", "that works", "i'm on it", "got you"],
    "ja": ["はい", "わかりました", "了解です", "なるほど", "いいです", "大丈夫", "いってきます", "ウケた", "レディ", "了"],
    "fr": ["d'accord", "oui", "c'est bon", "entendu", "bien sûr", "compris"],
    "es": ["vale", "de acuerdo", "entendido", "claro", "perfecto"],
    "de": ["ja", "okay", "verstanden", "alles klar", "verständnis"],
    "ko": ["네", "알겠습니다", "좋아", "맞아", "오케이"]
}

SMALLTALK_TOPICS = {
    "zh": [
        "今天天气真好", "天气这么热", "最近下雨太多了", "早上上班要挤地铁",
        "中午吃什么", "又到午餐时间了", "咖啡喝了没", "周末计划", "周末去哪玩",
        "这周太累了", "好累啊", "困死了", "快到周五了", "TGIF", "终于周末了",
        "哈哈真逗", "笑死我了", "有意思", "那个笑话太冷了", "最近有什么趣事",
        "球赛看了吗", "体育比赛", "最近身体如何", "去健身了吗"
    ],
    "en": [
        "nice weather today", "it's so hot", "raining a lot lately", "morning commute is rough",
        "what's for lunch", "coffee time", "weekend plans", "what did you do this weekend",
        "been a long week", "so tired", "can't wait for friday", "TGIF", "finally weekend",
        "that's funny", "made me laugh", "amusing", "that joke was terrible", "anything interesting lately",
        "did you catch the game", "sports are cool", "how's your health", "hitting the gym"
    ],
    "ja": [
        "いい天気だね", "暑いね", "最近雨が多い", "朝の通勤大変",
        "昼ご飯何にしよう", "コーヒータイム", "週末の予定", "週末どこ行った",
        "疲れたね", "眠い", "金曜日待ってた", "やっと週末", "面白いね", "笑った",
        "スポーツ好き", "体調どう", "最近のニュース"
    ]
}

VAGUE_WORDS = {
    "zh": [
        "以后再选择框架", "还没想好用什么", "考虑中", "未定", "未决", "还在评估",
        "可能会用K8s", "说不定某天迁移", "目前不确定", "TBD", "待定", "再想想",
        "没有最终决定", "还在纠结", "暂时这样", "先这样吧", "以后再看", "等等再说"
    ],
    "en": [
        "maybe we'll use K8s someday", "not decided on the database yet", "TBD", "pending",
        "let me think", "still evaluating", "haven't decided yet", "not sure yet", "might change later",
        "undecided for now", "will revisit later", "keeping options open", "could go either way",
        "figure it out later", "not finalized yet", "still considering"
    ],
    "ja": [
        "いつか移行する", "まだ決まってない", "検討中", "保留中", "未定",
        "考え中", "後で決める", "今のところ不明", "今後の課題", "いずれ", "また今度"
    ],
    "mixed": [
        "database migration is a maybe", "framework choice still up in the air",
        "not locked in yet", "flexible on the architecture", "no firm decision",
        "架构未定", "方案待评估", "还没敲定", "可能会变"
    ]
}

FAREWELL_WORDS = {
    "zh": ["再见", "拜拜", "回见", "待会儿见", "明天见", "周一见", "一会儿见"],
    "en": ["bye", "goodbye", "see you", "see you later", "ttyl", "catch you later", "talk soon", "cya"],
    "ja": ["さようなら", "じゃあね", "またね", "バイバイ", "後でね", "また明日"],
    "fr": ["adieu", "au revoir", "à bientôt"],
    "es": ["adiós", "hasta luego", "hasta pronto"],
    "de": ["tschüss", "auf wiedersehen", "alles gute"],
    "ko": ["안녕히", "잠깐", "나중에봐", "또봐"]
}

QUESTION_MARKERS = {
    "zh": ["这个怎么", "怎么", "如何", "哪里", "谁有", "什么时候", "为什么", "什么意思", "怎么回事", "这是什么", "在哪"],
    "en": ["how does", "how do", "what is", "why is", "who has", "where is", "when is", "what's", "how to", "any ideas"],
    "ja": ["どうやって", "なぜ", "いつ", "何時", "どこ", "誰"],
    "fr": ["comment", "pourquoi", "quand", "où"],
    "ko": ["어떻게", "왜", "언제", "누가", "어디"]
}

QUESTION_TOPICS = {
    "zh": ["部署的", "密码在哪", "权限怎么设置", "怎么配置", "在哪里", "是什么意思", "失败了", "出错了",
           "怎么用", "咋回事", "什么鬼", "太奇怪", "谁知道", "谁有", "如何解决", "该咋办", "给我讲讲"],
    "en": ["work", "access this", "this failing", "configure this", "where is", "what does this mean", "broke", "error",
           "how to use", "what's going on", "that's weird", "who knows", "who has", "what should i do", "explain"],
    "ja": ["動く", "アクセス", "失敗", "エラー", "どう設定", "どこに", "何のこと", "なぜ"],
}

EMOTION_WORDS = {
    "zh": ["太好了", "太棒了", "太强了", "糟糕", "哎呀", "天哪", "天啊", "赞", "厉害", "不错"],
    "en": ["amazing", "awesome", "great", "fantastic", "oh no", "damn", "wow", "cool", "nice", "awesome"],
    "ja": ["やった", "いいね", "すごい", "ヤバい", "大変", "やばい"],
    "emoji": ["🎉", "👍", "❤️", "😂", "🔥", "😱", "😨"]
}

OFFTOPIC_WORDS = {
    "zh": ["那部电影真好看", "电影不错", "去看了电影", "新片上映", "去登山了", "山里很舒服", "买了新键盘",
           "新手机", "购物", "逛街", "旅游", "度假", "小狗", "猫咪", "宠物", "家里的"],
    "en": ["watched a great movie", "new movie coming out", "went hiking", "beautiful scenery", "bought a new keyboard",
           "new phone", "shopping online", "went traveling", "vacation plans", "got a puppy", "my cat", "pets are great",
           "at home with", "love hiking"],
    "ja": ["映画見た", "いい映画", "登山した", "景色がいい", "新しいキーボード", "ペット"],
}

FILLER_WORDS = {
    "zh": ["嗯", "呵呵", "233", "...", "嘻嘻", "哈", "额", "呃", "呵"],
    "en": ["hmm", "well", "um", "uh", "like", "you know", "...", "haha", "lol"],
    "ja": ["えっと", "あのう", "うーん", "ふむ", "ふむふむ", "..."],
    "symbol": ["...", "...", "...", ";;;", "~~~"]
}


def generate_greeting_samples(count=300):
    """Generate greeting messages in multiple languages with diverse templates."""
    samples = []
    templates = {
        "zh": [
            "{greeting}", "{greeting}😊", "{greeting}～", "嘿，{greeting}", "{greeting}呀",
            "{greeting}，最近怎样", "早上好，{greeting}", "{greeting}，有空吗", "{greeting}！好久不见"
        ],
        "en": [
            "{greeting}", "{greeting}!", "{greeting} there", "hey, {greeting}", "{greeting} friend",
            "{greeting}, how are you", "{greeting}, what's up", "morning, {greeting}", "{greeting}, long time"
        ],
        "ja": [
            "{greeting}", "{greeting}ね", "{greeting}よ", "{greeting}あ", "{greeting}ー",
            "{greeting}、元気", "{greeting}、久しぶり", "おす、{greeting}"
        ]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja"])
        greeting = random.choice(GREETING_WORDS.get(lang, GREETING_WORDS["en"]))
        template = random.choice(templates.get(lang, templates["en"]))
        content = template.format(greeting=greeting)

        samples.append({
            "content": content,
            "should_store": False,
            "category": "greeting",
            "lang": lang
        })
    return samples


def generate_ack_samples(count=400):
    """Generate acknowledgment messages with diverse templates."""
    samples = []
    templates = {
        "zh": [
            "{ack}", "{ack}！", "{ack}呢", "{ack}的", "嗯，{ack}", "{ack}，我看看",
            "明白，{ack}", "{ack}吧", "那就{ack}了", "{ack}，就这样", "可以，{ack}"
        ],
        "en": [
            "{ack}", "{ack}!", "{ack}.", "{ack}, thanks", "sounds {ack}", "yep, {ack}",
            "{ack}, got it", "totally {ack}", "absolutely {ack}", "{ack} with you", "{ack}, let's go"
        ],
        "ja": [
            "{ack}", "{ack}ね", "{ack}です", "{ack}よ", "{ack}あ", "はい、{ack}",
            "{ack}ました", "了解、{ack}", "{ack}た", "いいね、{ack}"
        ]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja"])
        ack = random.choice(ACK_WORDS.get(lang, ACK_WORDS["en"]))
        template = random.choice(templates.get(lang, templates["en"]))
        content = template.format(ack=ack)

        samples.append({
            "content": content,
            "should_store": False,
            "category": "ack",
            "lang": lang
        })
    return samples


def generate_smalltalk_samples(count=300):
    """Generate small talk messages with diverse templates."""
    samples = []

    templates = {
        "zh": [
            "{phrase}", "{phrase}呢", "{phrase}啊", "{phrase}呀", "确实，{phrase}",
            "对啊，{phrase}", "{phrase}，是吧", "{phrase}，怎么样", "{phrase}吗", "感觉{phrase}"
        ],
        "en": [
            "{phrase}", "{phrase}.", "{phrase}!", "yeah, {phrase}", "really, {phrase}",
            "for sure, {phrase}", "{phrase}, right?", "{phrase}, honestly", "i know, {phrase}", "{phrase}..."
        ],
        "ja": [
            "{phrase}", "{phrase}ね", "{phrase}わ", "{phrase}な", "{phrase}よ",
            "本当に{phrase}", "{phrase}よね", "同感、{phrase}", "{phrase}～"
        ]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja"])
        phrase = random.choice(SMALLTALK_TOPICS.get(lang, SMALLTALK_TOPICS["en"]))
        template = random.choice(templates.get(lang, templates["en"]))
        content = template.format(phrase=phrase)

        samples.append({
            "content": content,
            "should_store": False,
            "category": "smalltalk",
            "lang": lang
        })
    return samples


def generate_vague_samples(count=400):
    """Generate vague/undecided samples - CRITICAL category with diverse templates."""
    samples = []

    templates = {
        "zh": [
            "{phrase}", "{phrase}吧", "{phrase}，看着说", "可能{phrase}", "我觉得{phrase}",
            "{phrase}，不确定", "目前{phrase}", "先这样，{phrase}", "等等再说，{phrase}",
            "{phrase}，再看看", "这个事儿，{phrase}", "{phrase}，以后再议"
        ],
        "en": [
            "{phrase}", "{phrase}.", "maybe {phrase}", "probably {phrase}", "not sure yet",
            "{phrase} for now", "could be, {phrase}", "{phrase}, haven't decided", "let's see, {phrase}",
            "{phrase}, might change", "flexible on {phrase}", "{phrase}, up in the air"
        ],
        "ja": [
            "{phrase}", "{phrase}ね", "{phrase}かな", "{phrase}かも", "{phrase}知れない",
            "{phrase}、未定", "{phrase}かしら", "多分{phrase}", "{phrase}、後で"
        ]
    }

    vague_phrases = {
        "zh": ["框架还没定", "数据库未决", "方案在评估", "技术栈未敲定", "部署方式待定", "还在考虑中"],
        "en": ["framework undecided", "database choice TBD", "architecture pending", "tech stack not final", "deployment method TBD", "still evaluating"],
        "ja": ["フレームワーク未定", "データベース検討中", "アーキテクチャ未定", "実装方法未定"]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja"])
        phrase = random.choice(vague_phrases.get(lang, vague_phrases["en"]))
        template = random.choice(templates.get(lang, templates["en"]))
        content = template.format(phrase=phrase)

        samples.append({
            "content": content,
            "should_store": False,
            "category": "vague",
            "lang": lang
        })
    return samples


def generate_question_samples(count=300):
    """Generate question samples asking for information with diverse templates."""
    samples = []

    question_templates = {
        "zh": [
            "{marker}{topic}？", "{marker}{topic}啊？", "{marker}{topic}呢？",
            "请问{marker}{topic}？", "谁知道{marker}{topic}？", "{marker}{topic}吗？",
            "{marker}{topic}了？", "怎样才能{marker}{topic}？", "能否{marker}{topic}？"
        ],
        "en": [
            "{marker} {topic}?", "do you know {marker} {topic}?", "can you {marker} {topic}?",
            "{marker} {topic} please?", "anyone know {marker} {topic}?", "how about {marker} {topic}?",
            "what if {marker} {topic}?", "{marker} {topic}, right?", "does {marker} {topic}?"
        ],
        "ja": [
            "{marker}{topic}？", "{marker}{topic}ですか？", "{marker}{topic}かな？",
            "{marker}{topic}だろう？", "どうやって{marker}{topic}？", "{marker}{topic}ですね？"
        ]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja"])
        marker = random.choice(QUESTION_MARKERS.get(lang, QUESTION_MARKERS["en"]))
        topic = random.choice(QUESTION_TOPICS.get(lang, QUESTION_TOPICS["en"]))
        template = random.choice(question_templates.get(lang, question_templates["en"]))
        content = template.format(marker=marker, topic=topic)

        samples.append({
            "content": content,
            "should_store": False,
            "category": "question",
            "lang": lang
        })
    return samples


def generate_emotion_samples(count=250):
    """Generate emotional reactions with diverse templates."""
    samples = []

    templates = {
        "zh": [
            "{emotion}", "{emotion}！", "{emotion}呀", "{emotion}啊", "太{emotion}了",
            "{emotion}极了", "真是{emotion}", "哇，{emotion}", "{emotion}~", "{emotion}！！"
        ],
        "en": [
            "{emotion}", "{emotion}!", "{emotion}!!", "wow, {emotion}", "so {emotion}",
            "absolutely {emotion}", "that's {emotion}", "how {emotion}", "{emotion} stuff", "pretty {emotion}"
        ],
        "ja": [
            "{emotion}", "{emotion}ね", "{emotion}わ", "{emotion}よ", "{emotion}！",
            "めっちゃ{emotion}", "超{emotion}", "{emotion}な", "{emotion}ー"
        ]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja"])
        emotion = random.choice(EMOTION_WORDS.get(lang, ["amazing"]))
        template = random.choice(templates.get(lang, templates["en"]))
        content = template.format(emotion=emotion)

        # Sometimes add emoji
        if random.random() < 0.3:
            emoji = random.choice(EMOTION_WORDS["emoji"])
            content = f"{content} {emoji}"

        samples.append({
            "content": content,
            "should_store": False,
            "category": "emotion",
            "lang": lang
        })
    return samples


def generate_farewell_samples(count=200):
    """Generate farewell messages with diverse templates."""
    samples = []

    templates = {
        "zh": [
            "{farewell}", "{farewell}！", "{farewell}啊", "好的，{farewell}", "{farewell}，改天聊",
            "{farewell}，很高兴", "{farewell}各位", "{farewell}各位", "那就{farewell}吧"
        ],
        "en": [
            "{farewell}", "{farewell}!", "{farewell}.", "okay, {farewell}", "{farewell}, chat soon",
            "{farewell}, was great", "{farewell} everyone", "{farewell} folks", "well, {farewell}"
        ],
        "ja": [
            "{farewell}", "{farewell}ね", "{farewell}よ", "{farewell}あ", "じゃあ、{farewell}",
            "{farewell}ました", "では{farewell}", "{farewell}～", "{farewell}！"
        ]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja"])
        farewell = random.choice(FAREWELL_WORDS.get(lang, FAREWELL_WORDS["en"]))
        template = random.choice(templates.get(lang, templates["en"]))
        content = template.format(farewell=farewell)

        samples.append({
            "content": content,
            "should_store": False,
            "category": "farewell",
            "lang": lang
        })
    return samples


def generate_offtopic_samples(count=200):
    """Generate off-topic messages with diverse templates."""
    samples = []

    templates = {
        "zh": [
            "{phrase}", "{phrase}！", "{phrase}呀", "{phrase}呢", "我{phrase}",
            "{phrase}，特好看", "{phrase}，真不错", "最近{phrase}", "{phrase}，推荐"
        ],
        "en": [
            "{phrase}", "{phrase}!", "{phrase}.", "i {phrase}", "{phrase}, so cool",
            "{phrase}, really nice", "recently {phrase}", "{phrase}, would recommend", "you know, {phrase}"
        ],
        "ja": [
            "{phrase}", "{phrase}！", "{phrase}ね", "最近{phrase}", "{phrase}、楽しい",
            "{phrase}よ", "{phrase}かな", "{phrase}な", "昨日{phrase}"
        ]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja"])
        phrase = random.choice(OFFTOPIC_WORDS.get(lang, OFFTOPIC_WORDS["en"]))
        template = random.choice(templates.get(lang, templates["en"]))
        content = template.format(phrase=phrase)

        samples.append({
            "content": content,
            "should_store": False,
            "category": "offtopic",
            "lang": lang
        })
    return samples


def generate_filler_samples(count=150):
    """Generate filler/thinking sounds with diverse variations."""
    samples = []

    variations = {
        "zh": ["嗯", "呵呵", "233", "嘻嘻", "哈", "额", "呃", "呵", "...", "呐", "嗯嗯", "嗯呀"],
        "en": ["hmm", "well", "um", "uh", "like", "you know", "...", "haha", "lol", "hm", "uh oh"],
        "ja": ["えっと", "あのう", "うーん", "ふむ", "ふむふむ", "あ", "あぁ", "えー"],
        "symbol": ["...", ";;", "~~~", "…", ";;;", "~~~~"]
    }

    for _ in range(count):
        lang = random.choice(["zh", "en", "ja", "symbol"])
        filler = random.choice(variations[lang])

        # Add variations like repetition, punctuation, combinations
        variant_type = random.choice(["basic", "repeat", "repeat", "combo"])

        if variant_type == "repeat":
            filler = filler * random.randint(2, 4)
        elif variant_type == "combo":
            if lang == "zh":
                filler = filler + random.choice(["呀", "啊", "哈"])
            elif lang == "en":
                filler = filler + ", " + random.choice(["like", "you know", "right"])
            elif lang == "ja":
                filler = filler + random.choice(["ね", "な", "か"])

        samples.append({
            "content": filler,
            "should_store": False,
            "category": "filler",
            "lang": lang if lang != "symbol" else "mixed"
        })
    return samples


def main():
    # Load existing data
    data_path = Path("/Users/lilanfeng/project/tttt/mamba-memory/data/training_data.json")

    with open(data_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    print(f"Loaded {len(existing_data)} existing samples")

    # Track original counts
    original_store = sum(1 for item in existing_data if item.get('should_store') == True)
    original_discard = sum(1 for item in existing_data if item.get('should_store') == False)

    print(f"  STORE: {original_store}")
    print(f"  DISCARD: {original_discard}")

    # Generate new discard samples by category
    # Increased counts to target ~2500 after deduplication (roughly 60% survive dedup)
    print("\nGenerating new DISCARD samples...")
    new_samples = []

    new_samples.extend(generate_greeting_samples(500))
    print(f"  Generated 500 greeting samples")

    new_samples.extend(generate_ack_samples(650))
    print(f"  Generated 650 ack samples")

    new_samples.extend(generate_smalltalk_samples(500))
    print(f"  Generated 500 smalltalk samples")

    new_samples.extend(generate_vague_samples(650))
    print(f"  Generated 650 vague samples (CRITICAL)")

    new_samples.extend(generate_question_samples(500))
    print(f"  Generated 500 question samples")

    new_samples.extend(generate_emotion_samples(400))
    print(f"  Generated 400 emotion samples")

    new_samples.extend(generate_farewell_samples(350))
    print(f"  Generated 350 farewell samples")

    new_samples.extend(generate_offtopic_samples(330))
    print(f"  Generated 330 offtopic samples")

    new_samples.extend(generate_filler_samples(250))
    print(f"  Generated 250 filler samples")

    print(f"\nTotal new samples generated: {len(new_samples)} (target ~4130 to yield ~2500 after dedup)")

    # Create content set from existing data
    existing_contents = {item['content'] for item in existing_data}

    # Deduplicate new samples against existing and themselves
    print("\nDeduplicating...")
    unique_new_samples = []
    seen_new = set()

    for sample in new_samples:
        content = sample['content']
        if content not in existing_contents and content not in seen_new:
            unique_new_samples.append(sample)
            seen_new.add(content)

    print(f"After deduplication: {len(unique_new_samples)} unique new samples")
    duplicates_removed = len(new_samples) - len(unique_new_samples)
    print(f"  (Removed {duplicates_removed} duplicates)")

    # Merge all data
    merged_data = existing_data + unique_new_samples
    print(f"\nMerged total: {len(merged_data)} samples")

    # Save back to file
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {data_path}")

    # Print final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)

    final_store = sum(1 for item in merged_data if item.get('should_store') == True)
    final_discard = sum(1 for item in merged_data if item.get('should_store') == False)

    print(f"\nOriginal distribution:")
    print(f"  STORE:   {original_store:5d} ({100*original_store/(original_store+original_discard):.1f}%)")
    print(f"  DISCARD: {original_discard:5d} ({100*original_discard/(original_store+original_discard):.1f}%)")
    print(f"  Total:   {original_store+original_discard:5d}")

    print(f"\nNew distribution:")
    print(f"  STORE:   {final_store:5d} ({100*final_store/(final_store+final_discard):.1f}%)")
    print(f"  DISCARD: {final_discard:5d} ({100*final_discard/(final_store+final_discard):.1f}%)")
    print(f"  Total:   {final_store+final_discard:5d}")

    print(f"\nImprovement:")
    print(f"  STORE:   {final_store:5d} → {final_store:5d} (Δ {final_store-original_store:+d})")
    print(f"  DISCARD: {original_discard:5d} → {final_discard:5d} (Δ {final_discard-original_discard:+d})")

    # Category breakdown for new samples
    print(f"\nNew samples by category:")
    category_counts = defaultdict(int)
    for sample in unique_new_samples:
        category_counts[sample.get('category', 'unknown')] += 1

    for cat in sorted(category_counts.keys()):
        print(f"  {cat:12s}: {category_counts[cat]:4d}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
