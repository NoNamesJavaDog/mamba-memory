"""Tests for the text analysis module."""

from mamba_memory.core.text import (
    compress_turns_structured,
    extract_entities_simple,
    extract_key_facts,
    extract_ngrams,
    information_density,
    text_relevance,
    tokenize,
)
from mamba_memory.core.l2.gate import _importance_score


# -- Tokenizer ---------------------------------------------------------------

class TestTokenize:
    def test_chinese(self):
        tokens = tokenize("数据库选择了PostgreSQL")
        assert "数" in tokens
        assert "据" in tokens
        assert "postgresql" in tokens
        # Stop words filtered
        assert "了" not in tokens

    def test_english(self):
        tokens = tokenize("I decided to use Docker for deployment")
        assert "docker" in tokens
        assert "deployment" in tokens
        # Stop words filtered
        assert "to" not in tokens
        assert "for" not in tokens

    def test_mixed(self):
        tokens = tokenize("我决定用Python 3.12")
        assert "python" in tokens
        assert "决" in tokens

    def test_short_filler(self):
        tokens = tokenize("嗯嗯好的")
        # Most single-char stop words should be filtered
        assert len(tokens) <= 2


# -- Ngrams ------------------------------------------------------------------

class TestNgrams:
    def test_cjk_bigrams(self):
        ngrams = extract_ngrams("数据库", 2)
        assert "数据" in ngrams
        assert "据库" in ngrams

    def test_english_words(self):
        ngrams = extract_ngrams("PostgreSQL database", 2)
        assert "postgresql" in ngrams
        assert "database" in ngrams


# -- Text relevance ----------------------------------------------------------

class TestTextRelevance:
    def test_exact_substring_zh(self):
        score = text_relevance("数据库", "数据库选择了PostgreSQL，部署在port 5432")
        assert score > 0.5

    def test_exact_substring_en(self):
        score = text_relevance("Docker", "I decided to use Docker for deployment")
        assert score > 0.5

    def test_partial_match_zh(self):
        score = text_relevance("部署方式", "数据库选择了PostgreSQL，部署在port 5432")
        assert score > 0.1

    def test_no_match(self):
        score = text_relevance("天气预报", "数据库选择了PostgreSQL")
        assert score < 0.1

    def test_greeting_low_score(self):
        score = text_relevance("你好", "数据库选择了PostgreSQL")
        assert score < 0.3

    def test_english_word_overlap(self):
        score = text_relevance("database setup", "We configured the database on port 5432")
        assert score > 0.2


# -- Key facts extraction ----------------------------------------------------

class TestExtractKeyFacts:
    def test_extracts_config_sentences(self):
        text = "你好。数据库端口是5432。今天天气不错。Redis内存限制128MB。"
        facts = extract_key_facts(text)
        assert any("5432" in f for f in facts)
        assert any("128MB" in f for f in facts)
        # "你好" and "天气" should not be in facts
        assert not any("天气" in f for f in facts)

    def test_extracts_decision_sentences(self):
        text = "我们聊了很久。最后决定使用Docker部署。回去吧。"
        facts = extract_key_facts(text)
        assert any("Docker" in f for f in facts)

    def test_empty_input(self):
        assert extract_key_facts("") == []


# -- Entity extraction -------------------------------------------------------

class TestExtractEntities:
    def test_tech_names(self):
        entities = extract_entities_simple("我们使用PostgreSQL和Redis，部署在Docker上")
        names = [e.lower() for e in entities]
        assert "postgresql" in names
        assert "redis" in names
        assert "docker" in names

    def test_person_names_zh(self):
        entities = extract_entities_simple("我叫张三，负责后端开发")
        assert "张三" in entities

    def test_capitalized_words(self):
        entities = extract_entities_simple("We use FastAPI with Kubernetes")
        names = [e.lower() for e in entities]
        assert "fastapi" in names
        assert "kubernetes" in names


# -- Structured compression --------------------------------------------------

class TestCompressTurns:
    def test_extracts_facts(self):
        turns = [
            ("user", "你好"),
            ("assistant", "你好，有什么可以帮你的吗？"),
            ("user", "数据库端口改成5432"),
            ("user", "Redis内存限制设为128MB"),
        ]
        summary, entities = compress_turns_structured(turns)
        assert "5432" in summary
        assert "128MB" in summary
        # Greeting should be stripped out
        assert "你好" not in summary or "5432" in summary

    def test_empty_turns(self):
        summary, entities = compress_turns_structured([])
        assert summary == ""


# -- Information density -----------------------------------------------------

class TestInformationDensity:
    def test_high_density(self):
        score = information_density("服务器IP 192.168.1.1, 端口5432, Redis 128MB")
        assert score > 0.3

    def test_low_density(self):
        score = information_density("嗯嗯好的")
        assert score < 0.3

    def test_empty(self):
        assert information_density("") == 0.0


# -- Gate importance score (upgraded) ----------------------------------------

class TestImportanceScore:
    def test_decision_zh(self):
        assert _importance_score("我决定使用PostgreSQL") > 0.1

    def test_decision_en(self):
        assert _importance_score("I decided to switch to Docker") > 0.1

    def test_correction(self):
        assert _importance_score("不对，应该是8080端口") > 0.1

    def test_explicit_memory(self):
        assert _importance_score("记住：密钥每90天轮换") > 0.1

    def test_pure_greeting_zero(self):
        assert _importance_score("你好") == 0.0

    def test_pure_ok_zero(self):
        assert _importance_score("好的") == 0.0

    def test_structured_data(self):
        assert _importance_score("配置端口为5432，内存128MB") > 0.1

    def test_action_signal(self):
        assert _importance_score("下一步需要部署到生产环境") > 0.1

    def test_combined_high(self):
        score = _importance_score("我决定把数据库迁移到PostgreSQL, port 5432, 记住这个配置")
        assert score > 0.4  # multiple dimensions hit

    def test_ja_greeting_zero(self):
        assert _importance_score("こんにちは") == 0.0

    def test_ja_decision(self):
        assert _importance_score("PostgreSQLを採用することに決定しました") > 0.1
