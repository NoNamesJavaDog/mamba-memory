"""Gate v2 Benchmark — Data Augmentation + Cross-Validation

1. Expand 55 → 500+ labeled samples via template-based augmentation
2. Run 5-fold cross-validation to get REAL generalization accuracy
3. Compare rule engine vs neural gate v2
4. Analyze failure modes

Run: python examples/benchmark_gate_v2.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from mamba_memory.core.l2.gate import Gate, _importance_score
from mamba_memory.core.l2.learned_gate import LearnedGate, extract_rule_features
from mamba_memory.core.types import GateInput, MemorySlot

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


# ╔══════════════════════════════════════════════════════════════╗
# ║  Data Augmentation — 55 → 500+ samples                     ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class Sample:
    content: str
    should_store: bool
    category: str


def generate_augmented_dataset() -> list[Sample]:
    """Generate 500+ labeled samples via template-based augmentation."""
    samples: list[Sample] = []

    # ── SHOULD STORE ──────────────────────────────────────

    # Decisions (templates × variations)
    decision_templates_zh = [
        "我决定{verb}{tool}",
        "最终选择了{tool}",
        "确定采用{tool}来{purpose}",
        "我们改用{tool}替代{old_tool}",
        "{tool}更合适，决定切换过去",
        "经过讨论，选择{tool}作为{purpose}方案",
    ]
    decision_templates_en = [
        "I decided to use {tool} for {purpose}",
        "We're going with {tool} instead of {old_tool}",
        "Settled on {tool} for the {purpose}",
        "Let's adopt {tool} going forward",
        "Switching from {old_tool} to {tool}",
        "After evaluation, we chose {tool}",
    ]
    tools = ["PostgreSQL", "Redis", "Docker", "Kubernetes", "Nginx", "FastAPI",
             "React", "Vue", "TypeScript", "Terraform", "Ansible", "Grafana",
             "Prometheus", "ElasticSearch", "RabbitMQ", "Kafka", "MongoDB",
             "GraphQL", "gRPC", "Celery", "Airflow", "Jenkins", "ArgoCD"]
    old_tools = ["MySQL", "Memcached", "Docker Swarm", "Apache", "Flask",
                 "jQuery", "JavaScript", "Chef", "Puppet", "Nagios",
                 "Splunk", "ActiveMQ", "REST", "Cron", "Travis CI"]
    purposes = ["部署", "缓存", "监控", "日志", "CI/CD", "前端",
                "deployment", "caching", "monitoring", "logging", "CI/CD", "frontend"]
    verbs = ["使用", "采用", "选择", "启用", "部署"]

    for _ in range(40):
        t = random.choice(tools)
        ot = random.choice(old_tools)
        p = random.choice(purposes)
        v = random.choice(verbs)
        tmpl = random.choice(decision_templates_zh)
        samples.append(Sample(
            tmpl.format(tool=t, old_tool=ot, purpose=p, verb=v),
            True, "decision",
        ))
        tmpl_en = random.choice(decision_templates_en)
        samples.append(Sample(
            tmpl_en.format(tool=t, old_tool=ot, purpose=p),
            True, "decision",
        ))

    # Facts with structured data
    fact_templates = [
        "服务器IP是{ip}，端口{port}",
        "{service}配置：maxmemory {mem}MB, port {port}",
        "The {service} runs on port {port} with {config}",
        "连接串：{proto}://{user}:{pw}@{host}:{port}/{db}",
        "{service} version {ver} deployed on {env}",
        "API endpoint: https://{host}/api/v{ver}/{path}",
        "{service}内存限制{mem}MB，CPU限制{cpu}核",
        "Database backup at {time}, retention {days} days",
        "Rate limit: {num} req/{unit} for {tier} tier",
        "Log level: {level} in {env}, {level2} in dev",
        "域名{domain}解析到{ip}",
        "SSL证书有效期至{date}",
        "{service}集群{nodes}个节点，{replicas}副本",
    ]
    services = ["Redis", "PostgreSQL", "Nginx", "Kafka", "ES", "MongoDB",
                "MySQL", "RabbitMQ", "Consul", "Vault", "MinIO"]
    ips = ["192.168.1.100", "10.0.0.5", "172.16.0.1", "10.1.2.3"]
    ports = ["3306", "5432", "6379", "8080", "8443", "9200", "27017", "9090"]
    protos = ["postgres", "redis", "mysql", "amqp", "mongodb"]
    envs = ["production", "staging", "生产环境", "测试环境"]
    levels = ["WARN", "ERROR", "INFO", "DEBUG"]
    dates = ["2025-12-31", "2026-06-30", "2027-01-01"]

    for _ in range(50):
        tmpl = random.choice(fact_templates)
        samples.append(Sample(
            tmpl.format(
                ip=random.choice(ips), port=random.choice(ports),
                service=random.choice(services), mem=random.choice([128, 256, 512, 1024, 2048]),
                config=random.choice(["SSL", "TLS", "auth enabled", "cluster mode"]),
                proto=random.choice(protos), user="app", pw="secret",
                host=random.choice(["prod-db", "cache-01", "mq.internal"]),
                db=random.choice(["myapp", "analytics", "logs"]),
                ver=random.choice(["1.0", "2.1", "3.5", "16", "8.0"]),
                env=random.choice(envs), path=random.choice(["users", "orders", "health"]),
                cpu=random.choice([1, 2, 4, 8]), time=random.choice(["3am", "凌晨2点", "midnight"]),
                days=random.choice([7, 14, 30, 90]), num=random.choice([100, 500, 1000]),
                unit=random.choice(["min", "sec", "hour"]),
                tier=random.choice(["free", "pro", "enterprise"]),
                level=random.choice(levels), level2=random.choice(levels),
                domain=random.choice(["api.example.com", "app.mysite.cn"]),
                date=random.choice(dates), nodes=random.choice([3, 5, 7]),
                replicas=random.choice([2, 3]),
            ),
            True, "fact",
        ))

    # Preferences
    pref_templates_zh = [
        "我喜欢用{tool}，不喜欢{tool2}",
        "偏好{style}的代码风格",
        "测试框架倾向用{tool}",
        "习惯用{tool}做{task}",
        "推荐用{tool}而不是{tool2}",
    ]
    pref_templates_en = [
        "I prefer {tool} over {tool2}",
        "Always use {style} for consistency",
        "Recommend {tool} for {task}",
        "I like {tool}, hate {tool2}",
        "Fan of {style} approach",
    ]
    styles = ["tabs", "spaces", "camelCase", "snake_case", "functional",
              "OOP", "TDD", "BDD", "monorepo", "microservices"]
    tasks = ["formatting", "testing", "deployment", "debugging", "代码审查", "日志分析"]

    for _ in range(20):
        t1, t2 = random.sample(tools, 2)
        s = random.choice(styles)
        task = random.choice(tasks)
        tmpl = random.choice(pref_templates_zh + pref_templates_en)
        samples.append(Sample(
            tmpl.format(tool=t1, tool2=t2, style=s, task=task),
            True, "preference",
        ))

    # Corrections
    correction_templates = [
        "不对，{thing}应该是{correct}不是{wrong}",
        "错了，{thing}改成{correct}",
        "其实{thing}是{correct}",
        "Actually, the {thing} is {correct}, not {wrong}",
        "Wrong, it should be {correct} instead of {wrong}",
        "Wait, I was wrong. {thing} is {correct}",
        "Sorry, I meant {correct} for {thing}",
    ]
    things = ["端口", "port", "版本", "version", "密码", "路径", "path",
              "域名", "超时时间", "timeout", "内存限制", "memory limit"]

    for _ in range(20):
        thing = random.choice(things)
        correct = random.choice(["8443", "5433", "v3.2", "256MB", "/opt/app", "30s"])
        wrong = random.choice(["8080", "5432", "v2.0", "128MB", "/var/app", "60s"])
        tmpl = random.choice(correction_templates)
        samples.append(Sample(
            tmpl.format(thing=thing, correct=correct, wrong=wrong),
            True, "correction",
        ))

    # Explicit memory requests
    explicit_templates = [
        "记住：{fact}",
        "别忘了：{fact}",
        "重要：{fact}",
        "务必记住{fact}",
        "Remember: {fact}",
        "Don't forget: {fact}",
        "Important: {fact}",
        "Keep in mind: {fact}",
        "Note: {fact}",
        "Make sure to {fact}",
    ]
    facts = [
        "每周三备份数据库", "API密钥每90天轮换", "禁止使用root账户",
        "SSL证书下月到期", "版本号规则用semver",
        "database backup every Wednesday", "rotate API keys quarterly",
        "never use root in production", "SSL cert expires next month",
        "always run tests before deploying",
    ]

    for _ in range(20):
        tmpl = random.choice(explicit_templates)
        fact = random.choice(facts)
        samples.append(Sample(tmpl.format(fact=fact), True, "explicit"))

    # Action items
    action_templates = [
        "下一步{action}",
        "TODO: {action}",
        "FIXME: {action}",
        "接下来要{action}",
        "待办：{action}",
        "Next step: {action}",
        "Need to {action} by {deadline}",
        "Action item: {action}",
        "Must {action} before {event}",
    ]
    actions = [
        "配置监控告警", "添加限流", "升级数据库版本", "修复登录Bug",
        "add rate limiting", "set up monitoring", "upgrade PostgreSQL",
        "fix the auth bug", "write integration tests", "deploy to staging",
        "迁移到新集群", "优化查询性能", "添加缓存层",
    ]
    deadlines = ["Friday", "end of sprint", "next week", "明天", "本周五", "下个迭代"]
    events = ["release", "demo", "launch", "上线", "发布"]

    for _ in range(20):
        tmpl = random.choice(action_templates)
        action = random.choice(actions)
        samples.append(Sample(
            tmpl.format(action=action, deadline=random.choice(deadlines),
                        event=random.choice(events)),
            True, "action",
        ))

    # Commands (should store — they're actionable knowledge)
    commands = [
        "docker compose up -d --build",
        "kubectl apply -f deployment.yaml",
        "pip install -r requirements.txt",
        "npm run build && npm run deploy",
        "git rebase -i HEAD~5",
        "sudo systemctl restart nginx",
        "psql -h prod-db -U admin -d myapp",
        "redis-cli -h cache-01 INFO memory",
        "curl -X POST https://api.example.com/deploy",
        "terraform plan -var-file=prod.tfvars",
        "ansible-playbook -i inventory deploy.yml",
        "ssh -L 5432:prod-db:5432 bastion",
    ]
    for cmd in commands:
        samples.append(Sample(cmd, True, "command"))

    # ── SHOULD DISCARD ────────────────────────────────────

    # Greetings (multi-language)
    greetings = [
        "你好", "嗨", "hi", "hello", "hey", "Hey there", "Hi!",
        "早上好", "下午好", "晚上好", "Good morning", "Good afternoon",
        "Good evening", "What's up", "Yo", "嘿", "Hello!",
        "こんにちは", "おはよう", "こんばんは",
        "안녕하세요", "안녕",
        "Bonjour", "Hola", "Ciao",
    ]
    for g in greetings:
        samples.append(Sample(g, False, "greeting"))

    # Acknowledgments
    acks = [
        "好的", "ok", "OK", "嗯嗯", "嗯", "sure", "got it", "收到",
        "了解", "明白", "知道了", "好", "行", "可以", "没问题",
        "thanks", "谢谢", "thank you", "感谢", "多谢",
        "understood", "roger", "copy that", "noted", "okay",
        "Yes", "yeah", "yep", "对", "是的", "right", "correct",
        "No problem", "np", "不客气", "You're welcome",
        "alright", "fine", "good", "nice",
    ]
    for a in acks:
        samples.append(Sample(a, False, "ack"))

    # Small talk
    smalltalk = [
        "今天天气真好", "天气不错", "好热啊", "下雨了",
        "The weather is nice today", "It's so hot",
        "周末有什么计划", "你吃饭了吗", "最近怎么样",
        "Happy Friday!", "TGIF", "How's your day going?",
        "Did you watch the game?", "哈哈哈", "笑死我了",
        "lol", "haha", "哈哈", "呵呵", "233",
        "太有意思了", "真搞笑", "That's funny",
        "今天好累", "I'm so tired", "周一综合症",
        "Have a nice weekend", "加油", "Fighting!",
        "Coffee time ☕", "该吃午饭了", "Lunch time",
    ]
    for s in smalltalk:
        samples.append(Sample(s, False, "smalltalk"))

    # Farewells
    farewells = [
        "再见", "拜拜", "bye", "see you", "回见", "晚安",
        "Good night", "See you tomorrow", "明天见",
        "Take care", "下次聊", "走了", "先走了",
        "Goodbye", "Catch you later", "改天再聊",
    ]
    for f in farewells:
        samples.append(Sample(f, False, "farewell"))

    # Vague / thinking / deferral
    # Key: these often contain decision/action WORDS but the intent is to DEFER
    vague = [
        "嗯", "hmm", "...", "让我想想", "我想想",
        "Maybe", "可能吧", "不确定", "不太清楚",
        "I'm not sure", "I think so", "Perhaps",
        "Well...", "看情况", "再说吧", "有空再看",
        "回头说", "到时候再决定", "TBD", "待定",
        # Deferral with decision/action words (tricky cases)
        "以后再选择用什么框架", "下次再决定部署方案",
        "还没想好用什么数据库", "暂时不改配置",
        "先不部署，等等看", "回头再说这个事",
        "Let me think about which database to use",
        "Haven't decided on the framework yet",
        "Not sure yet, will decide later",
        "Maybe we'll switch to Redis, not decided",
        "I'll figure out the deployment later",
        "还没确定", "尚未决定", "先不管这个",
        "以后再处理", "这个不急", "下周再看",
        "Pending review", "To be discussed",
        "We'll see", "还在考虑中",
    ]
    for v in vague:
        samples.append(Sample(v, False, "vague"))

    # Questions without information (asking, not telling)
    questions = [
        "什么意思？", "怎么做？", "为什么？", "真的吗？",
        "What do you mean?", "How?", "Why?", "Really?",
        "你确定？", "Are you sure?", "能解释一下吗？",
        "Can you explain?", "有什么区别？", "What's the difference?",
        "这个是啥", "哪个好", "Which one is better?",
    ]
    for q in questions:
        samples.append(Sample(q, False, "question"))

    # Emotional / reactions
    emotions = [
        "太好了！", "厉害", "牛逼", "Amazing!", "Awesome!",
        "Nice!", "Cool!", "不错", "赞", "Great job!",
        "糟糕", "完了", "Oh no", "Damn", "糟了",
        "无语", "醉了", "I can't even", "bruh",
        "加油💪", "👍", "🎉", "❤️", "👌",
    ]
    for e in emotions:
        samples.append(Sample(e, False, "emotion"))

    random.shuffle(samples)
    return samples


# ╔══════════════════════════════════════════════════════════════╗
# ║  Cross-Validation                                           ║
# ╚══════════════════════════════════════════════════════════════╝

def k_fold_split(data: list[Sample], k: int = 5) -> list[tuple[list[Sample], list[Sample]]]:
    """Split data into k folds for cross-validation."""
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(data)
        test = data[start:end]
        train = data[:start] + data[end:]
        folds.append((train, test))
    return folds


def evaluate_rule_engine(test_data: list[Sample]) -> dict:
    """Evaluate rule engine on test data."""
    gate = Gate()
    empty_slots = [MemorySlot(id=i) for i in range(8)]

    tp = fp = tn = fn = 0
    for s in test_data:
        inp = GateInput(source="turn", content=s.content, entities=[])
        decision = gate.evaluate(inp, empty_slots)
        pred = decision.should_write

        if s.should_store and pred:
            tp += 1
        elif s.should_store and not pred:
            fn += 1
        elif not s.should_store and pred:
            fp += 1
        else:
            tn += 1

    total = len(test_data)
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate_neural_gate(train_data: list[Sample], test_data: list[Sample],
                         epochs: int = 300) -> dict:
    """Train neural gate on train_data, evaluate on test_data."""
    gate = LearnedGate()
    train_tuples = [(s.content, s.should_store) for s in train_data]
    gate.train(train_tuples, epochs=epochs)

    tp = fp = tn = fn = 0
    confidences_store = []
    confidences_discard = []
    errors: list[tuple[Sample, float]] = []

    for s in test_data:
        conf, pred = gate.predict(s.content)

        if s.should_store:
            confidences_store.append(conf)
        else:
            confidences_discard.append(conf)

        if s.should_store and pred:
            tp += 1
        elif s.should_store and not pred:
            fn += 1
            errors.append((s, conf))
        elif not s.should_store and pred:
            fp += 1
            errors.append((s, conf))
        else:
            tn += 1

    total = len(test_data)
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    avg_store_conf = sum(confidences_store) / max(len(confidences_store), 1)
    avg_disc_conf = sum(confidences_discard) / max(len(confidences_discard), 1)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "avg_store_conf": avg_store_conf, "avg_disc_conf": avg_disc_conf,
            "errors": errors}


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    random.seed(42)
    np.random.seed(42)

    print(f"\n{'█' * 60}")
    print(f"  Gate v2 Benchmark — Data Augmentation + Cross-Validation")
    print(f"{'█' * 60}")

    # ── Phase 1: Generate data ──────────────────────────────
    print(f"\n{BOLD}Phase 1: Data Augmentation{RESET}")
    dataset = generate_augmented_dataset()

    # Count by category and label
    store_count = sum(1 for s in dataset if s.should_store)
    discard_count = sum(1 for s in dataset if not s.should_store)
    categories: dict[str, int] = {}
    for s in dataset:
        categories[s.category] = categories.get(s.category, 0) + 1

    print(f"  Total samples: {len(dataset)}")
    print(f"  Should store:  {store_count}")
    print(f"  Should discard: {discard_count}")
    print(f"  Balance ratio: {store_count/len(dataset):.1%} store / {discard_count/len(dataset):.1%} discard")
    print(f"  Categories:")
    for cat in sorted(categories.keys()):
        print(f"    {cat:<12} {categories[cat]:>4}")

    # ── Phase 2: 5-fold cross-validation ─────────────────────
    print(f"\n{BOLD}Phase 2: 5-Fold Cross-Validation{RESET}")
    folds = k_fold_split(dataset, k=5)

    rule_results = []
    neural_results = []

    for i, (train, test) in enumerate(folds):
        # Rule engine (no training needed)
        r_rule = evaluate_rule_engine(test)
        rule_results.append(r_rule)

        # Neural gate
        r_neural = evaluate_neural_gate(train, test, epochs=300)
        neural_results.append(r_neural)

        mark_r = PASS if r_rule["accuracy"] >= 0.9 else FAIL
        mark_n = PASS if r_neural["accuracy"] >= 0.9 else FAIL
        print(f"  Fold {i+1}: Rule {mark_r} acc={r_rule['accuracy']:.1%} f1={r_rule['f1']:.3f}"
              f"  |  Neural {mark_n} acc={r_neural['accuracy']:.1%} f1={r_neural['f1']:.3f}"
              f"  (train={len(train)}, test={len(test)})")

    # ── Phase 3: Aggregate results ─────────────────────────
    print(f"\n{BOLD}Phase 3: Aggregate Results (mean ± std across 5 folds){RESET}")

    def agg(results: list[dict], key: str) -> tuple[float, float]:
        vals = [r[key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    print(f"\n  {'metric':<12} │ {'Rule Engine':>20} │ {'Neural Gate v2':>20}")
    print(f"  {'─'*12}─┼─{'─'*20}─┼─{'─'*20}")

    for metric in ["accuracy", "precision", "recall", "f1"]:
        r_mean, r_std = agg(rule_results, metric)
        n_mean, n_std = agg(neural_results, metric)
        r_str = f"{r_mean:.1%} ± {r_std:.1%}"
        n_str = f"{n_mean:.1%} ± {n_std:.1%}"
        winner = "  ◀" if n_mean > r_mean else ""
        print(f"  {metric:<12} │ {r_str:>20} │ {n_str:>20}{winner}")

    # Confidence separation
    n_store_confs = [r["avg_store_conf"] for r in neural_results]
    n_disc_confs = [r["avg_disc_conf"] for r in neural_results]
    print(f"\n  Neural gate confidence:")
    print(f"    Should-store avg:   {np.mean(n_store_confs):.3f} ± {np.std(n_store_confs):.3f}")
    print(f"    Should-discard avg: {np.mean(n_disc_confs):.3f} ± {np.std(n_disc_confs):.3f}")
    gap = np.mean(n_store_confs) - np.mean(n_disc_confs)
    print(f"    Separation gap:     {gap:.3f}")

    # ── Phase 4: Error analysis ──────────────────────────────
    print(f"\n{BOLD}Phase 4: Error Analysis (last fold){RESET}")

    last_neural = neural_results[-1]
    if last_neural.get("errors"):
        print(f"\n  Neural gate errors ({len(last_neural['errors'])} total):")
        for sample, conf in last_neural["errors"][:15]:
            label = "STORE" if sample.should_store else "DISC"
            pred = "STORE" if conf >= 0.5 else "DISC"
            print(f"    {FAIL} [{sample.category:<10}] label={label} pred={pred} "
                  f"conf={conf:.3f} | {sample.content[:50]}")
    else:
        print(f"  {PASS} Neural gate: zero errors on last fold!")

    # Rule engine errors on last fold
    gate = Gate()
    empty_slots = [MemorySlot(id=i) for i in range(8)]
    rule_errors = []
    _, last_test = folds[-1]
    for s in last_test:
        inp = GateInput(source="turn", content=s.content, entities=[])
        decision = gate.evaluate(inp, empty_slots)
        if decision.should_write != s.should_store:
            rule_errors.append(s)

    if rule_errors:
        print(f"\n  Rule engine errors ({len(rule_errors)} total):")
        for s in rule_errors[:15]:
            label = "STORE" if s.should_store else "DISC"
            score = _importance_score(s.content)
            print(f"    {FAIL} [{s.category:<10}] label={label} "
                  f"score={score:.3f} | {s.content[:50]}")

    # ── Phase 5: Per-category breakdown ────────────────────
    print(f"\n{BOLD}Phase 5: Per-Category Accuracy (last fold, neural gate){RESET}")

    last_train, last_test = folds[-1]
    gate_final = LearnedGate()
    gate_final.train([(s.content, s.should_store) for s in last_train], epochs=300)

    cat_stats: dict[str, dict[str, int]] = {}
    for s in last_test:
        conf, pred = gate_final.predict(s.content)
        cat = s.category
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if pred == s.should_store:
            cat_stats[cat]["correct"] += 1

    print(f"\n  {'category':<12} │ {'total':>5} │ {'correct':>7} │ {'accuracy':>8}")
    print(f"  {'─'*12}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*8}")
    for cat in sorted(cat_stats.keys()):
        st = cat_stats[cat]
        acc = st["correct"] / st["total"] if st["total"] > 0 else 0
        mark = PASS if acc >= 0.9 else (FAIL if acc < 0.7 else "⚠")
        print(f"  {mark} {cat:<10} │ {st['total']:>5} │ {st['correct']:>7} │ {acc:>7.0%}")

    # Summary
    r_acc_mean, r_acc_std = agg(rule_results, "accuracy")
    n_acc_mean, n_acc_std = agg(neural_results, "accuracy")
    n_f1_mean, _ = agg(neural_results, "f1")

    print(f"\n{'█' * 60}")
    print(f"  Summary")
    print(f"{'█' * 60}")
    print(f"  Dataset:          {len(dataset)} samples ({store_count} store / {discard_count} discard)")
    print(f"  Rule engine:      {r_acc_mean:.1%} ± {r_acc_std:.1%} accuracy (5-fold CV)")
    print(f"  Neural gate v2:   {n_acc_mean:.1%} ± {n_acc_std:.1%} accuracy (5-fold CV)")
    print(f"  Neural F1:        {n_f1_mean:.3f}")
    print(f"  Confidence gap:   {gap:.3f}")
    print()


if __name__ == "__main__":
    main()
