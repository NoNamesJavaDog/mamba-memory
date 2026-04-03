#!/usr/bin/env python3
"""
Generate bulk training data for memory gate classifier.
Creates ~4500 samples using template-based augmentation.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Language-specific templates for SHOULD STORE categories
DECISION_TEMPLATES = {
    'zh': [
        "我决定用{tech1}代替{tech2}",
        "选择{framework}做{use_case}",
        "{tech1}太慢了，改用{tech2}",
        "项目改成{tech1}架构",
        "技术栈从{tech1}迁移到{tech2}",
        "决定采用{tech1}而不是{tech2}",
        "经过评估，选定{tech1}做{component}",
        "{tech1}性能不够，用{tech2}重写",
        "最终决定用{framework}做前端",
        "后端框架确定是{framework}",
    ],
    'en': [
        "We decided to switch from {tech1} to {tech2}",
        "Chose {framework} for {use_case}",
        "{tech1} is too slow, migrating to {tech2}",
        "Architecture changed to {tech1}",
        "Technology stack: replaced {tech1} with {tech2}",
        "Final decision: use {framework} instead of {tech1}",
        "After evaluation, we selected {tech1} for {component}",
        "Switching from {tech1} due to performance issues",
        "Frontend framework is {framework}",
        "Using {tech1} for {component}",
    ],
    'ja': [
        "{tech1}から{tech2}に変更することに決めた",
        "{use_case}には{framework}を選んだ",
        "{tech1}は遅いので{tech2}に置き換える",
        "アーキテクチャを{tech1}に変更した",
        "技術スタック：{tech1}から{tech2}へ移行",
        "{framework}を採用することにした",
        "評価の結果、{tech1}を選択した",
        "{tech1}のパフォーマンスが悪いので{tech2}に切り替える",
        "フロントエンドは{framework}にする",
        "{component}には{tech1}を使う",
    ],
    'ko': [
        "{tech1}에서 {tech2}로 변경하기로 결정했다",
        "{use_case}에는 {framework}를 선택했다",
        "{tech1}은 너무 느려서 {tech2}로 마이그레이션한다",
        "아키텍처를 {tech1}로 변경했다",
        "기술 스택: {tech1}에서 {tech2}로 교체",
        "{framework}을 채택하기로 결정했다",
        "검토 결과, {tech1}을 선택했다",
        "{tech1}의 성능이 부족해서 {tech2}로 바꾼다",
        "프론트엔드는 {framework}으로 정한다",
        "{component}에는 {tech1}을 사용한다",
    ],
    'fr': [
        "Nous avons décidé de remplacer {tech1} par {tech2}",
        "Choisi {framework} pour {use_case}",
        "{tech1} est trop lent, passage à {tech2}",
        "Architecture modifiée vers {tech1}",
        "Stack technologique: {tech1} remplacé par {tech2}",
        "Décision finale: utiliser {framework} au lieu de {tech1}",
        "Après évaluation, {tech1} sélectionné pour {component}",
        "Migration de {tech1} vers {tech2} pour les performances",
        "Framework frontend: {framework}",
        "Utilisation de {tech1} pour {component}",
    ],
    'de': [
        "Wir haben entschieden, {tech1} durch {tech2} zu ersetzen",
        "{framework} für {use_case} gewählt",
        "{tech1} ist zu langsam, Wechsel zu {tech2}",
        "Architektur auf {tech1} geändert",
        "Tech-Stack: {tech1} durch {tech2} ersetzt",
        "Endgültige Entscheidung: {framework} statt {tech1}",
        "Nach Bewertung {tech1} für {component} ausgewählt",
        "Migration von {tech1} zu {tech2} erforderlich",
        "Frontend-Framework: {framework}",
        "{component} nutzt {tech1}",
    ],
    'es': [
        "Decidimos cambiar de {tech1} a {tech2}",
        "Elegimos {framework} para {use_case}",
        "{tech1} es demasiado lento, migración a {tech2}",
        "Arquitectura cambiada a {tech1}",
        "Stack tecnológico: {tech1} reemplazado con {tech2}",
        "Decisión final: usar {framework} en lugar de {tech1}",
        "Tras evaluar, {tech1} seleccionado para {component}",
        "Cambio de {tech1} por problemas de rendimiento",
        "Framework frontend: {framework}",
        "Uso de {tech1} para {component}",
    ],
}

FACT_TEMPLATES = {
    'zh': [
        "服务器IP: {ip}",
        "数据库连接字符串: {connection_string}",
        "Redis缓存地址: {ip}:{port}",
        "API端口: {port}",
        "PostgreSQL版本: {version}",
        "配置文件位置: {file_path}",
        "Cron任务: {cron_schedule}",
        "内存限制: {memory_limit}",
        "请求超时: {timeout}",
        "DNS解析: {domain} -> {ip}",
        "SSL证书有效期: {date}",
        "数据库用户: {username}@{host}:{port}",
    ],
    'en': [
        "Server IP: {ip}",
        "Database connection: {connection_string}",
        "Redis endpoint: {ip}:{port}",
        "API port: {port}",
        "PostgreSQL version: {version}",
        "Config file: {file_path}",
        "Cron job: {cron_schedule}",
        "Memory limit: {memory_limit}",
        "Timeout: {timeout}",
        "DNS record: {domain} -> {ip}",
        "Certificate expires: {date}",
        "DB user: {username}@{host}:{port}",
    ],
    'ja': [
        "サーバーIP: {ip}",
        "データベース接続: {connection_string}",
        "Redisエンドポイント: {ip}:{port}",
        "APIポート: {port}",
        "PostgreSQLバージョン: {version}",
        "設定ファイル: {file_path}",
        "Cronジョブ: {cron_schedule}",
        "メモリ制限: {memory_limit}",
        "タイムアウト: {timeout}",
        "DNSレコード: {domain} -> {ip}",
        "証明書の有効期限: {date}",
        "DBユーザー: {username}@{host}:{port}",
    ],
    'ko': [
        "서버 IP: {ip}",
        "데이터베이스 연결: {connection_string}",
        "Redis 엔드포인트: {ip}:{port}",
        "API 포트: {port}",
        "PostgreSQL 버전: {version}",
        "설정 파일: {file_path}",
        "Cron 작업: {cron_schedule}",
        "메모리 제한: {memory_limit}",
        "타임아웃: {timeout}",
        "DNS 레코드: {domain} -> {ip}",
        "인증서 만료일: {date}",
        "DB 사용자: {username}@{host}:{port}",
    ],
}

PREFERENCE_TEMPLATES = {
    'zh': [
        "代码风格：都用驼峰命名，不用下划线",
        "喜欢用{tool}做日志，比{tool2}好用",
        "配置管理用YAML不用JSON",
        "代码审核一定要用{tool}",
        "更喜欢{pattern}的开发模式",
        "项目里统一用{style}的命名约定",
        "工作流用{workflow}，提高效率",
        "代码格式化工具使用{tool}",
        "偏好{approach}而不是{approach2}",
    ],
    'en': [
        "Code style: camelCase everywhere, no underscores",
        "Prefer {tool} for logging over {tool2}",
        "Use YAML for config, not JSON",
        "Code reviews must use {tool}",
        "Prefer {pattern} development approach",
        "Standardized naming convention: {style}",
        "Workflow tool: {workflow}",
        "Code formatter: {tool}",
        "Prefer {approach} over {approach2}",
    ],
    'ja': [
        "コードスタイル：キャメルケース推奨",
        "{tool}でのログ出力を好む（{tool2}より）",
        "設定はYAMLで、JSONではない",
        "コードレビューは{tool}必須",
        "{pattern}開発が好きだ",
        "命名規則：{style}統一",
        "ワークフロー：{workflow}",
        "コードフォーマッター：{tool}",
        "{approach}を{approach2}より好む",
    ],
    'ko': [
        "코드 스타일: 카멜케이스 사용",
        "{tool}로 로깅하는 것을 선호한다",
        "설정은 YAML, JSON 아님",
        "코드 리뷰는 {tool} 필수",
        "{pattern} 개발을 선호한다",
        "명명 규칙: {style} 통일",
        "워크플로우 도구: {workflow}",
        "코드 포매터: {tool}",
        "{approach}를 {approach2}보다 좋아한다",
    ],
}

CORRECTION_TEMPLATES = {
    'zh': [
        "纠正：Redis端口应该是{port}不是6380",
        "错了，密码是{password}而不是那个",
        "更新：版本应该是{version}，之前说的不对",
        "修正：连接字符串应该是{connection_string}",
        "之前记错了，正确的IP是{ip}",
        "改正：Kubernetes版本是{version}",
        "纠正前面的说法，端口确实是{port}",
    ],
    'en': [
        "Correction: Redis port should be {port}, not 6380",
        "Wrong, password is {password}, not that one",
        "Update: version should be {version}",
        "Fix: connection string is {connection_string}",
        "I was wrong, correct IP is {ip}",
        "Correction: Kubernetes version is {version}",
        "Actually, port is {port}",
    ],
    'ja': [
        "訂正：Redisのポートは{port}、6380ではない",
        "間違い、パスワードは{password}",
        "更新：バージョンは{version}にすべき",
        "修正：接続文字列は{connection_string}",
        "間違えました、正しいIPは{ip}",
        "訂正：Kubernetesバージョンは{version}",
        "実は、ポートは{port}です",
    ],
    'ko': [
        "수정: Redis 포트는 {port}, 6380이 아니다",
        "틀렸다, 비밀번호는 {password}",
        "업데이트: 버전은 {version}이어야 한다",
        "수정: 연결 문자열은 {connection_string}",
        "틀렸다, 올바른 IP는 {ip}",
        "수정: Kubernetes 버전은 {version}",
        "사실 포트는 {port}이다",
    ],
}

EXPLICIT_TEMPLATES = {
    'zh': [
        "记住：{content}",
        "一定要记住{content}",
        "不要忘记{content}",
        "记下来：{content}",
        "这点很重要，记住{content}",
        "别忘了{content}",
    ],
    'en': [
        "Remember: {content}",
        "Don't forget: {content}",
        "Important to remember: {content}",
        "Note: {content}",
        "Make sure to remember {content}",
    ],
    'ja': [
        "覚えて：{content}",
        "忘れずに：{content}",
        "大事：{content}を覚えておく",
        "記録：{content}",
    ],
    'ko': [
        "기억해: {content}",
        "잊지 말고: {content}",
        "중요: {content}",
        "기록: {content}",
    ],
}

ACTION_TEMPLATES = {
    'zh': [
        "TODO: {task}",
        "下一步：{task}",
        "明天要{task}",
        "Sprint目标：{task}",
        "这个迭代需要{task}",
        "截止日期：{date}之前要完成{task}",
        "Action Item: {task}",
        "待办：{task}",
    ],
    'en': [
        "TODO: {task}",
        "Next step: {task}",
        "Need to {task}",
        "Sprint goal: {task}",
        "Deadline: {task} by {date}",
        "Action item: {task}",
        "Follow up: {task}",
        "Must complete {task}",
    ],
    'ja': [
        "TODO: {task}",
        "次のステップ：{task}",
        "明日{task}",
        "スプリント目標：{task}",
        "期限：{date}までに{task}",
        "アクションアイテム：{task}",
        "確認事項：{task}",
    ],
    'ko': [
        "TODO: {task}",
        "다음 단계: {task}",
        "해야 할 일: {task}",
        "스프린트 목표: {task}",
        "마감일: {date}까지 {task}",
        "액션 아이템: {task}",
        "확인 필요: {task}",
    ],
}

COMMAND_TEMPLATES = {
    'zh': [
        "docker run -d {image}",
        "kubectl apply -f {file}",
        "git commit -m '{message}'",
        "psql -h {host} -U {user} {db}",
        "terraform apply {plan}",
        "ansible-playbook {playbook}",
        "aws s3 cp {file} s3://{bucket}/",
        "npm install {package}",
        "pip install {package}",
        "curl -X {method} {url}",
    ],
    'en': [
        "docker run -d {image}",
        "kubectl apply -f {file}",
        "git commit -m '{message}'",
        "psql -h {host} -U {user} {db}",
        "terraform apply {plan}",
        "ansible-playbook {playbook}",
        "aws s3 cp {file} s3://{bucket}/",
        "npm install {package}",
        "pip install {package}",
        "curl -X {method} {url}",
    ],
}

ERROR_LOG_TEMPLATES = {
    'zh': [
        "错误：{error_msg} at {file}:{line}",
        "堆栈跟踪：{stack_trace}",
        "原因分析：{root_cause}",
        "Exception: {error_type}: {error_msg}",
        "根本原因是{root_cause}导致的",
    ],
    'en': [
        "Error: {error_msg} at {file}:{line}",
        "Stack trace: {stack_trace}",
        "Root cause: {root_cause}",
        "Exception: {error_type}: {error_msg}",
        "Caused by {root_cause}",
    ],
    'ja': [
        "エラー：{error_msg} at {file}:{line}",
        "スタックトレース：{stack_trace}",
        "根本原因：{root_cause}",
        "Exception: {error_type}: {error_msg}",
        "原因は{root_cause}",
    ],
    'ko': [
        "오류: {error_msg} at {file}:{line}",
        "스택 추적: {stack_trace}",
        "근본 원인: {root_cause}",
        "Exception: {error_type}: {error_msg}",
        "{root_cause}로 인한 오류",
    ],
}

MEETING_NOTE_TEMPLATES = {
    'zh': [
        "会议纪要：决定了{decision}",
        "{person}负责{task}，截止{date}",
        "会上同意{decision}",
        "下次会议：{date}，讨论{topic}",
        "Action: {person} will do {task}",
    ],
    'en': [
        "Meeting notes: Decided on {decision}",
        "{person} will handle {task} by {date}",
        "Agreed to {decision}",
        "Next meeting: {date}, discuss {topic}",
        "Action: {person} will do {task}",
    ],
    'ja': [
        "会議記録：{decision}に決定した",
        "{person}が{task}を担当、期限{date}",
        "{decision}に同意した",
        "次回会議：{date}、{topic}を議論",
        "アクション：{person}が{task}を行う",
    ],
    'ko': [
        "회의 기록: {decision}으로 결정됨",
        "{person}이 {task}를 담당, 기한 {date}",
        "{decision}에 동의함",
        "다음 회의: {date}, {topic} 논의",
        "액션: {person}이 {task}를 수행",
    ],
}

ARCHITECTURE_TEMPLATES = {
    'zh': [
        "架构：{arch_desc}",
        "数据流：{data_flow}",
        "扩展策略：{scaling_strategy}",
        "设计模式：使用{pattern}",
        "系统间通信：{communication}",
        "故障转移方案：{failover}",
    ],
    'en': [
        "Architecture: {arch_desc}",
        "Data flow: {data_flow}",
        "Scaling strategy: {scaling_strategy}",
        "Design pattern: {pattern}",
        "Inter-service communication: {communication}",
        "Failover: {failover}",
    ],
    'ja': [
        "アーキテクチャ：{arch_desc}",
        "データフロー：{data_flow}",
        "スケーリング戦略：{scaling_strategy}",
        "デザインパターン：{pattern}",
        "サービス間通信：{communication}",
        "フェイルオーバー：{failover}",
    ],
    'ko': [
        "아키텍처: {arch_desc}",
        "데이터 흐름: {data_flow}",
        "스케일링 전략: {scaling_strategy}",
        "디자인 패턴: {pattern}",
        "서비스 간 통신: {communication}",
        "페일오버: {failover}",
    ],
}

REQUIREMENT_TEMPLATES = {
    'zh': [
        "需求：{requirement}",
        "验收标准：{acceptance_criteria}",
        "用户故事：{user_story}",
        "功能需求：{feature_req}",
        "非功能需求：{non_functional}",
    ],
    'en': [
        "Requirement: {requirement}",
        "Acceptance criteria: {acceptance_criteria}",
        "User story: {user_story}",
        "Feature: {feature_req}",
        "Non-functional: {non_functional}",
    ],
    'ja': [
        "要件：{requirement}",
        "受け入れ基準：{acceptance_criteria}",
        "ユーザーストーリー：{user_story}",
        "機能要件：{feature_req}",
        "非機能要件：{non_functional}",
    ],
    'ko': [
        "요구사항: {requirement}",
        "승인 기준: {acceptance_criteria}",
        "사용자 스토리: {user_story}",
        "기능 요구사항: {feature_req}",
        "비기능 요구사항: {non_functional}",
    ],
}

# Language-specific templates for SHOULD DISCARD categories
GREETING_TEMPLATES = {
    'zh': ["你好", "你好吗", "早上好", "晚上好", "嗨"],
    'en': ["Hello", "Hi there", "Good morning", "Hi!", "Hey"],
    'ja': ["こんにちは", "おはよう", "こんばんは", "やあ"],
    'ko': ["안녕하세요", "안녕", "좋은 아침", "저녁입니다"],
    'fr': ["Bonjour", "Salut", "Bonsoir", "Coucou"],
    'de': ["Hallo", "Guten Morgen", "Guten Abend", "Hallo!"],
    'es': ["Hola", "Buenos días", "Buenas noches", "¡Hola!"],
    'pt': ["Olá", "Bom dia", "Boa noite", "Oi"],
    'ru': ["Привет", "Здравствуй", "Доброе утро", "Добрый вечер"],
    'ar': ["السلام عليكم", "مرحبا", "صباح الخير"],
}

ACK_TEMPLATES = {
    'zh': [
        "好的", "嗯嗯", "了解", "明白了", "OK", "收到", "可以", "同意",
        "好", "是的", "没问题", "确认", "同意这个方案", "一致同意",
    ],
    'en': [
        "OK", "Sure", "Got it", "Understood", "Alright", "No problem",
        "Yes", "Confirmed", "I agree", "Sounds good", "That works",
        "Fine with me", "Great",
    ],
    'ja': [
        "はい", "了解", "わかりました", "いいです", "オッケー", "承知",
        "そうですね", "同意します", "問題ありません", "了承します",
    ],
    'ko': [
        "네", "알겠습니다", "좋습니다", "동의합니다", "괜찮습니다", "확인",
        "승인", "괜찮아요", "알겠어", "그러지",
    ],
}

SMALLTALK_TEMPLATES = {
    'zh': [
        "今天天气真好", "周末去哪玩", "太累了", "周末快乐", "今天吃什么",
        "最近在看什么电影", "你喜欢吃什么", "天气不好", "睡眠不足",
        "放假了真开心", "工作太忙", "最近学什么新东西",
    ],
    'en': [
        "Nice weather today", "Where are you going this weekend",
        "So tired", "Happy weekend", "What should we eat",
        "What movie are you watching", "Do you like pizza",
        "Bad weather today", "Haven't slept well", "Holiday is great",
        "Work is so busy", "Learning something new",
    ],
    'ja': [
        "今日はいい天気", "週末どこに行く", "疲れた", "週末楽しい",
        "今日何を食べようかな", "最近どんな映画を見てる",
        "ピザが好きです", "天気が悪い", "睡眠不足", "休日は楽しい",
        "仕事が忙しい", "新しいことを勉強中",
    ],
}

VAGUE_TEMPLATES = {
    'zh': [
        "以后再决定", "先这样，可能后来会改", "到时候再说",
        "也许我们以后用K8s", "还没想好", "TBD", "待定",
        "未来可能用这个技术", "暂时不确定", "需要再考虑",
        "后续讨论", "可能性有很多", "不是很确定",
    ],
    'en': [
        "Maybe later", "We'll decide later", "Sort of",
        "Might use Kubernetes eventually", "Not sure yet",
        "TBD", "Pending", "Undecided",
        "Will reconsider", "Not certain yet", "To be determined",
    ],
    'ja': [
        "後で決める", "今のところ不確定", "いずれは",
        "将来K8sを使うかも", "未決定", "TBD", "ペンディング",
        "後で考える", "まだ決まってない", "不確実",
    ],
    'ko': [
        "나중에 결정하자", "아직 미정이다", "나중에",
        "훗날 K8s를 쓸 수도", "미정", "TBD", "보류중",
        "나중에 재고려", "아직 확실하지 않음", "미결정",
    ],
}

FAREWELL_TEMPLATES = {
    'zh': ["再见", "拜拜", "回见", "再聊", "下次再说"],
    'en': ["Bye", "Goodbye", "See you later", "Talk soon", "Take care"],
    'ja': ["さようなら", "またね", "バイバイ", "では", "またあした"],
    'ko': ["안녕", "잘가", "다음에 봐", "또 봐", "안녕히"],
    'fr': ["Au revoir", "À bientôt", "Bye", "À plus tard"],
    'de': ["Auf Wiedersehen", "Tschüss", "Bis bald", "Viel Erfolg"],
    'es': ["Adiós", "Hasta luego", "Nos vemos", "Cuídate"],
    'pt': ["Adeus", "Até logo", "Tchau", "Até mais"],
}

QUESTION_TEMPLATES = {
    'zh': [
        "怎么部署的?", "密码在哪?", "为什么选择这个技术?",
        "这个端口是多少?", "How does this work?", "什么时候上线?",
        "为什么改成这样?", "成本是多少?", "这个框架怎么样?",
        "地址是什么?", "怎样配置?", "什么原因?",
    ],
    'en': [
        "How to deploy?", "Where is the password?", "Why choose this tech?",
        "What is the port?", "How does this work?", "When to release?",
        "Why change this?", "What is the cost?", "How is the framework?",
        "What is the URL?", "How to configure?", "Why did you do that?",
    ],
    'ja': [
        "どうやってデプロイする?", "パスワードはどこ?",
        "なぜこの技術を選んだ?", "ポート番号は?",
        "これどうやるの?", "いつリリース?",
        "なぜ変更した?", "コストは?", "このフレームワークは?",
    ],
    'ko': [
        "어떻게 배포하나?", "비밀번호는 어디?", "왜 이 기술을 선택?",
        "포트는 뭐?", "어떻게 작동해?", "언제 출시?",
        "왜 바꿨어?", "비용은?", "이 프레임워크는?",
    ],
}

EMOTION_TEMPLATES = {
    'zh': [
        "太好了!", "太棒了!", "Amazing!", "完美!", "😍",
        "우와!", "싫어!", "😭", "실망", "화났어", "哈哈",
        "😂", "정말 싫어", "좋아", "정말 좋아",
    ],
    'en': [
        "Great!", "Awesome!", "Perfect!", "Excellent!", "😍",
        "Horrible!", "Terrible!", "😭", "Disappointed", "Angry",
        "Haha", "😂", "I hate it", "I love it",
    ],
    'ja': [
        "やった!", "素晴らしい!", "完璧!", "😍", "最高!",
        "嫌だ!", "😭", "がっかり", "怒った", "ハハ", "😂",
    ],
    'ko': [
        "좋아!", "최고!", "완벽!", "😍", "훌륭해!",
        "싫어!", "😭", "실망", "화났어", "하하", "😂",
    ],
}

OFFTOPIC_TEMPLATES = {
    'zh': [
        "最近看了个电影，很不错", "我的宠物很可爱", "足球比赛很好看",
        "去年假期去了海边", "最近的新闻怎么样", "我喜欢打游戏",
        "天气很好，去散步了", "最近在健身", "看了个演唱会",
    ],
    'en': [
        "Watched a great movie recently", "My pet is so cute",
        "Soccer match was good", "Went to beach last year",
        "Recent news is interesting", "I like playing games",
        "Nice weather, took a walk", "Going to gym lately",
        "Went to a concert",
    ],
    'ja': [
        "最近いい映画を見た", "ペットがかわいい",
        "サッカーの試合がいい", "去年ビーチに行った",
        "最近のニュースはどう", "ゲームが好きです",
        "いい天気、散歩した", "最近ジムに行ってる",
    ],
    'ko': [
        "최근 좋은 영화 봤어", "반려동물이 귀여워",
        "축구 경기 좋았어", "작년에 해변 갔어",
        "최근 뉴스 어때", "게임하는 거 좋아",
        "날씨 좋아서 산책했어", "최근 헬스장 다녀",
    ],
}

FILLER_TEMPLATES = {
    'zh': [
        "...", "嗯", "额", "呃", "嗯嗯", "。。。", "唉",
        "呵呵", "嘿", "哈", "嗯啦", "噢", "这样啊",
    ],
    'en': [
        "...", "Hmm", "Well", "Like", "Um", "Uh",
        "Huh", "Yeah", "Hehe", "OK", "So",
    ],
    'ja': [
        "...", "あ", "まあ", "えっと", "ふむ", "へぇ",
        "そうか", "あらら", "へえ", "ですね", "ま",
    ],
    'ko': [
        "...", "음", "그런데", "어", "저", "뭐",
        "흠", "네", "그래", "아", "흐음",
    ],
}

# Data pools for template substitution
TECH_CHOICES = ["MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "DynamoDB",
                 "React", "Vue", "Angular", "Svelte", "Python", "Node.js", "Go", "Rust",
                 "TypeScript", "Java", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin"]
FRAMEWORKS = ["Django", "FastAPI", "Spring", "Express", "Laravel", "Rails", "NestJS", "Flask",
              "Fastify", "Gin", "Echo", "Fiber", "Actix", "Rocket"]
USE_CASES = ["缓存", "logging", "queue", "データベース", "메시지 큐", "search", "distributed_tracing",
             "analytics", "monitoring", "notifications"]
COMPONENTS = ["frontend", "backend", "database", "cache", "queue", "storage", "api-gateway",
              "load-balancer", "reverse-proxy", "worker", "scheduler"]
TECH_PAIRS = [("MySQL", "PostgreSQL"), ("Redis", "Memcached"), ("Docker", "Kubernetes"),
              ("Solr", "Elasticsearch"), ("ELK", "Loki"), ("Nginx", "Apache"),
              ("React", "Vue"), ("REST", "GraphQL"), ("SQL", "NoSQL"), ("Monolith", "Microservices"),
              ("HTTP", "gRPC"), ("JSON", "Protocol Buffers")]
TOOLS = ["Sentry", "DataDog", "Prometheus", "Grafana", "ELK", "Splunk", "NewRelic", "Datadog",
         "Jaeger", "Loki", "Tempo", "VictorOps", "PagerDuty"]
TOOL_PAIRS = [("Sentry", "DataDog"), ("Prometheus", "Grafana"), ("ELK", "Splunk"),
              ("Sentry", "NewRelic"), ("Jaeger", "Zipkin")]
PATTERNS = ["microservice", "monolith", "serverless", "event-driven", "CQRS", "saga", "strangler"]
STYLES = ["snake_case", "camelCase", "PascalCase", "kebab-case"]
WORKFLOWS = ["GitHub Flow", "Git Flow", "trunk-based", "feature-branch", "release-branch"]
APPROACHES = [("async", "sync"), ("REST", "GraphQL"), ("SQL", "NoSQL"), ("pull", "push"),
              ("strong-consistency", "eventual-consistency")]

IPS = ["43.156.240.47", "192.168.1.1", "10.0.0.1", "127.0.0.1", "172.16.0.1",
       "203.0.113.45", "198.51.100.89", "192.0.2.34", "10.20.30.40", "172.31.45.67"]
PORTS = [8000, 8001, 3000, 5432, 6379, 3306, 27017, 9200, 9000, 5000, 4000, 8080,
         8888, 9090, 5005, 6060, 7000, 7001, 8443, 9999]
VERSIONS = ["1.0", "2.0", "3.0", "14.0", "16.0", "8.0", "11.0", "18.0",
            "20.0", "22.0", "24.0", "1.23", "2.15", "3.12", "4.5", "5.1"]
CONNECTION_STRINGS = [
    "postgres://user:pass@localhost:5432/db",
    "mongodb://localhost:27017/db",
    "redis://localhost:6379",
    "mysql://user:pass@localhost:3306/db",
    "postgresql://dbuser:secret@db.example.com:5432/production",
    "mongodb+srv://user:pass@cluster.mongodb.net/dbname",
    "amqp://guest:guest@rabbitmq:5672/",
]
FILE_PATHS = ["/opt/config/app.yaml", "/etc/nginx/nginx.conf", "/data/secret.env",
              "/opt/docker/compose.yml", "/root/.ssh/id_rsa", "/var/log/app.log",
              "~/.config/settings.json", "/etc/systemd/system/app.service"]
CRON_SCHEDULES = ["0 */6 * * *", "0 0 * * 0", "0 2 * * *", "*/5 * * * *", "30 3 * * *",
                  "0 */12 * * *", "0 3,15 * * *", "*/15 * * * *", "0 0 * * *"]
MEMORY_LIMITS = ["512MB", "1GB", "2GB", "4GB", "8GB", "16GB", "32GB", "256MB"]
TIMEOUTS = ["30s", "60s", "300s", "600s", "5s", "10s", "120s", "900s"]
DOMAINS = ["example.com", "api.example.com", "db.internal", "cache.local",
           "app.example.org", "cdn.example.net", "ws.example.io"]
DATES = ["2025-06-15", "2026-01-01", "2026-12-31", "2025-03-31", "2026-04-30",
         "2026-05-15", "2025-12-25", "2026-03-20", "2025-09-10"]
USERNAMES = ["admin", "root", "appuser", "postgres", "mysql", "service_user",
             "deploy", "jenkins", "github", "app"]
HOSTS = ["localhost", "db.local", "redis.local", "example.com",
         "db.prod.internal", "cache.staging", "api.prod"]
PASSWORDS = ["SecurePass123!", "MyP@ssw0rd", "Qwerty@123", "P@ssw0rd!",
             "Tr0ub4dour&3", "SuperSecure#99", "Complex$Pass2024"]

TASKS = ["完成API开发", "修复关键bug", "写单元测试", "部署到生产", "性能优化",
         "code review", "documentation", "リリース準備", "테스트 작성", "Update dependencies",
         "Fix security vulnerability", "Refactor legacy code", "Add monitoring",
         "Document API", "Setup CI/CD", "Configure alerts", "Optimize queries"]
ERROR_MSGS = ["Connection refused", "ECONNREFUSED", "OOM killed", "Timeout",
              "Segmentation fault", "NullPointerException", "FATAL: max_connections",
              "ENOTFOUND", "EADDRINUSE", "Deadlock detected"]
ERROR_TYPES = ["RuntimeException", "TimeoutError", "ConnectionError", "ValueError",
               "IOException", "ParseException", "IndexError", "TypeError"]
STACK_TRACES = [
    "at Object.<anonymous> (/app/index.js:42:15)",
    "File \"/app/main.py\", line 123, in process",
    "goroutine 1 [running]",
    "thread 1: tid = 0x2e03, 0x00007fff5fbff87a in __syscall_SYS_select",
    "panic: index out of range",
]
ROOT_CAUSES = ["memory leak in loop", "N+1 query in ORM", "missing index on table",
               "incorrect configuration", "race condition", "unclosed connection",
               "buffer overflow", "infinite loop", "deadlock in transaction"]

DECISIONS = ["转向微服务架构", "使用Kubernetes", "切换数据库", "采用GraphQL",
             "migrate to cloud", "use event sourcing", "adopt serverless", "go multi-region",
             "switch container runtime", "implement service mesh"]
PEOPLE = ["张三", "李四", "Alice", "Bob", "田中", "김철수", "Charlie", "Diana", "Eve"]
TOPICS = ["性能优化", "安全加固", "成本控制", "新技术评估", "灾备计划", "CI/CD流程", "监控告警"]

ARCH_DESCS = ["microservices with service mesh", "monolithic with PostgreSQL",
              "event-driven with Kafka", "Lambda-based serverless", "CQRS with event sourcing",
              "hexagonal architecture", "layered architecture"]
DATA_FLOWS = ["请求 → API → DB → 响应", "event → processor → storage",
              "user → server → cache → response", "producer → Kafka → consumer → DB",
              "request → gateway → service → response"]
SCALING_STRATEGIES = ["horizontal scaling with load balancer",
                      "vertical scaling on demand", "database sharding",
                      "read replicas", "caching layer", "CDN distribution"]
DESIGN_PATTERNS = ["Strategy", "Factory", "Observer", "Decorator", "Adapter",
                   "Singleton", "Builder", "Facade", "Proxy"]
COMMUNICATIONS = ["gRPC", "REST APIs", "message queues", "WebSockets", "MQTT", "AMQP"]
FAILOVERS = ["read replicas", "backup server", "regional failover", "active-active replication"]

REQUIREMENTS = ["API response time < 200ms", "99.9% uptime", "支持百万级用户",
                "HIPAA compliant", "real-time notifications", "GDPR compliance",
                "multi-region support", "zero-downtime deployments"]
ACCEPTANCE_CRITERIA = ["given X, when Y, then Z", "all tests pass",
                       "performance meets SLA", "no security issues",
                       "code coverage > 80%", "no breaking changes"]
USER_STORIES = ["As a user, I want to X so that Y",
                "管理员能够批量导入数据", "ユーザーが簡単にXできる",
                "사용자가 Y할 수 있다"]
FEATURE_REQS = ["Authentication with OAuth2", "full-text search",
                "real-time collaboration", "offline mode", "dark mode support",
                "multi-language support", "accessibility compliance"]
NON_FUNCTIONAL = ["response time < 200ms", "support 10K concurrent users",
                  "99.99% availability", "encrypted in transit and at rest",
                  "sub-second query latency", "handle 1M req/min"]


def generate_samples_for_category(category, should_store, count, templates_dict):
    """Generate samples for a specific category."""
    samples = []

    for _ in range(count):
        # Select language
        lang = random.choice(list(templates_dict.keys()))

        # Select template
        template = random.choice(templates_dict[lang])

        # Prepare substitution dict based on category
        subs = {}

        if category == 'decision':
            tech1, tech2 = random.choice(TECH_PAIRS)
            subs = {
                'tech1': tech1, 'tech2': tech2,
                'framework': random.choice(FRAMEWORKS),
                'use_case': random.choice(USE_CASES),
                'component': random.choice(COMPONENTS),
            }
        elif category == 'fact':
            subs = {
                'ip': random.choice(IPS),
                'port': random.choice(PORTS),
                'version': random.choice(VERSIONS),
                'connection_string': random.choice(CONNECTION_STRINGS),
                'file_path': random.choice(FILE_PATHS),
                'cron_schedule': random.choice(CRON_SCHEDULES),
                'memory_limit': random.choice(MEMORY_LIMITS),
                'timeout': random.choice(TIMEOUTS),
                'domain': random.choice(DOMAINS),
                'date': random.choice(DATES),
                'username': random.choice(USERNAMES),
                'host': random.choice(HOSTS),
                'password': random.choice(PASSWORDS),
            }
        elif category == 'preference':
            subs = {
                'tool': random.choice(TOOLS),
                'tool2': random.choice(TOOLS),
                'pattern': random.choice(PATTERNS),
                'style': random.choice(STYLES),
                'workflow': random.choice(WORKFLOWS),
            }
            approach = random.choice(APPROACHES)
            subs['approach'] = approach[0]
            subs['approach2'] = approach[1]
        elif category == 'correction':
            subs = {
                'port': random.choice(PORTS),
                'password': random.choice(PASSWORDS),
                'version': random.choice(VERSIONS),
                'connection_string': random.choice(CONNECTION_STRINGS),
                'ip': random.choice(IPS),
            }
        elif category == 'explicit':
            subs = {'content': random.choice(["Redis端口是6379", "SSH密钥在/root/.ssh/",
                                               "database is on db.local", "API timeout is 30s"])}
        elif category == 'action':
            subs = {
                'task': random.choice(TASKS),
                'date': random.choice(DATES),
            }
        elif category == 'command':
            subs = {
                'image': 'nginx:latest',
                'file': 'deploy.yaml',
                'message': 'fix: update dependencies',
                'host': random.choice(HOSTS),
                'user': random.choice(USERNAMES),
                'db': 'myapp',
                'plan': 'main',
                'playbook': 'deploy.yml',
                'bucket': 'my-bucket',
                'package': 'requests',
                'method': 'POST',
                'url': 'https://api.example.com/v1/data',
            }
        elif category == 'error_log':
            subs = {
                'error_msg': random.choice(ERROR_MSGS),
                'file': 'app.py',
                'line': random.randint(10, 500),
                'stack_trace': random.choice(STACK_TRACES),
                'error_type': random.choice(ERROR_TYPES),
                'root_cause': random.choice(ROOT_CAUSES),
            }
        elif category == 'meeting_note':
            subs = {
                'decision': random.choice(DECISIONS),
                'person': random.choice(PEOPLE),
                'task': random.choice(TASKS),
                'date': random.choice(DATES),
                'topic': random.choice(TOPICS),
            }
        elif category == 'architecture':
            subs = {
                'arch_desc': random.choice(ARCH_DESCS),
                'data_flow': random.choice(DATA_FLOWS),
                'scaling_strategy': random.choice(SCALING_STRATEGIES),
                'pattern': random.choice(DESIGN_PATTERNS),
                'communication': random.choice(COMMUNICATIONS),
                'failover': random.choice(FAILOVERS),
            }
        elif category == 'requirement':
            subs = {
                'requirement': random.choice(REQUIREMENTS),
                'acceptance_criteria': random.choice(ACCEPTANCE_CRITERIA),
                'user_story': random.choice(USER_STORIES),
                'feature_req': random.choice(FEATURE_REQS),
                'non_functional': random.choice(NON_FUNCTIONAL),
            }

        try:
            content = template.format(**subs)
        except KeyError:
            # Skip if template doesn't have all required keys
            continue

        samples.append({
            'content': content,
            'should_store': should_store,
            'category': category,
            'lang': lang,
        })

    return samples


def main():
    # Load existing data
    data_path = Path('/Users/lilanfeng/project/tttt/mamba-memory/data/training_data.json')

    with open(data_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    print(f"Loaded {len(existing_data)} existing samples")

    # Generate new samples
    all_samples = list(existing_data)  # Start with existing data

    # SHOULD STORE categories (increased counts to account for deduplication)
    store_categories = {
        'decision': (DECISION_TEMPLATES, 2500),
        'fact': (FACT_TEMPLATES, 3500),
        'preference': (PREFERENCE_TEMPLATES, 1200),
        'correction': (CORRECTION_TEMPLATES, 1200),
        'explicit': (EXPLICIT_TEMPLATES, 1200),
        'action': (ACTION_TEMPLATES, 1200),
        'command': (COMMAND_TEMPLATES, 1200),
        'error_log': (ERROR_LOG_TEMPLATES, 1000),
        'meeting_note': (MEETING_NOTE_TEMPLATES, 1000),
        'architecture': (ARCHITECTURE_TEMPLATES, 1000),
        'requirement': (REQUIREMENT_TEMPLATES, 800),
    }

    # SHOULD DISCARD categories (increased counts to account for deduplication)
    discard_categories = {
        'greeting': (GREETING_TEMPLATES, 1200),
        'ack': (ACK_TEMPLATES, 2000),
        'smalltalk': (SMALLTALK_TEMPLATES, 1200),
        'vague': (VAGUE_TEMPLATES, 2000),
        'farewell': (FAREWELL_TEMPLATES, 800),
        'question': (QUESTION_TEMPLATES, 1200),
        'emotion': (EMOTION_TEMPLATES, 1000),
        'offtopic': (OFFTOPIC_TEMPLATES, 800),
        'filler': (FILLER_TEMPLATES, 800),
    }

    print("\nGenerating SHOULD STORE samples...")
    for category, (templates, count) in store_categories.items():
        samples = generate_samples_for_category(category, True, count, templates)
        all_samples.extend(samples)
        print(f"  {category}: {len(samples)} samples")

    print("\nGenerating SHOULD DISCARD samples...")
    for category, (templates, count) in discard_categories.items():
        samples = generate_samples_for_category(category, False, count, templates)
        all_samples.extend(samples)
        print(f"  {category}: {len(samples)} samples")

    # Deduplicate by content
    seen_contents = {}
    deduplicated = []
    duplicates = 0

    for sample in all_samples:
        content = sample['content']
        if content not in seen_contents:
            seen_contents[content] = True
            deduplicated.append(sample)
        else:
            duplicates += 1

    print(f"\nRemoved {duplicates} duplicate samples")
    print(f"Total unique samples: {len(deduplicated)}")

    # Save back to file
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    stats = {
        'total_count': len(deduplicated),
        'store_discard_ratio': len([s for s in deduplicated if s['should_store']]) / len(deduplicated),
        'categories': defaultdict(int),
        'languages': defaultdict(int),
    }

    for sample in deduplicated:
        stats['categories'][sample['category']] += 1
        stats['languages'][sample['lang']] += 1

    print(f"\nTotal samples: {stats['total_count']}")
    print(f"Store/Discard ratio: {stats['store_discard_ratio']:.2%}")

    print("\nPer-category count:")
    for category in sorted(stats['categories'].keys()):
        count = stats['categories'][category]
        should_store = len([s for s in deduplicated if s['category'] == category and s['should_store']])
        print(f"  {category:20s}: {count:4d} (store: {should_store})")

    print("\nPer-language count:")
    for lang in sorted(stats['languages'].keys()):
        count = stats['languages'][lang]
        print(f"  {lang}: {count}")

    print("\n" + "="*60)
    print(f"Training data saved to {data_path}")


if __name__ == '__main__':
    main()
