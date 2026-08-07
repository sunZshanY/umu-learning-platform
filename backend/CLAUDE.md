# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

UMU刷题助手 — 基于 AI 大模型的智能答题系统。通过 DeepSeek API 对 UMU 学习平台的题目进行 AI 作答，支持 Web 前端、API 接口和 CLI 命令行三种使用方式。

## 常用命令

```bash
# 启动后端服务 (开发模式，热重载)
python server.py
# 或
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# CLI 全自动刷题
python auto_quiz.py questions.txt                    # 基本用法
python auto_quiz.py questions.txt -c 15 -t 180       # 15并发, 3分钟限时
python auto_quiz.py questions.txt --dry-run          # 仅解析题目不调用AI

# 离线答案同步纠错（不联网，将AI答案与权威答案校对并生成可提交结果）
python answer_sync.py questions.txt --answers ai_answers.txt            # 规范化+纠错
python answer_sync.py questions.txt --answers ai_answers.txt --gold gold_answers.txt  # 批改+纠错
python test_answer_sync.py                                                 # 运行单元测试

# 安装依赖
pip install -r requirements.txt
```

## 架构概览

```
backend/
├── server.py          # FastAPI 入口 — API路由、中间件、生命周期管理
├── ai_service.py      # AI答题核心 — 封装 DeepSeek/OpenAI 兼容API调用、重试、结果解析
├── auto_quiz.py       # CLI全自动刷题 — 独立运行，文件→并发AI答题→答案输出
├── umu_client.py      # UMU平台客户端 — URL解析、会话管理、题目抓取、答案提交
├── models.py          # Pydantic数据模型 — 所有请求/响应类型定义
├── question_parser.py # 题目解析器 — 从复制文本中提取结构化题目（纯规则，不调AI）
├── offline_quiz_validator.py # 离线校验 — 解析题目/答案TXT，校验格式与数据完整性
├── answer_sync.py     # 离线答案同步纠错 — AI答案↔权威答案校对，输出可提交答案
├── deepseek_local_answers.py # 本地AI答题 — 用DeepSeek为本地题目生成答案建议（不连UMU）
├── requirements.txt   # Python依赖
└── .env               # API Key等敏感配置
```

## 关键设计

- **AI调用链路**: `server.py` → `AIService.answer_question()` → `AsyncOpenAI.chat.completions.create()` → 响应解析 → `AnswerResponse`
- **API兼容性**: `AIService` 使用 OpenAI SDK 调用 DeepSeek API（OpenAI兼容格式），通过 `base_url` 区分
- **重试策略**: 指数退避，最多3次重试，区分网络错误/限流/API错误
- **批量并发**: `answer_batch()` 使用 `asyncio.Semaphore` 限流(默认15并发)
- **题目解析**: `QuestionParser` 纯规则引擎，不消耗AI token，识别题号、选项、题型标签
- **题目解析**: `QuestionParser` 纯规则引擎，不消耗AI token，识别题号、选项、题型标签
- **离线纠错**: `answer_sync.py` 复用 `offline_quiz_validator` 的解析与规范化，权威答案优先、AI无效答案纠正、无权威则仅规范化
- **配置优先级**: 环境变量 > `.env`文件 > 代码默认值
- **CLI独立**: `auto_quiz.py` 有自己的 `AIClient` 和 `QuestionParser`，可在无服务环境下独立运行
