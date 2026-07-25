#  UMU刷题助手

<div align="center">

**基于AI大模型的智能答题工具 | 支持UMU互动学习平台**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Mobile](https://img.shields.io/badge/Mobile-Friendly-7C3AED.svg)]()

</div>

---

##  目录
- [为什么要开发这个](#-为什么开发)
- [功能特性](#-功能特性)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
  - [个人版（预配置API Key）](#个人版预配置api-key)
  - [公开版（用户自带API Key）](#公开版用户自带api-key)
- [API文档](#-api文档)
- [配置说明](#-配置说明)
- [移动端使用](#-移动端使用)
- [UMU答题规则](#-umu答题规则)
- [错误处理](#-错误处理)
- [开发指南](#-开发指南)
- [发布到GitHub](#-发布到github)
- [常见问题](#-常见问题)
- [免责声明](#-免责声明)

---

## 为什么要开发这个

- 不想写作业，以及这个神人老师还很几把严格，完成不了直接给你开学的时候给你留在教室里刷题以及。。。。。（你们懂得，处分
- 这个是我的第2个开发项目喵~不想开学喵~!
  
---

##  功能特性

- ✅ **AI智能答题** — 支持单选题、多选题、判断题、填空题、简答题
- ✅ **移动端优先** — 响应式设计，手机浏览器完美适配
- ✅ **批量处理** — 一次提交多道题目，并发获取答案
- ✅ **置信度显示** — 每道答案附带AI置信度评分(0-100%)
- ✅ **答案解析** — AI提供详细的推理过程和解析
- ✅ **三重防护** — 严格遵循UMU的3次错误上限规则
- ✅ **3分钟时限** — 适配UMU答题时间限制
- ✅ **会话追踪** — 本地记录答题历史、正确率、平均耗时
- ✅ **双版本** — 个人版(预配置Key) + 公开版(用户自带Key)
- ✅ **容错重试** — 网络异常自动重试(最多3次)，指数退避
- ✅ **速率限制** — 内置API频率控制，防止滥用
- ✅ **暗色模式** — 自动跟随系统主题
- ✅ **键盘快捷键** — `Ctrl+Enter`快速提交

---

##  系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    移动端浏览器                           │
│  ┌──────────────────┐   ┌──────────────────┐            │
│  │  UMU答题页面      │   │  刷题助手前端      │            │
│  │  (m.umu.cn)      │   │  (localhost:8000) │            │
│  └────────┬─────────┘   └────────┬─────────┘            │
│           │ 复制题目              │ HTTP请求              │
└───────────┼──────────────────────┼──────────────────────┘
            │                      │
            │              ┌───────▼────────┐
            │              │  FastAPI后端    │
            │              │  (Python 3.10+) │
            │              ├────────────────┤
            │              │ • AI答题服务    │
            │              │ • UMU客户端     │
            │              │ • 速率限制      │
            │              │ • 错误处理      │
            │              └───────┬────────┘
            │                      │
            │              ┌───────▼────────┐
            │              │  AI大模型API    │
            │              │  DeepSeek/OpenAI│
            │              └────────────────┘
```

---

##  快速开始

### 环境要求

- **Python 3.10+**
- **pip** (Python包管理器)
- **网络连接** (访问AI API)

### 1. 克隆/下载项目

```bash
cd "UMU学习平台"
```

### 2. 安装依赖

```bash
pip install -r backend/requirements.txt
```

### 3. 启动服务

**Windows:**
```bash
双击运行 start.bat
```
或
```bash
cd backend
python server.py
```

**macOS/Linux:**
```bash
chmod +x start.sh
./start.sh
```
或
```bash
cd backend && python server.py
```

### 4. 打开页面

- **个人版**: http://localhost:8000  
- **公开版**: http://localhost:8000/public
- **API文档**: http://localhost:8000/api/docs
**注意:localhost:8000是本地的喵~，嗯对.....**
  
---

##  使用指南

### 个人版（预配置API Key）

> 适用于：API Key拥有者本人使用

1. 启动服务后直接访问 `http://localhost:8000`
2. API Key已预配置（通过环境变量或代码中配置）
3. 在UMU答题页面复制题目文本
4. 粘贴到输入框 → 点击「获取AI答案」
5. 查看AI给出的答案、解析和置信度
6. 根据实际结果点击「答案正确」或「答案错误」

### 公开版（用户自带API Key）

> 适用于：分发给其他人使用

1. 访问 `http://localhost:8000/public`
2. 首次使用会自动弹出设置面板
3. 填入自己的API Key（支持DeepSeek/OpenAI等）
4. 点击「测试连接」验证配置
5. 保存后即可正常使用
6. **API Key仅保存在浏览器本地，不会上传到服务器**

#### 获取API Key

| 平台 | 地址 | 价格 | 推荐 |
|------|------|------|------|
| **DeepSeek** | [platform.deepseek.com](https://platform.deepseek.com) | ¥2/百万token | ⭐推荐
| **Kimi**    |  [platform.kimi.com](https://platform.kimi.com/)     | ¥2.00/百万token | ❤️可以试试
---

## 📚 API文档

启动服务后访问: `http://localhost:8000/api/docs`

### 主要端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/info` | 服务信息 |
| `POST` | `/api/answer` | 单题AI作答 |
| `POST` | `/api/answer-batch` | 批量AI作答 |
| `POST` | `/api/umu/parse-session` | 解析UMU会话 |
| `GET` | `/api/umu/session/{id}` | 查询会话状态 |
| `POST` | `/api/feedback` | 提交答题反馈 |

### 示例请求

```bash
curl -X POST http://localhost:8000/api/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "以下哪个是中国的首都？\nA. 上海\nB. 北京\nC. 广州\nD. 深圳",
    "question_type": "single_choice",
    "options": ["A. 上海", "B. 北京", "C. 广州", "D. 深圳"]
  }'
```

### 示例响应

```json
{
  "success": true,
  "answer": "B. 北京",
  "explanation": "北京是中华人民共和国的首都，这是基本的地理常识。",
  "confidence": 0.98,
  "question_type": "single_choice",
  "error": null,
  "retry_count": 0
}
```

---

##  配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `UMU_AI_API_KEY` | AI API密钥 | 空，需自行配置 |
| `UMU_AI_BASE_URL` | AI API基础URL | `https://api.deepseek.com/v1` |
| `UMU_AI_MODEL` | AI模型名称 | `deepseek-chat` |
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务监听端口 | `8000` |

### 配置方式

```bash
# Windows PowerShell
$env:UMU_AI_API_KEY="your-api-key-here"
$env:UMU_AI_BASE_URL="https://api.openai.com/v1"
$env:UMU_AI_MODEL="gpt-4o"
python backend/server.py

# Linux/macOS
export UMU_AI_API_KEY="your-api-key-here"
export UMU_AI_BASE_URL="https://api.openai.com/v1"
python backend/server.py
```

---

##  移动端使用

### 方式一：局域网访问（推荐）

1. 在电脑上启动服务
2. 确保手机和电脑在同一WiFi
3. 查看电脑IP地址：
   - Windows: `ipconfig`
   - macOS/Linux: `ifconfig`
4. 手机浏览器访问: `http://<电脑IP>:8000`

### 方式二：内网穿透

使用 ngrok / frp 等工具将本地服务暴露到公网

```bash
# ngrok示例
ngrok http 8000
```

### 方式三：部署到服务器

将项目部署到云服务器（阿里云/腾讯云/Vercel等）

---

##  UMU答题规则

UMU平台关键约束（系统已内置保护机制）：

| 规则 | 说明 | 系统保护 |
|------|------|----------|
| **3次错误** | 累计3次答错 → 自动返回大厅 | ✅ 实时追踪 |
| **3分钟时限** | 单次答题时长限制 | ✅ 计时提示 |
| **不可重复** | 同一题目不可重复作答 | ✅ 会话管理 |

**系统会在错误次数达上限时自动锁定，防止继续答题。**

---

## ❌ 错误处理

本项目实现了多层次的错误处理机制：

### AI服务层
- **自动重试**: 网络超时/连接失败自动重试3次
- **指数退避**: 重试间隔 1s → 2s → 4s
- **降级处理**: 重试全部失败后返回明确错误信息
- **速率限制**: 遇到429自动等待后重试

### 用户界面层
- **20秒超时**: 单题请求最长等待20秒
- **60秒超时**: 批量请求最长等待60秒
- **友好提示**: 中文错误消息，明确处理建议
- **手动重试**: 失败后可一键重试

### 业务逻辑层
- **输入校验**: 题目不能为空，批量上限30题
- **状态保护**: 错误达上限后禁用提交按钮
- **数据持久化**: 答题记录保存在浏览器本地

---

##  开发指南

### 项目结构

```
UMU学习平台/
├── backend/
│   ├── server.py          # FastAPI主服务
│   ├── ai_service.py      # AI答题核心服务
│   ├── umu_client.py      # UMU平台API客户端
│   ├── models.py          # Pydantic数据模型
│   ├── requirements.txt   # Python依赖
│   └── server.log         # 运行日志
├── frontend/
│   ├── personal/
│   │   └── index.html     # 个人版前端
│   └── public/
│       └── index.html     # 公开版前端
├── start.bat              # Windows启动脚本
├── start.sh               # Linux/macOS启动脚本
├── .gitignore
└── README.md
```

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI 0.115+ | 高性能异步Web框架 |
| AI SDK | OpenAI Python SDK | 兼容多平台API调用 |
| HTTP客户端 | httpx | 异步HTTP请求 |
| 数据校验 | Pydantic v2 | 类型安全的数据模型 |
| 前端 | Vanilla JS | 零依赖，纯原生实现 |

### 添加新题型支持

1. 在 `models.py` 中添加新的 `QuestionType` 枚举值
2. 在 `ai_service.py` 的 `_build_user_prompt` 中添加类型映射
3. 在 `_build_system_prompt` 中添加答题规则
4. 前端 `questionType` 下拉框中添加选项

---

## ❓ 常见问题

<details>
<summary><strong>Q: 公开版用户的API Key安全吗？</strong></summary>

API Key仅存储在用户浏览器本地（localStorage），不会发送到我们的服务器。公开版前端直接调用AI API，不经过后端代理。
</details>

<details>
<summary><strong>Q: 支持哪些AI平台？</strong></summary>

支持所有兼容OpenAI API格式的平台：DeepSeek、OpenAI、Kimi、Azure OpenAI、Moonshot、智谱、百川等.....
</details>

<details>
<summary><strong>Q: 手机如何访问？</strong></summary>

确保手机和电脑在同一WiFi下，手机浏览器访问 `http://<电脑IP>:8000`。查看电脑IP：Windows运行`ipconfig`，Mac运行`ifconfig`。
</details>

<details>
<summary><strong>Q: 答题准确率如何？</strong></summary>

使用DeepSeek-Chat模型，基础知识类题目准确率约90%+。专业领域题目建议结合自身知识判断。置信度<50%时建议谨慎采纳。
</details>

<details>
<summary><strong>Q: 为什么会提示"错误次数达上限"？</strong></summary>

这是系统保护机制。UMU在3次错误后会强制返回大厅。系统会追踪你的每次错误标记，达上限后自动禁用提交防止继续错答。
</details>

---

## ⚠️ 免责声明

本项目仅供学习和个人辅助使用。使用者应：

1. 遵守UMU平台的使用条款和学术诚信规范
2. 本项目不应被用于任何形式的考试作弊
3. AI生成的答案可能存在错误，使用者需自行判断
4. 作者不对因使用本项目导致的任何后果承担责任

---

<p align="center">最后就是这个项目要是有大佬闲着无聊的话可以来看下，以及帮我看下有什么不对之处修改，可投pr喵!~❤️</p>
<div align="center">
  <p> <b> 新人开发请多多关照喵❤️~ </b></p>
  
  <sub>Made with ❤️ for learning | MIT License</sub>
</div>
