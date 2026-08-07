"""
UMU刷题助手 — FastAPI后端服务
=================================
提供AI智能答题API，支持移动端访问。
支持个人模式（预配置API Key）和公开模式（用户自带Key）。

启动方式:
    python server.py
    或
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

作者: UMU Quiz Helper Team
版本: 1.0.0
"""
import os
import sys
import time
import logging
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

from models import (
    AnswerRequest,
    BatchAnswerRequest,
    AnswerResponse,
    BatchAnswerResponse,
    HealthResponse,
    FeedbackRequest,
    UmuSessionRequest,
    ParseQuestionsRequest,
    ParseQuestionsResponse,
    ParsedQuestion,
    QuestionType,
)
from ai_service import AIService
from umu_client import UmuClient
from question_parser import QuestionParser

# ── 日志配置 ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "server.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# ── 配置常量 ─────────────────────────────────────────────────
APP_VERSION = "1.0.0"
DEFAULT_API_KEY = os.environ.get(
    "UMU_AI_API_KEY",
    "",
)
DEFAULT_BASE_URL = os.environ.get(
    "UMU_AI_BASE_URL",
    "https://api.deepseek.com/v1",
)
DEFAULT_MODEL = os.environ.get("UMU_AI_MODEL", "deepseek-chat")
MAX_BATCH_SIZE = 300
MAX_PARSE_TEXT_LENGTH = 100000
RATE_LIMIT_PER_MINUTE = 60

# ── 全局服务实例 ────────────────────────────────────────────
ai_service: Optional[AIService] = None
umu_client: Optional[UmuClient] = None
server_start_time: float = 0.0

# ── 简单的内存速率限制 ───────────────────────────────────────
_rate_limit_store: dict = {}


def check_rate_limit(client_ip: str, max_per_minute: int = RATE_LIMIT_PER_MINUTE) -> bool:
    """简单的滑动窗口速率限制"""
    now = time.time()
    window_start = now - 60

    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []

    # 清理过期记录
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= max_per_minute:
        return False

    _rate_limit_store[client_ip].append(now)
    return True


# ── 应用生命周期 ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理"""
    global ai_service, umu_client, server_start_time

    logger.info("=" * 60)
    logger.info("UMU刷题助手后端服务启动中...")
    logger.info(f"版本: {APP_VERSION}")
    logger.info(f"AI模型: {DEFAULT_MODEL}")
    logger.info(f"API端点: {DEFAULT_BASE_URL}")
    logger.info("=" * 60)

    # 初始化AI服务
    try:
        ai_service = AIService(
            api_key=DEFAULT_API_KEY,
            base_url=DEFAULT_BASE_URL,
            model=DEFAULT_MODEL,
        )
        is_connected, msg = await ai_service.test_connection()
        if is_connected:
            logger.info(f"✅ AI服务连接成功: {msg}")
        else:
            logger.warning(f"⚠️  AI服务连接异常: {msg}")
            logger.warning("服务仍可启动，但答题功能可能不可用")
    except Exception as e:
        logger.error(f"❌ AI服务初始化失败: {e}")
        ai_service = None

    # 初始化UMU客户端
    umu_client = UmuClient()
    logger.info("✅ UMU客户端初始化完成")

    server_start_time = time.time()
    logger.info("✅ 服务器就绪，等待请求...")

    yield

    # 关闭逻辑
    logger.info("服务器正在关闭...")
    if ai_service:
        stats = ai_service.stats
        logger.info(f"AI服务统计: {stats}")
    logger.info("服务器已关闭")


# ── FastAPI应用实例 ─────────────────────────────────────────
app = FastAPI(
    title="UMU刷题助手 API",
    description="基于AI大模型的智能答题服务，支持UMU互动学习平台",
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── CORS中间件（允许移动端跨域访问）─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id", "X-Response-Time"],
)


# ── 请求计时中间件 ──────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """为每个请求添加响应时间头"""
    start = time.time()
    response = await call_next(request)
    process_time = (time.time() - start) * 1000
    response.headers["X-Response-Time"] = f"{process_time:.0f}ms"
    response.headers["X-App-Version"] = APP_VERSION
    return response


# ── 全局异常处理 ────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求参数校验异常"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    logger.warning(f"请求参数校验失败: {errors}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "请求参数校验失败",
            "details": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理 — 捕获所有未预期的异常"""
    logger.exception(f"未预期的服务器错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "服务器内部错误，请稍后重试",
            "error_id": str(time.time()).replace(".", "")[-12:],
        },
    )


# ═══════════════════════════════════════════════════════════════
# API路由
# ═══════════════════════════════════════════════════════════════

# ── 健康检查 ─────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """服务健康检查"""
    api_connected = False
    if ai_service:
        connected, _ = await ai_service.test_connection()
        api_connected = connected

    return HealthResponse(
        status="ok" if api_connected else "degraded",
        version=APP_VERSION,
        api_connected=api_connected,
        uptime_seconds=time.time() - server_start_time if server_start_time else 0,
    )


# ── 自动提取题目 ───────────────────────────────────────────────
@app.post("/api/questions/parse", response_model=ParseQuestionsResponse)
async def parse_questions(request: ParseQuestionsRequest, req: Request):
    """
    从复制的整页文本或题目文本中自动提取结构化题目。

    - **raw_text**: UMU页面复制文本或题目列表
    - **source**: 文本来源，默认 paste
    """
    client_ip = req.client.host if req.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")

    if len(request.raw_text) > MAX_PARSE_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"文本过长，最多支持{MAX_PARSE_TEXT_LENGTH}个字符",
        )

    parsed = QuestionParser.parse_text(request.raw_text)
    questions = [
        ParsedQuestion(
            id=q.id,
            question=q.question,
            question_type=q.question_type,
            options=q.options,
            raw_text=q.raw_text,
            confidence=q.confidence,
            warnings=q.warnings,
        )
        for q in parsed
    ]
    warnings = []
    if not questions:
        warnings.append("未检测到题目，请检查复制内容或改用手动分隔")

    logger.info(
        f"题目提取请求 | IP: {client_ip} | 来源: {request.source} | "
        f"字符: {len(request.raw_text)} | 题数: {len(questions)}"
    )

    return ParseQuestionsResponse(
        success=True,
        questions=questions,
        meta={
            "source": request.source,
            "total": len(questions),
            "warnings": warnings,
        },
    )


# ── 单题作答 ─────────────────────────────────────────────────
@app.post("/api/answer", response_model=AnswerResponse)
async def answer_single_question(
    request: AnswerRequest,
    req: Request,
):
    """
    单题AI作答

    提交一道题目，返回AI生成的答案、解析和置信度。
    支持所有常见题型（单选、多选、判断、填空、简答）。

    - **question**: 题目文本（必填）
    - **question_type**: 题目类型（可选，AI会自动判断）
    - **options**: 选项列表（可选，用于选择题）
    - **context**: 额外上下文（可选）
    """
    # 速率限制
    client_ip = req.client.host if req.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试（每分钟60次限制）")

    if not ai_service:
        raise HTTPException(status_code=503, detail="AI服务未初始化，请检查API Key配置")

    logger.info(f"单题作答请求 | IP: {client_ip} | 类型: {request.question_type.value}")
    logger.debug(f"题目内容: {request.question[:100]}...")

    result = await ai_service.answer_question(
        question=request.question,
        question_type=request.question_type,
        options=request.options,
        context=request.context,
    )

    if not result.success:
        logger.error(f"答题失败: {result.error}")

    return result


# ── 批量答题 ─────────────────────────────────────────────────
@app.post("/api/answer-batch", response_model=BatchAnswerResponse)
async def answer_batch_questions(
    request: BatchAnswerRequest,
    req: Request,
):
    """
    批量AI答题

    一次提交多道题目（最多30题），并发处理，大幅提升效率。

    - **questions**: 题目列表
    - **session_id**: UMU会话ID（可选，用于追踪）
    """
    client_ip = req.client.host if req.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")

    if not ai_service:
        raise HTTPException(status_code=503, detail="AI服务未初始化")

    if len(request.questions) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多处理{MAX_BATCH_SIZE}道题目，当前提交了{len(request.questions)}道",
        )

    logger.info(
        f"批量答题请求 | IP: {client_ip} | "
        f"题数: {len(request.questions)} | 会话: {request.session_id or 'N/A'}"
    )

    start_time = time.time()
    results = await ai_service.answer_batch(request.questions)
    elapsed = (time.time() - start_time) * 1000

    answered = sum(1 for r in results if r.success)
    failed = len(results) - answered

    logger.info(f"批量答题完成 | 成功: {answered} | 失败: {failed} | 耗时: {elapsed:.0f}ms")

    return BatchAnswerResponse(
        success=failed == 0,
        total=len(results),
        answered=answered,
        failed=failed,
        results=results,
        total_time_ms=elapsed,
    )


# ── UMU会话解析 ──────────────────────────────────────────────
@app.post("/api/umu/parse-session")
async def parse_umu_session(request: UmuSessionRequest):
    """
    解析UMU答题会话

    从UMU URL中提取会话信息，可选地使用session_token进行API访问。

    - **session_url**: UMU答题页URL
    - **session_token**: 可选的登录认证token
    """
    if not umu_client:
        raise HTTPException(status_code=503, detail="UMU客户端未初始化")

    session, error = umu_client.create_session(request.session_url)
    if error:
        return {
            "success": False,
            "error": error,
        }

    # 尝试获取题目（如果提供了token/cookie）
    questions = None
    fetch_error = None
    if request.session_token:
        cookies = {"session_token": request.session_token}
        questions, fetch_error = await umu_client.fetch_questions(
            session.session_id, cookies=cookies
        )

    return {
        "success": True,
        "session_id": session.session_id,
        "questions": questions,
        "fetch_error": fetch_error,
        "remaining_attempts": umu_client.get_remaining_attempts(session.session_id),
    }


# ── UMU会话状态查询 ──────────────────────────────────────────
@app.get("/api/umu/session/{session_id}")
async def get_session_status(session_id: str):
    """查询UMU答题会话状态"""
    if not umu_client:
        raise HTTPException(status_code=503, detail="UMU客户端未初始化")

    session = umu_client.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")

    return {
        "success": True,
        "session": {
            "session_id": session.session_id,
            "current_index": session.current_index,
            "total_questions": session.total_questions,
            "wrong_count": session.wrong_count,
            "max_wrong_attempts": session.max_wrong_attempts,
            "remaining_attempts": umu_client.get_remaining_attempts(session_id),
            "is_active": session.is_active,
            "time_limit_seconds": session.time_limit_seconds,
            "elapsed_seconds": time.time() - session.created_at,
        },
    }


# ── 用户反馈（用于调优）──────────────────────────────────────
@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    提交答题反馈

    当AI答案不正确时，用户可以提交正确答案，帮助我们优化提示词。
    反馈数据记录到日志文件，可用于后续分析。
    """
    logger.info(
        f"用户反馈 | 正确: {request.was_correct} | "
        f"AI答案: {request.ai_answer[:50]}... | "
        f"正确答案: {request.correct_answer or 'N/A'}"
    )

    # 记录到专门的反馈日志
    feedback_logger = logging.getLogger("feedback")
    feedback_logger.info(
        f"题目: {request.question[:100]}... | "
        f"AI: {request.ai_answer} | "
        f"正确: {request.correct_answer or '?'} | "
        f"是否正确: {request.was_correct}"
    )

    return {"success": True, "message": "感谢您的反馈！"}


# ── API信息 ──────────────────────────────────────────────────
@app.get("/api/info")
async def api_info():
    """获取API服务信息"""
    ai_stats = ai_service.stats if ai_service else {}
    return {
        "success": True,
        "version": APP_VERSION,
        "model": DEFAULT_MODEL,
        "base_url": DEFAULT_BASE_URL,
        "stats": ai_stats,
        "max_batch_size": MAX_BATCH_SIZE,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "uptime_seconds": time.time() - server_start_time if server_start_time else 0,
    }


# ═══════════════════════════════════════════════════════════════
# 前端静态文件服务
# ═══════════════════════════════════════════════════════════════

# 获取前端目录路径
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_MAIN = os.path.join(PROJECT_DIR, "frontend")
FRONTEND_PERSONAL = os.path.join(PROJECT_DIR, "frontend", "personal")
FRONTEND_PUBLIC = os.path.join(PROJECT_DIR, "frontend", "public")


@app.get("/", response_class=HTMLResponse)
async def serve_main_index():
    """全自动刷题主页"""
    index_path = os.path.join(FRONTEND_MAIN, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>前端文件未找到</h1>", status_code=404)


@app.get("/personal", response_class=HTMLResponse)
async def serve_personal_index():
    """个人版（旧版兼容）"""
    index_path = os.path.join(FRONTEND_PERSONAL, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>个人版前端文件未找到</h1>", status_code=404)


@app.get("/public", response_class=HTMLResponse)
async def serve_public_index():
    """公开版 — 用户自行配置API Key"""
    index_path = os.path.join(FRONTEND_PUBLIC, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>公开版前端文件未找到</h1>", status_code=404)


@app.get("/{filename:path}")
async def serve_static(filename: str):
    """静态文件服务"""
    # 按优先级查找: 主目录 → 个人版 → 公开版
    for base in [FRONTEND_MAIN, FRONTEND_PERSONAL, FRONTEND_PUBLIC]:
        file_path = os.path.join(base, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)

    raise HTTPException(status_code=404, detail="文件未找到")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║         UMU刷题助手 - 后端服务 v{APP_VERSION}                  ║
╠══════════════════════════════════════════════════════════╣
║  个人版:  http://{host}:{port}/                        ║
║  公开版:  http://{host}:{port}/public                  ║
║  API文档: http://{host}:{port}/api/docs                ║
║  健康检查: http://{host}:{port}/api/health             ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
