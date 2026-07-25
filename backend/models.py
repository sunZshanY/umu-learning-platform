"""
数据模型定义 — UMU刷题助手
所有请求/响应模型集中管理，确保类型安全与数据校验。
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from enum import Enum


# ── 题目类型枚举 ─────────────────────────────────────────────
class QuestionType(str, Enum):
    SINGLE_CHOICE = "single_choice"     # 单选题
    MULTI_CHOICE = "multi_choice"       # 多选题
    TRUE_FALSE = "true_false"           # 判断题
    FILL_BLANK = "fill_blank"           # 填空题
    SHORT_ANSWER = "short_answer"       # 简答题
    UNKNOWN = "unknown"                 # 未知类型


# ── 请求模型 ────────────────────────────────────────────────
class AnswerRequest(BaseModel):
    """单题作答请求"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="题目文本内容",
        examples=["以下哪个是中国的首都？A. 上海 B. 北京 C. 广州 D. 深圳"]
    )
    question_type: QuestionType = Field(
        default=QuestionType.UNKNOWN,
        description="题目类型，未知时由AI自动判断"
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="选项列表，如 ['A. 上海', 'B. 北京', 'C. 广州', 'D. 深圳']",
        examples=[["A. 上海", "B. 北京", "C. 广州", "D. 深圳"]]
    )
    context: Optional[str] = Field(
        default=None,
        max_length=3000,
        description="额外的上下文信息（如课程名称、知识点等）"
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("题目内容不能为空")
        return stripped


class BatchAnswerRequest(BaseModel):
    """批量答题请求"""
    questions: List[AnswerRequest] = Field(
        ...,
        min_length=1,
        max_length=300,
        description="题目列表，单次最多300题"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="UMU会话ID，用于追踪答题上下文"
    )


class QuestionOption(BaseModel):
    """解析出的选项"""
    label: str = Field(..., min_length=1, max_length=8, description="选项标识，如 A")
    text: str = Field(..., min_length=1, max_length=1000, description="选项内容")


class ParsedQuestion(BaseModel):
    """自动提取出的结构化题目"""
    id: str = Field(..., description="前端展示用题目ID")
    question: str = Field(..., description="题干文本")
    question_type: QuestionType = Field(default=QuestionType.UNKNOWN, description="推断题型")
    options: List[str] = Field(default_factory=list, description="选项列表，如 ['A. 选项']")
    raw_text: str = Field(default="", description="原始题块")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="解析置信度")
    warnings: List[str] = Field(default_factory=list, description="需要用户检查的问题")


class ParseQuestionsRequest(BaseModel):
    """自动提取题目请求"""
    raw_text: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="复制的整页文本或题目文本"
    )
    source: str = Field(default="paste", max_length=30, description="文本来源")
    options: dict[str, Any] = Field(default_factory=dict, description="解析选项")

    @field_validator("raw_text")
    @classmethod
    def raw_text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("题目文本不能为空")
        return stripped


class ParseQuestionsResponse(BaseModel):
    """自动提取题目响应"""
    success: bool = Field(default=True, description="是否成功解析请求")
    questions: List[ParsedQuestion] = Field(default_factory=list, description="解析出的题目")
    meta: dict[str, Any] = Field(default_factory=dict, description="解析统计和提示")


class UmuSessionRequest(BaseModel):
    """UMU会话解析请求"""
    session_url: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="UMU答题会话URL",
        examples=["https://m.umu.cn/session/quiz/4enVY3211?fwx=1"]
    )
    session_token: Optional[str] = Field(
        default=None,
        description="UMU登录后的session token（如需认证）"
    )


class FeedbackRequest(BaseModel):
    """用户反馈请求 — 用于调优"""
    question: str = Field(..., description="题目原文")
    ai_answer: str = Field(..., description="AI给出的答案")
    correct_answer: Optional[str] = Field(
        default=None,
        description="正确答案（用户手动填入）"
    )
    was_correct: bool = Field(
        default=False,
        description="AI答案是否正确"
    )


# ── 响应模型 ────────────────────────────────────────────────
class AnswerResponse(BaseModel):
    """单题作答响应"""
    success: bool = Field(..., description="是否成功获取答案")
    answer: str = Field(default="", description="AI给出的答案文本")
    explanation: Optional[str] = Field(
        default=None,
        description="答案解析与推理过程"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="AI对答案的置信度 (0.0-1.0)"
    )
    question_type: QuestionType = Field(
        default=QuestionType.UNKNOWN,
        description="AI识别出的题目类型"
    )
    error: Optional[str] = Field(
        default=None,
        description="错误信息（仅success=false时有值）"
    )
    retry_count: int = Field(
        default=0,
        description="本次请求重试次数"
    )


class BatchAnswerResponse(BaseModel):
    """批量答题响应"""
    success: bool = Field(..., description="整体是否成功")
    total: int = Field(..., description="总题数")
    answered: int = Field(..., description="成功答题数")
    failed: int = Field(..., description="失败题数")
    results: List[AnswerResponse] = Field(
        default_factory=list,
        description="每题作答结果"
    )
    total_time_ms: float = Field(
        default=0.0,
        description="总耗时（毫秒）"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    api_connected: bool = Field(default=False)
    uptime_seconds: float = Field(default=0.0)
