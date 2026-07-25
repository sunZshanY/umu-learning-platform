"""
AI答题服务 — 调用大语言模型API进行智能答题
支持：DeepSeek / OpenAI / 兼容OpenAI格式的API
内置重试、超时、降级、错误恢复机制
"""
import asyncio
import time
import logging
from typing import Optional, Tuple
from openai import AsyncOpenAI, OpenAIError, APITimeoutError, APIConnectionError, RateLimitError

from models import QuestionType, AnswerResponse

logger = logging.getLogger(__name__)

# ── 配置常量 ─────────────────────────────────────────────────
MAX_RETRIES = 3                    # 最大重试次数
RETRY_BASE_DELAY = 1.0            # 重试基础延迟(秒)
RETRY_MAX_DELAY = 8.0             # 重试最大延迟(秒)
REQUEST_TIMEOUT = 15.0            # 单次请求超时(秒)
TEMPERATURE = 0.0                 # 模型温度（0=更确定性）
MAX_TOKENS = 1024                 # 最大输出token


class AIService:
    """
    AI答题服务核心类
    — 封装模型调用、重试逻辑、结果解析
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ):
        """
        初始化AI服务

        Args:
            api_key: API密钥
            base_url: API基础URL (默认DeepSeek)
            model: 模型名称
        """
        if not api_key or not api_key.strip():
            raise ValueError("API Key不能为空")

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Optional[AsyncOpenAI] = None
        self._total_requests: int = 0
        self._total_failures: int = 0
        self._total_retries: int = 0

    @property
    def client(self) -> AsyncOpenAI:
        """延迟初始化OpenAI客户端"""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=REQUEST_TIMEOUT,
                max_retries=0,  # 我们自己管理重试
            )
        return self._client

    @property
    def stats(self) -> dict:
        """获取服务统计信息"""
        return {
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "total_retries": self._total_retries,
            "success_rate": (
                (self._total_requests - self._total_failures) / self._total_requests
                if self._total_requests > 0
                else 1.0
            ),
        }

    # ── 构建Prompt ───────────────────────────────────────────
    @staticmethod
    def _build_system_prompt() -> str:
        """构建系统提示词 — 高准确率答题版"""
        return """你是严谨的考试答题专家。你必须先在内部完成审题、排除、复核，再只输出最终答案。

答题流程（内部完成，不要输出过程）：
1. 先判断题型：单选、多选、判断、填空、简答。
2. 逐字检查否定词、限定词和例外：不、不是、错误、无关、不能、除外、至少、全部、均、一定、最、首要。
3. 对选择题逐项排除，确认选项字母与内容对应，不要只凭印象选。
4. 对多选题必须找出所有正确项，不能漏选；不确定时宁可保守，不选明显可疑项。
5. 对判断题只输出“正确”或“错误”。
6. 对计算、日期、概念题要复核单位、年份、公式和常识冲突。
7. 如果题目含“根据材料/上文/视频/课程内容”，优先使用题目给出的上下文，不要自行补充无关内容。

输出格式（严格遵守）：
- 单选题：字母+选项内容，如“B. 北京”
- 多选题：连续字母+选项内容，如“ABD. 光合作用、呼吸作用、蒸腾作用”
- 判断题：仅“正确”或“错误”
- 填空题：仅填空内容，不超过30字
- 简答题：仅核心答案，不超过80字

禁止输出：
- Markdown 标记、解释过程、解析、置信度、前缀“答案:”
- “我认为”“可能”“无法确定”等废话
- 与选项无关的扩写内容

如果题目或选项不完整：仍给出最可能的最终答案，但只输出答案本身。"""

    @staticmethod
    def _build_user_prompt(
        question: str,
        question_type: QuestionType,
        options: Optional[list] = None,
        context: Optional[str] = None,
    ) -> str:
        """构建用户提示词"""
        parts = []

        if context:
            parts.append(f"【上下文】{context}")

        type_map = {
            QuestionType.SINGLE_CHOICE: "单选题（只有一个正确答案）",
            QuestionType.MULTI_CHOICE: "多选题（有多个正确答案）",
            QuestionType.TRUE_FALSE: "判断题（正确或错误）",
            QuestionType.FILL_BLANK: "填空题",
            QuestionType.SHORT_ANSWER: "简答题",
            QuestionType.UNKNOWN: "未知类型（请自动判断）",
        }
        parts.append(f"【题目类型】{type_map.get(question_type, '未知')}")
        parts.append(f"【题目内容】\n{question}")

        if options:
            opts_text = "\n".join(options)
            parts.append(f"【选项列表】\n{opts_text}")
            if question_type == QuestionType.SINGLE_CHOICE:
                parts.append("【作答要求】只能选择一个选项，答案必须以选项字母开头。")
            elif question_type == QuestionType.MULTI_CHOICE:
                parts.append("【作答要求】选择所有正确选项，答案必须以多个选项字母开头，字母按选项顺序排列。")
        elif question_type == QuestionType.TRUE_FALSE:
            parts.append("【作答要求】只输出“正确”或“错误”。")

        parts.append("【最终复核】输出前再次核对题干是否问的是“正确”还是“错误/不正确/不属于”。只输出最终答案。")

        return "\n\n".join(parts)

    # ── 结果解析 ─────────────────────────────────────────────
    @staticmethod
    def _parse_response(
        text: str,
        question_type: QuestionType = QuestionType.UNKNOWN,
        options: Optional[list] = None,
    ) -> Tuple[str, str, float]:
        """解析AI返回的文本 — 新格式为纯文本答案，无结构化标签"""
        raw = text.strip()

        # 清洗Markdown残留
        import re
        # 移除粗体/斜体标记
        raw = re.sub(r'\*\*(.+?)\*\*', r'\1', raw)
        raw = re.sub(r'\*(.+?)\*', r'\1', raw)
        raw = re.sub(r'__(.+?)__', r'\1', raw)
        raw = re.sub(r'_(.+?)_', r'\1', raw)
        # 移除行内代码标记
        raw = re.sub(r'`(.+?)`', r'\1', raw)
        # 移除标题标记
        raw = re.sub(r'^#{1,6}\s+', '', raw, flags=re.MULTILINE)
        # 移除"答案:"/"解析:"等前缀（兼容旧格式）
        raw = re.sub(r'^(答案|解析|置信度)[：:]\s*', '', raw, flags=re.MULTILINE)
        # 合并多余空白
        raw = re.sub(r'\n{3,}', '\n\n', raw)
        raw = raw.strip()

        # 取第一有效行作为答案（多行时只保留核心答案）
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        if question_type == QuestionType.MULTI_CHOICE:
            answer = ' '.join(lines) if lines else raw[:200]
        else:
            answer = lines[0] if lines else raw[:200]

        # 解释取剩余内容（最多2行）
        remaining = lines[1:] if len(lines) > 1 else []
        explanation = '; '.join(remaining[:2]) if remaining else ''

        # 默认置信度
        confidence = 0.88

        # 规范化选择题答案，减少模型输出解释或漏掉选项内容。
        if options and question_type in (QuestionType.SINGLE_CHOICE, QuestionType.MULTI_CHOICE):
            option_map = {}
            for opt in options:
                match = re.match(r'^\s*([A-Ha-h])\s*[\.、\)）:：]\s*(.+?)\s*$', str(opt))
                if match:
                    option_map[match.group(1).upper()] = match.group(2).strip()
            found = re.findall(r'\b([A-Ha-h])\b|(?<![A-Za-z])([A-Ha-h])(?=[\.、\)）:：])', answer)
            letters = ''.join(dict.fromkeys((a or b).upper() for a, b in found if (a or b)))
            if not letters:
                compact = answer.replace(' ', '')
                letters = ''.join(letter for letter, text in option_map.items() if text and text.replace(' ', '') in compact)
            if letters:
                if question_type == QuestionType.SINGLE_CHOICE:
                    letters = letters[0]
                ordered = ''.join(letter for letter in option_map if letter in set(letters))
                labels = '、'.join(option_map[letter] for letter in ordered if letter in option_map)
                answer = f"{ordered}. {labels}" if labels else ordered

        return answer, explanation, confidence

    # ── 核心调用方法 ─────────────────────────────────────────
    async def answer_question(
        self,
        question: str,
        question_type: QuestionType = QuestionType.UNKNOWN,
        options: Optional[list] = None,
        context: Optional[str] = None,
    ) -> AnswerResponse:
        """
        对单道题目进行AI作答

        Args:
            question: 题目文本
            question_type: 题目类型
            options: 选项列表
            context: 额外上下文

        Returns:
            AnswerResponse: 包含答案、解析和置信度
        """
        self._total_requests += 1
        start_time = time.monotonic()
        retry_count = 0

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, question_type, options, context)

        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                raw_text = response.choices[0].message.content or ""
                answer, explanation, confidence = self._parse_response(
                    raw_text,
                    question_type=question_type,
                    options=options,
                )

                elapsed = (time.monotonic() - start_time) * 1000
                logger.info(
                    f"AI答题成功 | 耗时: {elapsed:.0f}ms | "
                    f"置信度: {confidence:.2f} | 重试: {retry_count}"
                )

                return AnswerResponse(
                    success=True,
                    answer=answer,
                    explanation=explanation,
                    confidence=confidence,
                    question_type=question_type,
                    retry_count=retry_count,
                )

            except (APITimeoutError, APIConnectionError) as e:
                retry_count = attempt
                last_error = f"网络超时/连接失败: {str(e)[:100]}"
                logger.warning(f"AI调用网络错误 (尝试{attempt+1}/{MAX_RETRIES+1}): {last_error}")
                if attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                    self._total_retries += 1

            except RateLimitError as e:
                retry_count = attempt
                last_error = f"API速率限制: {str(e)[:100]}"
                logger.warning(f"AI调用限流 (尝试{attempt+1}/{MAX_RETRIES+1})")
                if attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** (attempt + 1)), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                    self._total_retries += 1

            except OpenAIError as e:
                retry_count = attempt
                last_error = f"API错误: {str(e)[:150]}"
                logger.error(f"AI调用失败 (尝试{attempt+1}/{MAX_RETRIES+1}): {last_error}")
                if attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                    self._total_retries += 1

            except Exception as e:
                retry_count = attempt
                last_error = f"未知错误: {str(e)[:150]}"
                logger.exception(f"AI调用异常 (尝试{attempt+1}/{MAX_RETRIES+1})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BASE_DELAY)
                    self._total_retries += 1

        # ── 所有重试均失败 ──────────────────────────────────
        self._total_failures += 1
        elapsed = (time.monotonic() - start_time) * 1000
        logger.error(f"AI答题彻底失败 | 耗时: {elapsed:.0f}ms | 重试: {retry_count}")

        return AnswerResponse(
            success=False,
            answer="",
            explanation=None,
            confidence=0.0,
            question_type=question_type,
            error=f"AI服务调用失败（已重试{MAX_RETRIES}次）: {last_error}",
            retry_count=retry_count,
        )

    async def answer_batch(
        self,
        questions: list,
        concurrency: int = 15,
    ) -> list:
        """
        批量答题 — 并发处理多道题目

        Args:
            questions: 题目列表 (AnswerRequest)
            concurrency: 并发数

        Returns:
            AnswerResponse列表
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _answer_one(q):
            async with semaphore:
                return await self.answer_question(
                    question=q.question,
                    question_type=q.question_type,
                    options=q.options,
                    context=q.context,
                )

        tasks = [_answer_one(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        handled_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"批量答题第{i+1}题异常: {r}")
                handled_results.append(AnswerResponse(
                    success=False,
                    answer="",
                    error=f"并发处理异常: {str(r)[:100]}",
                    question_type=questions[i].question_type,
                ))
            else:
                handled_results.append(r)

        return handled_results

    async def test_connection(self) -> Tuple[bool, str]:
        """测试API连接是否正常"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "回复'OK'，不要其他内容。"}
                ],
                temperature=0.0,
                max_tokens=10,
            )
            text = response.choices[0].message.content or ""
            return "OK" in text or "ok" in text.lower(), "连接正常"
        except Exception as e:
            return False, f"连接失败: {str(e)[:150]}"
