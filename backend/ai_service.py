"""
AI答题服务 — 调用大语言模型API进行智能答题
支持：DeepSeek / OpenAI / 兼容OpenAI格式的API
内置重试、超时、降级、错误恢复、双重验证机制
"""
import asyncio
import time
import logging
import re
from typing import Optional, Tuple

from openai import AsyncOpenAI, OpenAIError, APITimeoutError, APIConnectionError, RateLimitError

from models import QuestionType, AnswerResponse

logger = logging.getLogger(__name__)

# ── 配置常量 ─────────────────────────────────────────────────
MAX_RETRIES = 3                    # 最大重试次数
RETRY_BASE_DELAY = 1.0            # 重试基础延迟(秒)
RETRY_MAX_DELAY = 8.0             # 重试最大延迟(秒)
REQUEST_TIMEOUT = 30.0            # 单次请求超时(秒) — 提升以应对复杂推理
TEMPERATURE = 0.0                 # 模型温度（0=最确定性）
MAX_TOKENS = 2048                 # 最大输出token — 提升以容纳推理链+答案
VERIFY_THRESHOLD = 0.80           # 低于此置信度触发二次验证


class AIService:
    """
    AI答题服务核心类
    — 封装模型调用、重试逻辑、结果解析、双重验证
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ):
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
    def temperature(self) -> float:
        return TEMPERATURE

    @property
    def client(self) -> AsyncOpenAI:
        """延迟初始化OpenAI客户端"""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=REQUEST_TIMEOUT,
                max_retries=0,
            )
        return self._client

    @property
    def stats(self) -> dict:
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

    # ═════════════════════════════════════════════════════════════
    # 系统提示词 — 精装修版（Few-shot + CoT + 自检）
    # ═════════════════════════════════════════════════════════════
    @staticmethod
    def _build_system_prompt() -> str:
        return """你是国家级标准化考试命题专家，拥有各学科博士学位。你的目标是100%准确率。

【核心答题协议 — 必须逐条执行】
第1步·破题：提取题干的1-2个核心关键词，判定本题考察的知识点或逻辑关系。
第2步·审问：明确题干问的是什么——「正确的是/错误的是」「属于/不属于」「包括/不包括」。用笔圈出否定词：不、非、无、未、否、除、只、仅。
第3步·析选项：
  - 单选题：逐项与题干核心关键词比对，先排除明显错误项，再从剩余中选最优。
    * 「以上都是」选项：必须逐一验证A/B/C是否都正确，全部正确才选D。不可只看到第一个正确的就选。
    * 「不属于/错误的是」：用排除法，先标记所有正确项，剩下的那个就是答案。
    * 术语辨析：注意相似概念的区别（如「外观介绍」vs「文案描述」、「曝光量」vs「点击率」）。
  - 多选题：独立评判每个选项（假设其他选项不存在），对则留、错则弃、疑则弃。
  - 判断题：先忽略题干的否定词判断陈述本身的真假，再根据题干问法决定输出「正确」还是「错误」。
第4步·验答：答案与题干再次比对——我选的是题干问的那个方向吗？我有没有把「选错误的」当成「选正确的」？
第5步·输出：严格按格式输出，一字不多。

【输出格式 — 违反格式=答错】
单选题 → "<字母>. <选项内容>"              示例: C. 线粒体
多选题 → "<字母序列>. <选项内容>"          示例: ABD. 光合作用、呼吸作用、蒸腾作用
                                           字母按A→Z顺序排列，不可打乱
判断题 → "正确" 或 "错误"                 仅此两字，不加任何标点
填空题 → 填空内容                         不超过30字，只写答案本身
简答题 → 核心答案                         不超过80字

【高频错误及防御策略】
错误1「审反」：题干问「不属于/错误的是」，你选了属于/正确的。
  → 防御：输出前强制检查题干最后一个问句的否定词，反向确认答案。
错误2「漏选」：多选题少选了正确选项。
  → 防御：每个选项独立评估，选项A对吗？选项B对吗？...不要横向比较。
错误3「多选」：多选题选了错误选项。
  → 防御：对任何选项有10%以上怀疑就坚决不选。保守比冒进好。
错误4「看串行」：把选项B的内容当成选项A的。
  → 防御：每个选项连字母带内容一起读，确认字母-内容映射无误。
错误5「想当然」：凭常识选而不看题目具体表述。
  → 防御：答案必须严格基于题目给出的信息，不要引入题目未提供的外部知识。

【判断题决策树】
1. 陈述本身是真还是假？（忽略题干的问法）
2. 题干最终问的是「正确/是」还是「错误/否」？
3. 如果问「正确」且陈述为真 → 输出「正确」
4. 如果问「正确」且陈述为假 → 输出「错误」
5. 如果问「错误」且陈述为真 → 输出「错误」
6. 如果问「错误」且陈述为假 → 输出「正确」
核心：陈述真假 ≠ 最终答案，最终答案取决于题干问法！

【多选题决策树】
1. 逐一检查每个选项，独立判断其正确性
2. 确定正确的选项数量（至少1个）
3. 只选100%确定正确的，有疑虑的不选
4. 按字母顺序排列输出

【常见学科陷阱识别】
- 政治/法律：注意「根本」「基本」「核心」「首要」等限定词
- 历史：注意年代、人物、事件的精确对应
- 地理：注意方位、气候类型、地形特征
- 生物/化学：注意专业术语的精确表述
- 物理/数学：注意单位、符号、公式的适用条件
- 计算机：注意术语定义、协议层级、算法特征

【最终自检 — 输出前必须回答】
1. 题干问的是「正确」方向还是「错误」方向？我的答案方向对吗？
2. 如果是选择题，我的字母和选项内容正确对应吗？
3. 如果是多选题，我有没有漏掉正确项？有没有多选错误项？
4. 如果是判断题，我是否按决策树走了三步？"""

    # ═════════════════════════════════════════════════════════════
    # 用户提示词构建 — 结构化输入
    # ═════════════════════════════════════════════════════════════
    @staticmethod
    def _build_user_prompt(
        question: str,
        question_type: QuestionType,
        options: Optional[list] = None,
        context: Optional[str] = None,
        is_verify: bool = False,
        first_answer: str = "",
        is_tf_converted: bool = False,
    ) -> str:
        parts = []

        if context:
            parts.append(f"【背景上下文】{context}")

        type_map = {
            QuestionType.SINGLE_CHOICE: "单选题 — 有且仅有一个正确答案。排除法+关键词匹配。",
            QuestionType.MULTI_CHOICE: "多选题 — 每个选项独立评判，对的留、错的弃、疑的弃。",
            QuestionType.TRUE_FALSE: "判断题 — 先判陈述真假，再看题干问法（正确/错误），按决策树三步走。",
            QuestionType.FILL_BLANK: "填空题 — 仅输出填写内容，注意单位、术语标准化。",
            QuestionType.SHORT_ANSWER: "简答题 — 输出核心答案，不超过80字。",
            QuestionType.UNKNOWN: "未知类型 — 先根据选项数量和题干特征判断题型，再作答。",
        }
        parts.append(f"【题型】{type_map.get(question_type, '未知')}")

        # ── 题干标注：否定词 + 关键限定词高亮 ──
        highlighted = question
        negations = [
            "不是", "不正确", "不属于", "错误的是", "不正确的是", "错误的说法",
            "无关", "不能", "不包括", "不包含", "没有", "无", "非", "除外",
            "不正确的选项", "错误的选项", "不能体现", "无法说明", "不恰当",
            "错误的是", "不正确的是", "不符合", "不涉及", "不能解释",
        ]
        qualifiers = [
            "至少", "全部", "均", "一定", "最", "首要", "主要", "根本",
            "核心", "基本", "唯一", "必须", "绝对", "完全", "任何",
            "从不", "始终", "总是",
        ]
        for word in negations + qualifiers:
            if word in highlighted:
                highlighted = highlighted.replace(word, f"⚠️【{word}】⚠️")

        # ── 检测题干最终问法 ──
        # ── 题干方向检测 ──
        is_negation = any(w in question for w in ["不属于", "错误的是", "不正确", "错误的", "不包括", "无关", "不能"])
        direction_hint = ""
        if is_negation:
            direction_hint = "\n\n🔴【否定方向】本题问的是「不属于/错误的是」→ 用排除法：先找出所有正确/属于的选项，剩下的就是要选的答案。"
        elif any(w in question for w in ["属于", "正确的是", "包括", "有关"]):
            direction_hint = "\n\n🟢【肯定方向】本题问的是「属于/正确的是」→ 直接选出正确选项。"

        # ── "以上都是" 检测 ──
        has_all_above = options and any(re.match(r'^\s*[A-Ja-j]\s*[\.、]?\s*以上都是', str(o)) for o in options)

        parts.append(f"【题目】\n{highlighted}{direction_hint}")

        # ── 选项 ──
        if options:
            opts_text = "\n".join(options)
            parts.append(f"【选项】\n{opts_text}")

            # 单选题：提示对比相似选项
            if question_type == QuestionType.SINGLE_CHOICE and len(options) >= 3:
                if has_all_above:
                    parts.append("【⚠️ 「以上都是」选项】逐一检查A/B/C是否都正确，全部正确才选「以上都是」。不可看到一个正确项就直接选。")
                if is_negation:
                    parts.append("【排除法】题干问「不属于」→ 先标记哪些选项是正确的/属于的，排除它们，剩下的就是答案。不可直接选第一个看起来不对的。")
                parts.append("【排除策略】逐项读每个选项，找出与题干关键词最匹配的一项。辨析相似术语（如外观介绍≠文案描述，曝光量≠点击率）。")
            elif question_type == QuestionType.MULTI_CHOICE:
                parts.append("【多选策略】独立评判每个选项（A对吗？B对吗？...），确定正确的全部选出，不确定的不选。按ABCD顺序排列。")

        # ── 题型指令 ──
        if question_type == QuestionType.SINGLE_CHOICE:
            parts.append("【指令】输出1个字母+选项内容。格式: X. 选项内容")
        elif question_type == QuestionType.MULTI_CHOICE:
            parts.append("【指令】输出所有正确选项字母+内容，字母按顺序排列。格式: XYZ. 选项1、选项2、选项3")
        elif question_type == QuestionType.TRUE_FALSE:
            parts.append("【指令】按决策树三步走：①陈述真假？②题干问法？③定答案。只输出「正确」或「错误」。")
            if is_tf_converted:
                parts.append("【⚠️ 判断题专项 — 逐条验证，不预设答案】")
                parts.append("请执行以下验证流程（不要凭感觉，要逐条核实）：")
                parts.append("1. 拆解：将题干陈述拆成1-3个关键条件，逐一标注。")
                parts.append("2. 验证每个条件：")
                parts.append("   - 电商平台实际操作中是这样做吗？（从买家/卖家/平台角度想）")
                parts.append("   - 这个说法是否过于绝对？是否有例外情况？")
                parts.append("   - 如果有数据/工具/功能能支撑这个说法 → 倾向「正确」")
                parts.append("3. 综合判断：所有条件都成立 → 「正确」；任一条件不成立 → 「错误」。")
                parts.append("4. 特别注意：不要因为题目是「培训考核题」就预设答案为错——")
                parts.append("   正确的规则陈述同样是常见的考点。凭逻辑和常识判断，不为反而反。")
            else:
                parts.append("【⚠️ 批判性提醒】逐条验证题干条件后再判断，不要预设答案。")

        # ── 二次验证 ──
        if is_verify and first_answer:
            parts.append(f"\n⚠️【复核模式】你之前的答案是: {first_answer}")
            parts.append("请你以最严厉的批判性思维重新审题：")
            parts.append("1. 之前的答案是否真的正确？题干有无否定词被忽略？")
            parts.append("2. 陈述中的每个条件都100%成立吗？有无反例？")
            parts.append("3. 如果是平台规则/行业规范类题目，规则是否比直觉更严格？")
            parts.append("4. 常见的错误是「默认同意」——请刻意寻找陈述中的漏洞。")
            parts.append("若确认正确→输出相同答案。若发现错误→输出修正后的答案。")

        parts.append("\n【最终指令】只输出答案本身。不要推理过程，不要解释，不要括号注释，不要Markdown。")
        return "\n\n".join(parts)

    # ═════════════════════════════════════════════════════════════
    # 答案解析 — 鲁棒提取
    # ═════════════════════════════════════════════════════════════
    @staticmethod
    def _parse_response(
        text: str,
        question_type: QuestionType = QuestionType.UNKNOWN,
        options: Optional[list] = None,
    ) -> Tuple[str, str, float]:
        """解析AI返回文本，提取干净答案"""
        raw = text.strip()

        # ── 清洗Markdown和格式残留 ──
        raw = re.sub(r'\*\*(.+?)\*\*', r'\1', raw)
        raw = re.sub(r'\*(.+?)\*', r'\1', raw)
        raw = re.sub(r'__(.+?)__', r'\1', raw)
        raw = re.sub(r'_(.+?)_', r'\1', raw)
        raw = re.sub(r'`(.+?)`', r'\1', raw)
        raw = re.sub(r'^#{1,6}\s+', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'^(答案|解析|置信度|正确选项|正确答案)[：:]\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'^(我的答案是?|最终答案[：:]?|选择)\s*[：:]?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'^[\[【].*?[\]】]\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\n{3,}', '\n\n', raw)
        raw = raw.strip()

        # ── 判断题特殊处理 ──
        if question_type == QuestionType.TRUE_FALSE:
            # 精确匹配"正确"/"错误"
            for line in raw.split('\n'):
                line = line.strip()
                if line in ("正确", "错误"):
                    return line, "", 0.95
            # 模糊匹配
            if "正确" in raw[:10] and "错误" not in raw[:10]:
                return "正确", "", 0.85
            if "错误" in raw[:10] and "正确" not in raw[:10]:
                return "错误", "", 0.85
            # fallback
            first_line = raw.split('\n')[0].strip()
            return first_line[:10], "", 0.70

        # ── 构建选项映射 ──
        option_map = {}
        if options:
            for opt in options:
                m = re.match(r'^\s*([A-Ja-j])\s*[\.、\)）:：]\s*(.+?)\s*$', str(opt))
                if m:
                    option_map[m.group(1).upper()] = m.group(2).strip()

        # ── 提取答案行 ──
        lines = [l.strip() for l in raw.split('\n') if l.strip()]

        # 尝试找到以字母开头的答案行
        answer_line = None
        answer_line_idx = -1
        for i, line in enumerate(lines):
            # 匹配以字母.开头（如 "A. xxx" 或 "ABD. xxx"）
            if re.match(r'^[A-Ja-j]{1,8}[\.、\)）]\s', line):
                answer_line = line
                answer_line_idx = i
                break
            # 匹配纯字母（如 "A" 或 "ABD"）
            if re.match(r'^[A-Ja-j]{1,8}$', line):
                answer_line = line
                answer_line_idx = i
                break

        # ── 从答案行提取字母 ──
        if answer_line:
            # 提取首部字母序列
            letter_match = re.match(r'^([A-Ja-j]{1,8})', answer_line)
            if letter_match:
                letters_raw = letter_match.group(1).upper()
            else:
                letters_raw = ""
        else:
            # 从整个文本中找字母
            all_letters = re.findall(r'\b([A-J])\b', raw, re.IGNORECASE)
            letters_raw = ''.join(dict.fromkeys(l.upper() for l in all_letters))

        # ── 验证和修正字母 ──
        if letters_raw and option_map:
            valid_letters = ''.join(l for l in letters_raw if l in option_map)

            if question_type == QuestionType.SINGLE_CHOICE:
                # 单选题：只保留第一个有效字母
                valid_letters = valid_letters[0] if valid_letters else ""
                # 如果提取了多个字母但题型是单选，只取第一个
                if len(letters_raw) > 1 and valid_letters:
                    logger.warning(f"单选题检测到多个字母 '{letters_raw}'，仅保留 '{valid_letters}'")
            elif question_type == QuestionType.MULTI_CHOICE:
                # 多选题：排序字母
                valid_letters = ''.join(sorted(set(valid_letters)))

            if valid_letters:
                # 构建最终答案：字母 + 选项内容
                labels = '、'.join(
                    option_map[l] for l in valid_letters if l in option_map
                )
                if labels:
                    answer = f"{valid_letters}. {labels}"
                else:
                    answer = valid_letters
                confidence = 0.92
            else:
                # 字母与选项不匹配，回退到文本答案
                answer = lines[0] if lines else raw[:200]
                confidence = 0.72
        elif not letters_raw and options and question_type in (
            QuestionType.SINGLE_CHOICE, QuestionType.MULTI_CHOICE
        ):
            # 没找到字母，尝试内容匹配
            compact_answer = (lines[0] if lines else raw[:200]).replace(' ', '')
            matched = []
            for letter, text in option_map.items():
                if text.replace(' ', '') in compact_answer:
                    matched.append(letter)
            if matched:
                if question_type == QuestionType.SINGLE_CHOICE:
                    matched = matched[:1]
                labels = '、'.join(option_map[l] for l in matched if l in option_map)
                answer = f"{''.join(matched)}. {labels}" if labels else ''.join(matched)
                confidence = 0.82
            else:
                answer = lines[0] if lines else raw[:200]
                confidence = 0.65
        else:
            # 非选择题或无需选项映射
            if question_type == QuestionType.MULTI_CHOICE:
                answer = ' '.join(lines[:2]) if lines else raw[:300]
            else:
                answer = lines[0] if lines else raw[:200]
            confidence = 0.85

        # ── 解释文本 ──
        exp_start = answer_line_idx + 1 if answer_line_idx >= 0 else 1
        remaining = lines[exp_start:]
        explanation = '; '.join(remaining[:2]) if remaining else ''

        # 清理解释中的格式残留
        explanation = re.sub(r'\*\*|__|##|`', '', explanation)

        return answer, explanation, min(confidence, 0.99)

    # ═════════════════════════════════════════════════════════════
    # 答案验证
    # ═════════════════════════════════════════════════════════════
    @staticmethod
    def _validate_answer(
        answer: str,
        question_type: QuestionType,
        options: Optional[list] = None,
    ) -> Tuple[bool, str]:
        """
        验证答案是否合理，返回 (is_valid, issue_description)
        """
        if not answer or not answer.strip():
            return False, "答案为空"

        # 判断题验证
        if question_type == QuestionType.TRUE_FALSE:
            if answer.strip() not in ("正确", "错误"):
                return False, f"判断题答案格式异常: {answer[:30]}"
            return True, ""

        # 选择题验证
        if options and question_type in (QuestionType.SINGLE_CHOICE, QuestionType.MULTI_CHOICE):
            # 提取答案中的字母
            letters = re.findall(r'^([A-Ja-j]{1,8})', answer.strip())
            if not letters:
                return False, f"未提取到选项字母: {answer[:50]}"

            opt_letters = set()
            for opt in options:
                m = re.match(r'^\s*([A-Ja-j])', str(opt))
                if m:
                    opt_letters.add(m.group(1).upper())

            ans_letters = set(letters[0].upper())

            # 检查字母是否都在选项中
            unknown = ans_letters - opt_letters
            if unknown:
                return False, f"答案含无效选项: {unknown}"

            # 单选题检查
            if question_type == QuestionType.SINGLE_CHOICE and len(ans_letters) > 1:
                return False, f"单选题出现多个选项: {ans_letters}"

            # 多选题检查
            if question_type == QuestionType.MULTI_CHOICE and len(ans_letters) < 1:
                return False, "多选题至少需要一个选项"

        return True, ""

    # ═════════════════════════════════════════════════════════════
    # 核心调用 — 单题作答
    # ═════════════════════════════════════════════════════════════
    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        底层API调用，含重试逻辑。
        Returns: (raw_text, error_message)
        """
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=MAX_TOKENS,
                    top_p=0.95,
                )
                return response.choices[0].message.content or "", None

            except (APITimeoutError, APIConnectionError) as e:
                last_error = f"网络超时/连接失败: {str(e)[:100]}"
                logger.warning(f"AI调用网络错误 (尝试{attempt+1}/{MAX_RETRIES+1}): {last_error}")
                if attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                    self._total_retries += 1

            except RateLimitError as e:
                last_error = f"API速率限制: {str(e)[:100]}"
                logger.warning(f"AI调用限流 (尝试{attempt+1}/{MAX_RETRIES+1})")
                if attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** (attempt + 1)), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                    self._total_retries += 1

            except OpenAIError as e:
                last_error = f"API错误: {str(e)[:150]}"
                logger.error(f"AI调用失败 (尝试{attempt+1}/{MAX_RETRIES+1}): {last_error}")
                if attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
                    self._total_retries += 1

            except Exception as e:
                last_error = f"未知错误: {str(e)[:150]}"
                logger.exception(f"AI调用异常 (尝试{attempt+1}/{MAX_RETRIES+1})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BASE_DELAY)
                    self._total_retries += 1

        return None, last_error

    async def answer_question(
        self,
        question: str,
        question_type: QuestionType = QuestionType.UNKNOWN,
        options: Optional[list] = None,
        context: Optional[str] = None,
        enable_verify: bool = True,
    ) -> AnswerResponse:
        """
        对单道题目进行AI作答（含双重验证）

        Args:
            question: 题目文本
            question_type: 题目类型
            options: 选项列表
            context: 额外上下文
            enable_verify: 是否启用二次验证
        """
        self._total_requests += 1
        start_time = time.monotonic()
        retry_count = 0

        # ── 智能题型转换：对/错选项 → 判断题 ──
        effective_type = question_type
        tf_option_map = {}  # 记录对/错→选项字母的映射
        if options and question_type in (QuestionType.SINGLE_CHOICE, QuestionType.UNKNOWN):
            is_tf = False
            for opt in options:
                # 提取字母和内容：A. 对 → letter=A, content=对
                m = re.match(r'^\s*([A-Ja-j])\s*[\.、\)）:：]?\s*(.+?)\s*$', str(opt))
                if not m:
                    continue
                letter = m.group(1).upper()
                content = m.group(2).strip()
                # 判断内容是否是对/错/正确/错误等
                if re.match(r'^(对|正确|[√✅]|是|true|yes)$', content, re.IGNORECASE):
                    tf_option_map["correct"] = letter
                elif re.match(r'^(错|错误|[×x❌]|否|false|no)$', content, re.IGNORECASE):
                    tf_option_map["incorrect"] = letter
            if "correct" in tf_option_map and "incorrect" in tf_option_map:
                effective_type = QuestionType.TRUE_FALSE
                logger.info(
                    f"检测到对/错选项，自动转为判断题处理 | "
                    f"对→{tf_option_map.get('correct','?')} 错→{tf_option_map.get('incorrect','?')}"
                )

        is_tf_converted = bool(tf_option_map)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            question, effective_type, options, context,
            is_tf_converted=is_tf_converted,
        )

        # ── 第一次调用 ──
        raw_text, error = await self._call_api(system_prompt, user_prompt)

        if error:
            self._total_failures += 1
            elapsed = (time.monotonic() - start_time) * 1000
            logger.error(f"AI答题彻底失败 | 耗时: {elapsed:.0f}ms")
            return AnswerResponse(
                success=False,
                answer="",
                explanation=None,
                confidence=0.0,
                question_type=question_type,
                error=f"AI服务调用失败（已重试{MAX_RETRIES}次）: {error}",
                retry_count=retry_count,
            )

        # ── 解析第一次结果 ──
        answer, explanation, confidence = self._parse_response(
            raw_text, question_type=effective_type, options=options
        )

        # ── 验证答案合理性 ──
        is_valid, issue = self._validate_answer(answer, effective_type, options)
        if not is_valid:
            logger.warning(f"答案验证异常: {issue} | 原始输出: {raw_text[:100]}")
            confidence = min(confidence, 0.60)

        # ── 二次验证策略 ──
        # 多选和判断题错误率最高 → 强制验证
        # 单选/填空/简答 → 低置信度时验证
        force_verify = effective_type in (QuestionType.MULTI_CHOICE, QuestionType.TRUE_FALSE)
        need_verify = (
            not is_valid
            or confidence < 0.85
            or force_verify
        )

        if enable_verify and need_verify:
            reason = "强制验证" if force_verify else ("格式异常" if not is_valid else f"低置信度({confidence:.2f})")
            logger.info(
                f"触发二次验证 | {reason} | "
                f"有效: {is_valid} | 题型: {effective_type.value}"
            )
            self._total_retries += 1

            verify_prompt = self._build_user_prompt(
                question, effective_type, options, context,
                is_verify=True, first_answer=answer,
                is_tf_converted=is_tf_converted,
            )

            verify_text, verify_error = await self._call_api(system_prompt, verify_prompt)

            if verify_text and not verify_error:
                verify_answer, verify_explanation, verify_confidence = self._parse_response(
                    verify_text, question_type=effective_type, options=options
                )
                verify_valid, verify_issue = self._validate_answer(
                    verify_answer, effective_type, options
                )

                # 如果验证答案更合理，采用验证结果
                if verify_valid and (not is_valid or verify_confidence > confidence):
                    logger.info(
                        f"二次验证采用新答案 | {answer[:50]} → {verify_answer[:50]} | "
                        f"置信度: {confidence:.2f} → {verify_confidence:.2f}"
                    )
                    answer = verify_answer
                    explanation = verify_explanation
                    confidence = verify_confidence
                elif verify_valid and verify_answer.strip() == answer.strip():
                    # 答案一致，提升置信度
                    confidence = min(confidence + 0.08, 0.98)
                    logger.info(f"二次验证确认答案一致，置信度提升至 {confidence:.2f}")
                else:
                    logger.info(f"二次验证结果: {verify_answer[:50]} (保留原答案)")

        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(
            f"AI答题成功 | 耗时: {elapsed:.0f}ms | "
            f"置信度: {confidence:.2f} | 答案: {answer[:60]}"
        )

        # ── 答案回转换：判断题结果 → 对/错选项格式 ──
        if tf_option_map:
            if "正确" in answer and "correct" in tf_option_map:
                letter = tf_option_map["correct"]
                answer = f"{letter}. 对"
                logger.info(f"判断题结果转换: 正确 → {answer}")
            elif "错误" in answer and "incorrect" in tf_option_map:
                letter = tf_option_map["incorrect"]
                answer = f"{letter}. 错"
                logger.info(f"判断题结果转换: 错误 → {answer}")

        return AnswerResponse(
            success=True,
            answer=answer,
            explanation=explanation,
            confidence=confidence,
            question_type=question_type,
            retry_count=retry_count,
        )

    # ═════════════════════════════════════════════════════════════
    # 批量答题
    # ═════════════════════════════════════════════════════════════
    async def answer_batch(
        self,
        questions: list,
        concurrency: int = 15,
        enable_verify: bool = True,
    ) -> list:
        """
        批量答题 — 并发处理多道题目

        Args:
            questions: 题目列表 (AnswerRequest)
            concurrency: 并发数
            enable_verify: 是否启用二次验证
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _answer_one(q):
            async with semaphore:
                return await self.answer_question(
                    question=q.question,
                    question_type=q.question_type,
                    options=q.options,
                    context=q.context,
                    enable_verify=enable_verify,
                )

        tasks = [_answer_one(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

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

    # ═════════════════════════════════════════════════════════════
    # 连接测试
    # ═════════════════════════════════════════════════════════════
    async def test_connection(self) -> Tuple[bool, str]:
        """测试API连接是否正常"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "回复'OK'，不要其他内容。"}
                ],
                temperature=self.temperature,
                max_tokens=10,
            )
            text = response.choices[0].message.content or ""
            return "OK" in text or "ok" in text.lower(), "连接正常"
        except Exception as e:
            return False, f"连接失败: {str(e)[:150]}"
