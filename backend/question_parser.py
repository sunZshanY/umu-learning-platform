"""
题目解析器 — 从复制文本中提取结构化题目。

解析器只做确定性规则提取，不调用 AI，适合后端 API、CLI 和前端预览复用。
"""
import re
from dataclasses import dataclass, field
from typing import List

from models import QuestionType


# ── 编译正则（性能优化）────────────────────────────────────────
OPTION_RE = re.compile(
    r"^\s*(?:选项\s*)?([A-Ja-j])\s*[\.、\)）:：]?\s*(.+?)\s*$"
)
# 内联选项：识别行内的 A.xxx B.xxx C.xxx
INLINE_OPTION_RE = re.compile(r"(?<![A-Za-z])([A-Ja-j])\s*[\.、\)）:：]\s*")

QUESTION_START_RE = re.compile(
    r"^\s*(?:"
    r"\d+\s*[\.、\)）]\s+|"
    r"第\s*\d+\s*题\s*|"
    r"Q\s*\d+\s*[\.、\s]\s*|"
    r"[（(]\s*\d+\s*[）)]\s*|"
    r"[一二三四五六七八九十百]+[、.．]\s*|"
    r"【\s*(?:单选题|多选题|判断题|填空题|简答题|不定项|不定项选择)\s*】\s*"
    r")"
)
QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"\d+\s*[\.、\)）]\s*|"
    r"第\s*\d+\s*题\s*[:：、.．]?\s*|"
    r"Q\s*\d+\s*[\.、\s]\s*|"
    r"[（(]\s*\d+\s*[）)]\s*|"
    r"[一二三四五六七八九十百]+[、.．]\s*"
    r")"
)
TYPE_TAG_RE = re.compile(r"[\[【]\s*(单选题|多选题|判断题|填空题|简答题|不定项|不定项选择)\s*[\]】]")

NOISE_RE = re.compile(
    r"^(?:提交|下一题|上一题|开始答题|查看答案|复制|确定|取消|返回|完成|重新答题|"
    r"我的|首页|课程|学习|答题|展开|收起|加载中|暂无数据|请选择|请输入|"
    r"单选题|多选题|判断题|填空题|简答题|正确答案|答案解析|解析|得分|分数|已答|未答|"
    r"本题\d*分|满分\d*分|答题进度|剩余时间|倒计时)\s*$"
)


@dataclass
class ParsedQuestionData:
    """解析后的单题数据。"""
    id: str
    question: str
    question_type: QuestionType = QuestionType.UNKNOWN
    options: List[str] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


class QuestionParser:
    """从复制文本中识别题目、选项和题型。"""

    @classmethod
    def parse_text(cls, raw_text: str) -> List[ParsedQuestionData]:
        raw = cls._normalize(raw_text)
        if not raw:
            return []

        blocks = cls.split_text(raw)
        questions: List[ParsedQuestionData] = []
        for index, block in enumerate(blocks, start=1):
            parsed = cls._parse_block(index, block)
            if parsed:
                questions.append(parsed)
        return questions

    @classmethod
    def split_text(cls, raw_text: str) -> List[str]:
        """兼容旧 CLI 和前端的题块分割逻辑。"""
        raw = cls._normalize(raw_text)
        if not raw:
            return []

        by_number = cls._split_by_question_starts(raw)
        if len(by_number) > 1:
            return by_number

        by_blank = [s.strip() for s in re.split(r"\n\s*\n", raw) if cls._looks_useful(s)]
        if len(by_blank) > 1:
            return by_blank

        lines = [line.strip() for line in raw.split("\n") if cls._looks_useful(line)]
        if any(OPTION_RE.match(line) for line in lines):
            return [raw]
        return lines if lines else [raw]

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ 　]", " ", text)
        lines = []
        for line in text.split("\n"):
            clean = line.strip()
            if not clean or NOISE_RE.match(clean):
                continue
            lines.append(clean)
        return "\n".join(lines).strip()

    @staticmethod
    def _looks_useful(text: str) -> bool:
        text = text.strip()
        return bool(text) and not NOISE_RE.match(text)

    @staticmethod
    def _expand_inline_options(text: str) -> str:
        """把同一行里的 A/B/C/D 选项拆到独立行，避免题干吞掉选项。"""
        def repl(match: re.Match) -> str:
            prefix = "" if match.start() == 0 else "\n"
            return f"{prefix}{match.group(1).upper()}. "

        expanded = INLINE_OPTION_RE.sub(repl, text)
        return re.sub(r"\n{2,}", "\n", expanded).strip()

    @classmethod
    def _split_by_question_starts(cls, raw: str) -> List[str]:
        blocks: List[str] = []
        current: List[str] = []
        seen_question = False

        for line in raw.split("\n"):
            is_start = bool(QUESTION_START_RE.match(line)) and not OPTION_RE.match(line)
            if is_start:
                if current:
                    blocks.append("\n".join(current).strip())
                current = [line]
                seen_question = True
            elif current:
                current.append(line)
            elif seen_question:
                current = [line]
            else:
                current = [line]

        if current:
            blocks.append("\n".join(current).strip())

        useful = [b for b in blocks if cls._looks_useful(b)]
        starts = sum(1 for b in useful if QUESTION_START_RE.match(b.split("\n", 1)[0]))
        return useful if starts >= 2 else []

    @classmethod
    def _parse_block(cls, index: int, block: str) -> ParsedQuestionData | None:
        raw = block.strip()
        raw = cls._expand_inline_options(raw)
        lines = [line.strip() for line in raw.split("\n") if cls._looks_useful(line)]
        if not lines:
            return None

        question_lines: List[str] = []
        options: List[str] = []
        current_option = ""

        for line in lines:
            option_match = OPTION_RE.match(line)
            if option_match:
                if current_option:
                    options.append(current_option.strip())
                label = option_match.group(1).upper()
                content = option_match.group(2).strip()
                current_option = f"{label}. {content}"
                continue

            if current_option and not QUESTION_START_RE.match(line):
                # 选项跨行：续接到当前选项
                current_option = f"{current_option} {line.strip()}"
            else:
                question_lines.append(line)

        if current_option:
            options.append(current_option.strip())

        # ── 清理题干 ──
        question = "\n".join(question_lines).strip()
        # 移除题型标签
        question = TYPE_TAG_RE.sub("", question)
        # 移除题号前缀
        question = QUESTION_PREFIX_RE.sub("", question, count=1).strip()

        # ── 从原始文本提取题型标签（优先级最高）──
        tag_type = cls._extract_type_tag(raw)

        # ── 自动检测题型 ──
        detected_type = cls._detect_type(raw, question, options)
        question_type = tag_type if tag_type != QuestionType.UNKNOWN else detected_type

        # ── 收集警告 ──
        warnings: List[str] = []
        if not question:
            question = raw
            warnings.append("未能明确分离题干")
        if len(question) < 4:
            warnings.append("题干较短，请检查是否提取完整")
        if question_type in (QuestionType.SINGLE_CHOICE, QuestionType.MULTI_CHOICE) and len(options) < 2:
            warnings.append("选择题选项少于2个")
        if options and question_type == QuestionType.TRUE_FALSE:
            warnings.append("判断题含选项，请检查题型是否正确")
        if question_type == QuestionType.SINGLE_CHOICE:
            multi_signals = cls._has_multi_choice_signals(raw)
            if multi_signals:
                warnings.append(f"可能是多选题（检测到: {multi_signals}），请检查题型")
        if question_type == QuestionType.UNKNOWN:
            warnings.append("题型不确定")

        confidence = cls._confidence(question, options, question_type, warnings, tag_type)
        return ParsedQuestionData(
            id=f"q_{index}",
            question=question,
            question_type=question_type,
            options=options,
            raw_text=raw,
            confidence=confidence,
            warnings=warnings,
        )

    @staticmethod
    def _extract_type_tag(raw: str) -> QuestionType:
        """从【题型标签】中提取题型"""
        match = TYPE_TAG_RE.search(raw)
        if not match:
            return QuestionType.UNKNOWN
        tag = match.group(1)
        mapping = {
            "单选题": QuestionType.SINGLE_CHOICE,
            "多选题": QuestionType.MULTI_CHOICE,
            "判断题": QuestionType.TRUE_FALSE,
            "填空题": QuestionType.FILL_BLANK,
            "简答题": QuestionType.SHORT_ANSWER,
            "不定项": QuestionType.MULTI_CHOICE,
            "不定项选择": QuestionType.MULTI_CHOICE,
        }
        return mapping.get(tag, QuestionType.UNKNOWN)

    @staticmethod
    def _has_multi_choice_signals(text: str) -> str:
        """检测多选题信号词，返回匹配到的信号"""
        signals = [
            (r"多选", "多选"),
            (r"多项", "多项"),
            (r"不定项", "不定项"),
            (r"至少.*项", "至少X项"),
            (r"哪些", "哪些"),
            (r"以下.*正确", "以下正确"),
            (r"下列.*正确", "下列正确"),
            (r"属于.*的有", "属于...的有"),
            (r"包括.*的有", "包括...的有"),
            (r"正确.*有[哪几]?", "正确的有"),
            (r"错误.*有[哪几]?", "错误的有"),
            (r"属于.*是", "属于...是"),
            (r"不是.*的是", "不是...的是"),
        ]
        compact = text.replace(" ", "")
        for pattern, label in signals:
            if re.search(pattern, compact):
                return label
        return ""

    @staticmethod
    def _detect_type(raw: str, question: str, options: List[str]) -> QuestionType:
        """智能题型检测"""
        combined = f"{raw} {question}"
        compact = combined.replace(" ", "")

        # 1. 多选题检测（优先级最高 — 信号词最明确）
        if QuestionParser._has_multi_choice_signals(compact):
            return QuestionType.MULTI_CHOICE

        # 2. 判断题检测
        if len(options) <= 2 and re.search(r"判断|是否|对错|正误|正确.*错误|对.*错", compact):
            return QuestionType.TRUE_FALSE
        # 判断题：只有"正确""错误"两个选项
        if options and len(options) == 2:
            opt_texts = " ".join(options).replace(" ", "")
            if re.search(r"(正确|对|√|✅).*(错误|错|×|❌)", opt_texts) or \
               re.search(r"(错误|错|×|❌).*(正确|对|√|✅)", opt_texts):
                return QuestionType.TRUE_FALSE

        # 3. 填空题检测
        if re.search(r"填空|_{2,}|（\s*）|\(\s*\)|【\s*】|\[空格\]|填入", raw):
            return QuestionType.FILL_BLANK

        # 4. 简答题检测
        if re.search(r"简答|论述|说明理由|阐述|分析原因|试述|概述|简述|请说明|请分析", compact):
            return QuestionType.SHORT_ANSWER

        # 5. 有选项默认为单选
        if len(options) >= 2:
            return QuestionType.SINGLE_CHOICE

        return QuestionType.UNKNOWN

    @staticmethod
    def _confidence(
        question: str,
        options: List[str],
        question_type: QuestionType,
        warnings: List[str],
        tag_type: QuestionType = None,
    ) -> float:
        score = 0.55
        if len(question) >= 8:
            score += 0.15
        if options:
            score += 0.2 if len(options) >= 2 else 0.05
        if question_type != QuestionType.UNKNOWN:
            score += 0.1
        # 有题型标签时大幅提升置信度
        if tag_type and tag_type != QuestionType.UNKNOWN:
            score += 0.15
        score -= min(len(warnings) * 0.12, 0.35)
        return max(0.1, min(0.98, round(score, 2)))
