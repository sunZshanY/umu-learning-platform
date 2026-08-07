"""
UMU全自动刷题脚本 — CLI命令行版
================================
直接读取题目文件 → 并发AI答题 → 3分钟限时 → 纯文本答案输出

用法:
    python auto_quiz.py questions.txt                    # 从文件读取题目
    python auto_quiz.py questions.txt --concurrency 15   # 15并发
    python auto_quiz.py questions.txt --timeout 180      # 3分钟限时
    python auto_quiz.py --help                           # 查看帮助

题目文件格式（自动识别）:
    - 空行分隔多题（推荐）
    - 数字编号: 1. 题目 / 2、题目 / 第1题
    - 每题可包含选项行

输出:
    - 控制台实时进度
    - answers.txt 纯文本答案（可直接复制到UMU）
    - quiz_result.json 完整结果记录

作者: UMU Quiz Helper
版本: 2.0.0
"""
import asyncio
import sys
import os
import re
import json
import time
import signal
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# ── 加载 .env 配置 ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── 依赖检查 ─────────────────────────────────────────────────
try:
    from openai import AsyncOpenAI, OpenAIError
except ImportError:
    print("错误: 缺少 openai 库，请运行: pip install openai")
    sys.exit(1)

try:
    from question_parser import QuestionParser as SharedQuestionParser
except ImportError:
    SharedQuestionParser = None

# ── 日志 ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("auto_quiz")

# ── 配置 ─────────────────────────────────────────────────────
DEFAULT_API_KEY = os.environ.get("UMU_AI_API_KEY", "")
DEFAULT_API_BASE = os.environ.get("UMU_AI_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.environ.get("UMU_AI_MODEL", "deepseek-chat")
DEFAULT_TIMEOUT = 180          # 3分钟
DEFAULT_CONCURRENCY = 20       # 并发数（300题约90秒完成）
DEFAULT_MAX_WRONG = 3          # 最大错误数
AI_TIMEOUT = 30                # 单次AI调用超时(秒) — 提升以应对复杂推理
MAX_RETRIES = 2                # AI调用重试次数
MAX_TOKENS = 2048              # 最大输出token


# ═══════════════════════════════════════════════════════════════
# 题目解析器
# ═══════════════════════════════════════════════════════════════
@dataclass
class Question:
    """单道题目"""
    index: int
    text: str
    raw: str = ""


class QuestionParser:
    """解析题目文件，自动识别分隔格式"""

    @staticmethod
    def parse(filepath: str) -> List[Question]:
        """从文件读取并解析题目列表"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError("文件内容为空")

        blocks = QuestionParser._split(raw)
        questions = []
        for i, block in enumerate(blocks):
            text = block.strip()
            if text:
                questions.append(Question(index=i + 1, text=text, raw=block))

        log.info(f"从 {filepath} 解析出 {len(questions)} 道题目")
        return questions

    @staticmethod
    def _split(raw: str) -> List[str]:
        """智能分割题目"""
        if SharedQuestionParser:
            return [q.raw_text or q.question for q in SharedQuestionParser.parse_text(raw)]

        by_blank = [s.strip() for s in re.split(r'\n\s*\n', raw) if s.strip()]
        if len(by_blank) > 1:
            return by_blank

        by_num = [s.strip() for s in re.split(
            r'\n(?=\s*(?:\d+[\.\、\)]\s*|第\d+题|Q\d+[\.\s]|[（(]\d+[）)]))',
            raw
        ) if s.strip()]
        if len(by_num) > 1:
            return by_num

        return [s.strip() for s in raw.split('\n') if s.strip()]


# ═══════════════════════════════════════════════════════════════
# AI答题客户端
# ═══════════════════════════════════════════════════════════════
class AIClient:
    """异步AI答题客户端"""

    SYSTEM_PROMPT = """你是国家级标准化考试命题专家，拥有各学科博士学位。你的目标是100%准确率。

【核心答题协议 — 必须逐条执行】
第1步·破题：提取题干的1-2个核心关键词，判定本题考察的知识点或逻辑关系。
第2步·审问：明确题干问的是什么——「正确的是/错误的是」「属于/不属于」「包括/不包括」。用笔圈出否定词。
第3步·析选项：
  - 单选题：逐项与题干核心关键词比对，先排除明显错误项，再从剩余中选最优。遇到两个相似选项时，仔细找区分点。
  - 多选题：独立评判每个选项（假设其他选项不存在），对则留、错则弃、疑则弃。
  - 判断题：先忽略题干的否定词判断陈述本身的真假，再根据题干问法决定输出「正确」还是「错误」。
第4步·验答：答案与题干再次比对——我选的是题干问的那个方向吗？
第5步·输出：严格按格式输出，一字不多。

【输出格式 — 违反格式=答错】
单选题 → "<字母>. <选项内容>"              示例: C. 线粒体
多选题 → "<字母序列>. <选项内容>"          示例: ABD. 光合作用、呼吸作用、蒸腾作用
                                           字母按A→Z顺序排列，不可打乱
判断题 → "正确" 或 "错误"                 仅此两字，不加任何标点
填空题 → 填空内容                         不超过30字
简答题 → 核心答案                         不超过80字

【高频错误防御】
错误1「审反」：题干问「不属于/错误的是」→ 输出前强制检查，反向确认。
错误2「漏选」：多选题少选了正确选项 → 每个选项独立评估，不要横向比较。
错误3「多选」：多选题选了错误选项 → 有疑虑就坚决不选。
错误4「看串行」：选项字母与内容对应错 → 每个选项连字母带内容一起读。
错误5「想当然」：凭常识选而不看题目表述 → 答案严格基于题目给出的信息。

【判断题决策树】
1. 陈述本身是真还是假？
2. 题干最终问的是「正确」还是「错误」？
3. 问「正确」+陈述真 → 「正确」；问「正确」+陈述假 → 「错误」
4. 问「错误」+陈述真 → 「错误」；问「错误」+陈述假 → 「正确」

【多选题决策树】
1. 逐一检查每个选项，独立判断其正确性
2. 只选100%确定正确的，有疑虑的不选
3. 按字母顺序排列输出"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=AI_TIMEOUT,
            max_retries=0,
        )
        self.model = model
        self.stats = {"total": 0, "success": 0, "fail": 0, "retries": 0}

    async def answer(self, question: str, qid: int = 0) -> Tuple[str, bool, float]:
        """
        对单题作答
        Returns: (answer_text, success, elapsed_seconds)
        """
        self.stats["total"] += 1
        t0 = time.time()

        # 构建结构化输入
        user_prompt = self._build_user_prompt(question)

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                    top_p=0.95,
                )
                raw = resp.choices[0].message.content or ""
                answer = self._clean(raw)
                elapsed = time.time() - t0
                self.stats["success"] += 1
                self.stats["retries"] += attempt
                return (answer, True, elapsed)

            except Exception as e:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    elapsed = time.time() - t0
                    self.stats["fail"] += 1
                    self.stats["retries"] += attempt
                    return (f"[错误: {str(e)[:80]}]", False, elapsed)

    @staticmethod
    def _build_user_prompt(question: str) -> str:
        """构建结构化用户提示词"""
        # 高亮否定词和限定词
        highlighted = question
        negations = [
            "不是", "不正确", "不属于", "错误的是", "不正确的是", "错误的说法",
            "无关", "不能", "不包括", "不包含", "没有", "无", "非", "除外",
            "不符合", "不涉及", "不能解释",
        ]
        qualifiers = [
            "至少", "全部", "均", "一定", "最", "首要", "主要", "根本",
            "核心", "基本", "唯一", "必须", "绝对", "完全",
        ]
        for word in negations + qualifiers:
            if word in highlighted:
                highlighted = highlighted.replace(word, f"【{word}】")

        # 检测题干方向
        direction = ""
        if any(w in question for w in ["不属于", "错误的是", "不正确", "错误的", "不包括", "无关"]):
            direction = "\n\n【警告】本题问的是否定方向！要选不正确/不属于的选项！"
        elif any(w in question for w in ["属于", "正确的是", "正确", "包括"]):
            direction = "\n\n【确认】本题问的是肯定方向，选出正确/属于的选项。"

        return f"""【题目】
{highlighted}{direction}

【题型】请根据题干和选项自动判断题型（单选/多选/判断/填空/简答）

【指令】只输出最终答案。单选→字母+内容；多选→字母序列+内容(按字母序)；判断→正确/错误；填空→填空内容；简答→核心答案。"""

    @staticmethod
    def _clean(text: str) -> str:
        """清洗AI输出中的格式残留"""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(答案|解析|置信度|正确选项|正确答案|我的答案是?|最终答案)[：:]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\[【].*?[\]】]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


# ═══════════════════════════════════════════════════════════════
# 答题运行器
# ═══════════════════════════════════════════════════════════════
@dataclass
class QuizResult:
    """单题结果"""
    index: int
    question_preview: str
    answer: str
    success: bool
    elapsed: float
    is_wrong: bool = False


@dataclass
class QuizSession:
    """答题会话"""
    questions: List[Question]
    results: List[QuizResult] = field(default_factory=list)
    wrong_count: int = 0
    max_wrong: int = DEFAULT_MAX_WRONG
    started_at: float = 0.0
    finished_at: float = 0.0
    timeout: int = DEFAULT_TIMEOUT
    stopped_early: bool = False
    stop_reason: str = ""

    @property
    def answered(self) -> int:
        return len(self.results)

    @property
    def remaining(self) -> int:
        return max(0, self.max_wrong - self.wrong_count)

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def time_remaining(self) -> float:
        return max(0, self.timeout - self.elapsed)

    @property
    def is_over(self) -> bool:
        return self.stopped_early or self.wrong_count >= self.max_wrong or self.time_remaining <= 0


class QuizRunner:
    """答题运行器 — 核心调度引擎"""

    def __init__(
        self,
        ai_client: AIClient,
        concurrency: int = DEFAULT_CONCURRENCY,
        timeout: int = DEFAULT_TIMEOUT,
        max_wrong: int = DEFAULT_MAX_WRONG,
        output_dir: str = ".",
    ):
        self.ai = ai_client
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_wrong = max_wrong
        self.output_dir = Path(output_dir)
        self.semaphore = asyncio.Semaphore(concurrency)
        self._stop_flag = False
        self._timer_task = None

    async def run(self, questions: List[Question]) -> QuizSession:
        """执行完整答题流程"""
        session = QuizSession(
            questions=questions,
            max_wrong=self.max_wrong,
            timeout=self.timeout,
            started_at=time.time(),
        )

        # 3分钟倒计时显示
        self._timer_task = asyncio.create_task(self._countdown(session))
        # 并发答题
        await self._process_all(session)
        # 停止计时器
        if self._timer_task:
            self._timer_task.cancel()

        session.finished_at = time.time()
        return session

    async def _countdown(self, session: QuizSession):
        """倒计时显示"""
        try:
            while not session.is_over and session.answered < len(session.questions):
                remaining = session.time_remaining
                mins, secs = int(remaining // 60), int(remaining % 60)
                pct = (session.answered / max(len(session.questions), 1)) * 100

                # 动态颜色
                color = "\033[92m" if remaining > 120 else "\033[93m" if remaining > 60 else "\033[91m"
                reset = "\033[0m"

                sys.stderr.write(
                    f"\r{color}⏱ {mins:02d}:{secs:02d}{reset} | "
                    f"📝 {session.answered}/{len(session.questions)} ({pct:.0f}%) | "
                    f"❌ 错误: {session.wrong_count}/{session.max_wrong} | "
                    f"🔵 并发: {self.concurrency}  "
                )
                sys.stderr.flush()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def _process_all(self, session: QuizSession):
        """并发处理所有题目"""
        tasks = []
        for q in session.questions:
            if session.is_over:
                # 超时或错误满 → 停止创建新任务
                unprocessed = len(session.questions) - session.answered
                if unprocessed > 0:
                    session.stopped_early = True
                    session.stop_reason = (
                        "时间耗尽" if session.time_remaining <= 0
                        else f"错误次数达上限({session.wrong_count}/{session.max_wrong})"
                    )
                    log.warning(f"⛔ 答题终止: {session.stop_reason}，剩余{unprocessed}题未处理")
                break

            task = asyncio.create_task(self._answer_one(q, session))
            tasks.append(task)

        # 等待所有已创建的任务完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _answer_one(self, question: Question, session: QuizSession) -> None:
        """处理单道题目（带信号量限流）"""
        async with self.semaphore:
            if session.is_over:
                return

            answer, success, elapsed = await self.ai.answer(
                question.text, question.index
            )

            result = QuizResult(
                index=question.index,
                question_preview=question.text[:80].replace('\n', ' | '),
                answer=answer,
                success=success,
                elapsed=elapsed,
            )
            session.results.append(result)

            # 实时输出单题结果
            status = "✅" if success else "❌"
            sys.stderr.write(
                f"\n  {status} #{question.index:03d} [{elapsed:.1f}s] "
                f"{answer[:60]}\n"
            )
            sys.stderr.flush()


# ═══════════════════════════════════════════════════════════════
# 输出生成器
# ═══════════════════════════════════════════════════════════════
class OutputWriter:
    """生成答案输出文件"""

    @staticmethod
    def write_answers(session: QuizSession, output_dir: Path) -> Path:
        """写入纯文本答案文件"""
        path = output_dir / "answers.txt"
        lines = []
        lines.append("=" * 50)
        lines.append(f"UMU刷题答案 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"总题数: {len(session.questions)} | 已答: {session.answered}")
        lines.append(f"成功: {sum(1 for r in session.results if r.success)} | "
                     f"失败: {sum(1 for r in session.results if not r.success)}")
        lines.append(f"耗时: {session.elapsed:.1f}秒 | 错误: {session.wrong_count}/{session.max_wrong}")
        lines.append("=" * 50)
        lines.append("")

        for r in sorted(session.results, key=lambda x: x.index):
            marker = "❌" if r.is_wrong else "✅"
            lines.append(f"{marker} #{r.index:03d}  {r.answer}")
            lines.append(f"    📄 {r.question_preview}")
            lines.append("")

        # 未答题目
        answered_ids = {r.index for r in session.results}
        for q in session.questions:
            if q.index not in answered_ids:
                lines.append(f"⏭  #{q.index:03d}  [未作答]")
                lines.append(f"    📄 {q.text[:80].replace(chr(10), ' | ')}")
                lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def write_answers_compact(session: QuizSession, output_dir: Path) -> Path:
        """写入紧凑版答案（仅答案，便于复制到UMU）"""
        path = output_dir / "answers_compact.txt"
        lines = []
        for r in sorted(session.results, key=lambda x: x.index):
            if r.success:
                lines.append(f"#{r.index:03d}  {r.answer}")
            else:
                lines.append(f"#{r.index:03d}  [失败] {r.answer}")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def write_json(session: QuizSession, output_dir: Path) -> Path:
        """写入JSON完整记录"""
        path = output_dir / "quiz_result.json"
        data = {
            "timestamp": datetime.now().isoformat(),
            "total": len(session.questions),
            "answered": session.answered,
            "elapsed": round(session.elapsed, 1),
            "wrong_count": session.wrong_count,
            "stopped_early": session.stopped_early,
            "stop_reason": session.stop_reason,
            "results": [
                {
                    "index": r.index,
                    "answer": r.answer,
                    "success": r.success,
                    "elapsed": round(r.elapsed, 2),
                    "preview": r.question_preview,
                }
                for r in sorted(session.results, key=lambda x: x.index)
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


# ═══════════════════════════════════════════════════════════════
# 终端UI
# ═══════════════════════════════════════════════════════════════
class TerminalUI:
    """终端输出美化"""

    @staticmethod
    def print_header():
        print("""
╔══════════════════════════════════════════════════════════╗
║       🎯 UMU 全自动刷题脚本 v2.0.0                       ║
║       3分钟限时 | 并发AI答题 | 自动防错                    ║
╚══════════════════════════════════════════════════════════╝
""")

    @staticmethod
    def print_session_summary(session: QuizSession):
        """打印最终汇总"""
        print("\n" + "=" * 55)
        print("                    📊 答题报告")
        print("=" * 55)
        print(f"  总题数    : {len(session.questions)}")
        print(f"  已作答    : {session.answered}")
        print(f"  AI成功    : {sum(1 for r in session.results if r.success)}")
        print(f"  AI失败    : {sum(1 for r in session.results if not r.success)}")
        print(f"  标记错误  : {session.wrong_count}/{session.max_wrong}")
        print(f"  总耗时    : {session.elapsed:.1f} 秒")
        print(f"  平均每题  : {(session.elapsed / max(session.answered, 1)):.1f} 秒")
        if session.stopped_early:
            print(f"  ⛔ 终止原因: {session.stop_reason}")
        print("=" * 55)

        # 打印所有答案
        print("\n📋 全部答案:")
        print("-" * 55)
        for r in sorted(session.results, key=lambda x: x.index):
            status = "❌" if r.is_wrong else "✅"
            print(f"  {status} #{r.index:03d}  {r.answer}")

        if session.answered < len(session.questions):
            print(f"\n  ⚠️  剩余 {len(session.questions) - session.answered} 题未作答")

    @staticmethod
    def print_output_files(answers_path, compact_path, json_path):
        print(f"\n📁 输出文件:")
        print(f"  详细答案: {answers_path}")
        print(f"  紧凑答案: {compact_path}")
        print(f"  完整记录: {json_path}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════
async def main():
    parser = argparse.ArgumentParser(
        description="UMU全自动刷题脚本 — AI并发答题，3分钟限时",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_quiz.py questions.txt                     # 基本用法
  python auto_quiz.py questions.txt -c 15 -t 180        # 15并发,3分钟
  python auto_quiz.py questions.txt --max-wrong 3       # 最多3次错误
  python auto_quiz.py questions.txt -o ./output         # 指定输出目录
        """,
    )
    parser.add_argument("file", help="题目文件路径（txt格式）")
    parser.add_argument("-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"并发数（默认: {DEFAULT_CONCURRENCY}）")
    parser.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"限时秒数（默认: {DEFAULT_TIMEOUT}s = 3分钟）")
    parser.add_argument("--max-wrong", type=int, default=DEFAULT_MAX_WRONG,
                        help=f"最大错误次数（默认: {DEFAULT_MAX_WRONG}）")
    parser.add_argument("-o", "--output", type=str, default=".",
                        help="输出目录（默认: 当前目录）")
    parser.add_argument("-k", "--api-key", type=str, default=DEFAULT_API_KEY,
                        help="AI API Key")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE,
                        help="AI API Base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="AI模型名称")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅解析题目不调用AI（测试用）")

    args = parser.parse_args()

    TerminalUI.print_header()

    # 1. 解析题目
    print(f"📖 读取题目文件: {args.file}")
    questions = QuestionParser.parse(args.file)
    print(f"   解析出 {len(questions)} 道题目")

    if args.dry_run:
        print("\n📋 题目预览（前5题）:")
        for q in questions[:5]:
            print(f"   #{q.index:03d}: {q.text[:100]}...")
        print(f"\n   共 {len(questions)} 题，dry-run 模式结束")
        return

    # 2. 检查API Key
    api_key = args.api_key
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("\n❌ 错误: 请设置 API Key")
        print("   方式1: 设置环境变量 UMU_AI_API_KEY")
        print("   方式2: 使用 --api-key 参数")
        sys.exit(1)

    # 3. 初始化
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ai_client = AIClient(
        api_key=api_key,
        base_url=args.api_base,
        model=args.model,
    )

    runner = QuizRunner(
        ai_client=ai_client,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_wrong=args.max_wrong,
        output_dir=str(output_dir),
    )

    # 4. 运行
    print(f"\n🚀 开始答题 | {len(questions)}题 | {args.concurrency}并发 | {args.timeout}秒限时")
    print(f"   模型: {args.model} | 最大错误: {args.max_wrong}")
    print("-" * 55)

    t0 = time.time()
    session = await runner.run(questions)
    total_elapsed = time.time() - t0

    # 5. 输出结果
    TerminalUI.print_session_summary(session)

    answers_path = OutputWriter.write_answers(session, output_dir)
    compact_path = OutputWriter.write_answers_compact(session, output_dir)
    json_path = OutputWriter.write_json(session, output_dir)

    TerminalUI.print_output_files(answers_path, compact_path, json_path)

    # 6. AI统计
    print(f"\n📊 AI服务统计:")
    print(f"   总请求: {ai_client.stats['total']} | "
          f"成功: {ai_client.stats['success']} | "
          f"失败: {ai_client.stats['fail']} | "
          f"重试: {ai_client.stats['retries']}")

    print(f"\n✅ 完成! 总耗时: {total_elapsed:.1f}秒")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⛔ 用户中断")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
