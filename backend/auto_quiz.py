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
AI_TIMEOUT = 20                # 单次AI调用超时(秒)
MAX_RETRIES = 2                # AI调用重试次数


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

    SYSTEM_PROMPT = """你是一个追求100%正确率的专业答题专家。先在脑中严谨推理，确认无误后只输出最终答案。

核心原则：
1. 绝对准确第一，速度第二
2. 逐字审题，警惕陷阱（绝对化表述、双重否定、偷换概念）
3. 理工科题目用公式/定理验证，文科题目以权威教材为准
4. 不确定时优先选最常见的标准答案，而非偏门选项

输出规范（严格遵守）：
- 单选题：选项字母+内容，如"B. 北京"
- 多选题：选项字母+内容，如"ABD. 光合作用、呼吸作用、蒸腾作用"
- 判断题：仅"正确"或"错误"
- 填空题：仅答案内容，不超过30字
- 简答题：仅核心答案，不超过50字

严禁事项：
- 禁止任何Markdown标记（** __ ## ` 等）
- 禁止输出推理过程、解析步骤、解释说明
- 禁止使用下划线、星号、井号、反引号
- 禁止输出"答案:"、"解析:"、"正确选项是"等前缀
- 禁止输出置信度数值
- 禁止多行输出（多选除外）
- 禁止任何括号注释"""

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

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.1,
                    max_tokens=512,
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
    def _clean(text: str) -> str:
        """清洗AI输出中的格式残留"""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(答案|解析|置信度)[：:]\s*', '', text, flags=re.MULTILINE)
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
