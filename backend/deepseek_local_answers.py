"""Generate local answer suggestions with DeepSeek; never accesses UMU."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ai_service import AIService
from models import QuestionType
from offline_quiz_validator import QuestionRecord, parse_questions


@dataclass
class Suggestion:
    number: int
    success: bool
    answer: str
    raw_answer: str
    error: str
    elapsed_ms: int


def extract_choice(answer: str, question: QuestionRecord) -> Optional[str]:
    labels = "".join(question.options)
    match = re.match(r"^\s*([A-J]+)(?:\b|[.．、:：)）])", answer, re.IGNORECASE)
    if match:
        choice = "".join(dict.fromkeys(match.group(1).upper()))
        if choice and all(label in labels for label in choice):
            return choice

    compact = answer.strip().casefold()
    truthy = {"对", "正确", "是", "true", "yes"}
    falsy = {"错", "错误", "否", "false", "no"}
    target = truthy if compact in truthy else falsy if compact in falsy else None
    if target:
        matches = [label for label, text in question.options.items() if text.strip().casefold() in target]
        if len(matches) == 1:
            return matches[0]
    return None


def build_question_type(question: QuestionRecord) -> QuestionType:
    try:
        parsed = QuestionType(question.question_type)
    except ValueError:
        parsed = QuestionType.UNKNOWN
    if parsed == QuestionType.FILL_BLANK and question.options:
        return QuestionType.SINGLE_CHOICE
    return parsed


async def generate_suggestions(
    questions: list[QuestionRecord],
    service: AIService,
    concurrency: int,
) -> list[Suggestion]:
    semaphore = asyncio.Semaphore(concurrency)

    async def answer_one(question: QuestionRecord) -> Suggestion:
        async with semaphore:
            started = time.monotonic()
            try:
                response = await service.answer_question(
                    question=question.question,
                    question_type=build_question_type(question),
                    options=[f"{label}. {text}" for label, text in question.options.items()],
                )
                raw_answer = response.answer.strip()
                normalized = extract_choice(raw_answer, question)
                return Suggestion(
                    number=question.number,
                    success=bool(response.success and normalized),
                    answer=normalized or "",
                    raw_answer=raw_answer,
                    error="" if normalized else "无法规范化为有效选项",
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                )
            except Exception as exc:
                return Suggestion(
                    number=question.number,
                    success=False,
                    answer="",
                    raw_answer="",
                    error=str(exc)[:200],
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                )

    results = await asyncio.gather(*(answer_one(question) for question in questions))
    return sorted(results, key=lambda item: item.number)


def write_outputs(suggestions: list[Suggestion], text_path: Path, json_path: Path) -> None:
    text_lines = [
        f"{item.number}={item.answer}" if item.success else f"{item.number}=[ERROR] {item.error}"
        for item in suggestions
    ]
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "summary": {
            "total": len(suggestions),
            "success": sum(item.success for item in suggestions),
            "failed": sum(not item.success for item in suggestions),
        },
        "suggestions": [asdict(item) for item in suggestions],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 DeepSeek 为本地题目 TXT 生成答案建议（不连接或提交到 UMU）"
    )
    parser.add_argument("questions", type=Path, help="本地题目 TXT")
    parser.add_argument("--concurrency", type=int, default=8, choices=range(1, 21), metavar="1-20")
    parser.add_argument("--answers-output", type=Path, default=Path("ai_answers.txt"))
    parser.add_argument("--json-output", type=Path, default=Path("ai_answers.json"))
    parser.add_argument("--dry-run", action="store_true", help="只解析题目，不调用 DeepSeek")
    return parser


async def async_main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        questions, warnings = parse_questions(args.questions)
    except (OSError, UnicodeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if not questions:
        print("错误: 没有解析出题目", file=sys.stderr)
        return 2
    print(f"解析题目: {len(questions)}，解析警告: {len(warnings)}")
    if args.dry_run:
        return 0

    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.environ.get("UMU_AI_API_KEY", "").strip()
    if not api_key:
        print("错误: 请通过 UMU_AI_API_KEY 环境变量配置已轮换的 DeepSeek API key", file=sys.stderr)
        return 2

    service = AIService(
        api_key=api_key,
        base_url=os.environ.get("UMU_AI_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.environ.get("UMU_AI_MODEL", "deepseek-chat"),
    )
    suggestions = await generate_suggestions(questions, service, args.concurrency)
    write_outputs(suggestions, args.answers_output, args.json_output)
    success = sum(item.success for item in suggestions)
    print(f"生成完成: 成功 {success}，失败 {len(suggestions) - success}")
    print(f"答案文件: {args.answers_output}")
    print(f"报告文件: {args.json_output}")
    return 0 if success == len(suggestions) else 1


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
