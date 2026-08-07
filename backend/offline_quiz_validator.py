"""Offline question and answer validation; never accesses network or submits answers."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from question_parser import QuestionParser

QUESTION_NUMBER_RE = re.compile(
    r"^\s*(?:Q\s*)?(\d+)\s*[.．、:：)]?\s*", re.IGNORECASE
)
ANSWER_RE = re.compile(
    r"^\s*(?:Q\s*)?(\d+)\s*(?:[=：:\t]|[.．、)])\s*(.*?)\s*$",
    re.IGNORECASE,
)
OPTION_RE = re.compile(r"^\s*([A-J])\s*[.．、:：)）]\s*(.*?)\s*$", re.IGNORECASE)
LABELS_RE = re.compile(r"^[A-J]+$", re.IGNORECASE)


@dataclass
class QuestionRecord:
    number: int
    parser_id: str
    question: str
    question_type: str
    options: dict[str, str]
    confidence: float
    warnings: list[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class AnswerRecord:
    number: int
    raw_answer: str
    normalized_answer: Optional[str]
    source_line: int


def remove_export_line_number(line: str) -> str:
    """Remove a copied table row number only when followed by whitespace/tab."""
    return re.sub(r"^\s*\d+\t+", "", line).strip()


def extract_question_number(text: str) -> Optional[int]:
    match = QUESTION_NUMBER_RE.match(text)
    return int(match.group(1)) if match else None


def parse_questions(path: Path) -> tuple[list[QuestionRecord], list[str]]:
    raw = path.read_text(encoding="utf-8-sig")
    cleaned = "\n".join(remove_export_line_number(line) for line in raw.splitlines())
    parsed = QuestionParser.parse_text(cleaned)
    records: list[QuestionRecord] = []
    warnings: list[str] = []

    for item in parsed:
        number = extract_question_number(item.raw_text)
        if number is None:
            warnings.append(f"无法提取题号: {item.id}")
            continue
        options: dict[str, str] = {}
        for option in item.options:
            match = OPTION_RE.match(option)
            if not match:
                warnings.append(f"题目 {number}: 无法解析选项 {option}")
                continue
            label = match.group(1).upper()
            if label in options:
                warnings.append(f"题目 {number}: 选项 {label} 重复")
            options[label] = match.group(2).strip()
        records.append(
            QuestionRecord(
                number=number,
                parser_id=item.id,
                question=item.question,
                question_type=item.question_type.value,
                options=options,
                confidence=item.confidence,
                warnings=list(item.warnings),
                raw_text=item.raw_text,
            )
        )

    seen: set[int] = set()
    for record in records:
        if record.number in seen:
            warnings.append(f"题号重复: {record.number}")
        seen.add(record.number)
        if len(record.options) < 2:
            warnings.append(f"题目 {record.number}: 选项少于 2 个")
    for previous, current in zip(records, records[1:]):
        if current.number != previous.number + 1:
            warnings.append(f"题号不连续: {previous.number} 后为 {current.number}")
    return records, warnings


def normalize_answer(raw: str, options: dict[str, str]) -> Optional[str]:
    value = raw.strip()
    option_match = re.match(r"^([A-J](?:\s*[,，、/]\s*[A-J])*)\b", value, re.IGNORECASE)
    if option_match:
        value = option_match.group(1)
    compact = re.sub(r"[\s,，、/]+", "", value).upper()
    if LABELS_RE.fullmatch(compact):
        return compact

    lowered = value.casefold()
    true_words = {"对", "正确", "是", "true", "yes"}
    false_words = {"错", "错误", "否", "false", "no"}
    target = "true" if lowered in true_words else "false" if lowered in false_words else None
    if target:
        matches = []
        for label, text in options.items():
            option_lower = text.casefold()
            if target == "true" and option_lower in true_words:
                matches.append(label)
            if target == "false" and option_lower in false_words:
                matches.append(label)
        if len(matches) == 1:
            return matches[0]
    return None


def parse_answers(path: Path) -> tuple[list[AnswerRecord], list[str]]:
    answers: list[AnswerRecord] = []
    warnings: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = ANSWER_RE.match(line)
        if not match:
            warnings.append(f"答案第 {line_number} 行格式错误: {line[:80]}")
            continue
        answers.append(
            AnswerRecord(
                number=int(match.group(1)),
                raw_answer=match.group(2).strip(),
                normalized_answer=None,
                source_line=line_number,
            )
        )
    return answers, warnings


def validate(questions: list[QuestionRecord], answers: list[AnswerRecord]) -> dict:
    question_map: dict[int, QuestionRecord] = {}
    issues: list[dict] = []
    for question in questions:
        if question.number in question_map:
            issues.append({"code": "duplicate_question", "number": question.number})
        question_map[question.number] = question

    answer_map: dict[int, AnswerRecord] = {}
    for answer in answers:
        if answer.number in answer_map:
            issues.append({"code": "duplicate_answer", "number": answer.number, "line": answer.source_line})
        else:
            answer_map[answer.number] = answer

    question_results = []
    for question in questions:
        answer = answer_map.get(question.number)
        if answer is None:
            question_results.append({"number": question.number, "status": "missing", "answer": None})
            continue
        normalized = normalize_answer(answer.raw_answer, question.options)
        answer.normalized_answer = normalized
        if normalized is None:
            status = "invalid"
            issues.append({"code": "invalid_answer", "number": question.number, "line": answer.source_line})
        elif len(normalized) != 1:
            status = "invalid"
            issues.append({"code": "multiple_answers_for_single_choice", "number": question.number})
        elif any(label not in question.options for label in normalized):
            status = "invalid"
            issues.append({"code": "unknown_option", "number": question.number, "answer": normalized})
        else:
            status = "ok"
        question_results.append({"number": question.number, "status": status, "answer": normalized, "raw_answer": answer.raw_answer})

    for number in sorted(set(answer_map) - set(question_map)):
        issues.append({"code": "extra_answer", "number": number})

    counts = {status: sum(item["status"] == status for item in question_results) for status in ("ok", "missing", "invalid")}
    return {
        "schema_version": 1,
        "valid": not issues and counts["missing"] == 0,
        "summary": {"questions": len(questions), "answers": len(answers), **counts, "issues": len(issues)},
        "issues": issues,
        "questions": [asdict(question) for question in questions],
        "results": question_results,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="离线解析题目并校验答案 TXT")
    parser.add_argument("questions", type=Path)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--output", type=Path, default=Path("review_report.json"))
    args = parser.parse_args(argv)
    try:
        questions, question_warnings = parse_questions(args.questions)
        answers: list[AnswerRecord] = []
        answer_warnings: list[str] = []
        if args.answers:
            answers, answer_warnings = parse_answers(args.answers)
        report = validate(questions, answers) if args.answers else {
            "schema_version": 1,
            "valid": bool(questions) and not question_warnings,
            "summary": {"questions": len(questions), "answers": 0, "issues": len(question_warnings)},
            "issues": [{"code": "question_warning", "message": warning} for warning in question_warnings],
            "questions": [asdict(question) for question in questions],
            "results": [],
        }
        report["warnings"] = question_warnings + answer_warnings
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(f"解析题目: {summary['questions']}")
    if args.answers:
        print(f"答案有效: {summary['ok']}，缺失: {summary['missing']}，无效: {summary['invalid']}")
    print(f"报告已写入: {args.output}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
