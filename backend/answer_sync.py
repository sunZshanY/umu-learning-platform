"""离线答案同步纠错 — 将 AI 答案与权威答案校对并生成可提交的最终答案.

工作流（纯本地，不连接或提交到 UMU）:
  1. 解析本地题目 TXT（复用 offline_quiz_validator.parse_questions）。
  2. 解析候选答案（AI 生成）与可选权威答案（人工复核的正确答案）。
  3. 规范化每个答案到题干实际存在的选项字母（修复对/错类题目 C. 对 → 正确的 A/B 字母）。
  4. 若给定权威答案：
        依权威答案为准生成"最终答案"，并统计 AI 答案与权威答案的一致率（批改）。
        — correct      : AI 与权威一致
        — wrong        : AI 有值但与权威不同（已按权威纠错）
        — corrected     : AI 无值/无效，已按权威补齐
        — unsupported   : AI 无效且无权威可依，保持原样并标注
  5. 若无权威答案：
        仅做规范化纠错（对/错 → 有效选项字母、清洗格式），逐条输出纠正记录。
  6. 输出: corrected_answers.txt（可直接提交）+ answer_sync_report.json（明细报告）。

用法:
    python answer_sync.py questions.txt --answers ai_answers.txt
    python answer_sync.py questions.txt --answers ai_answers.txt --gold gold_answers.txt
    python answer_sync.py questions.txt --answers ai_answers.txt --output fixed.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from offline_quiz_validator import AnswerRecord, QuestionRecord, normalize_answer, parse_answers, parse_questions


class AnswerSyncError(Exception):
    """answer_sync 的领域异常。"""


@dataclass
class SyncDecision:
    number: int
    question_type: str
    ai_answer: str
    ai_normalized: Optional[str]
    gold_answer: str
    gold_normalized: Optional[str]
    final_answer: str
    status: str            # correct / wrong / corrected / unsupported / untouched
    note: str = ""


def normalize(text: str, question: QuestionRecord) -> Optional[str]:
    """规范化答案到真实存在的选项字母，并修复对/错类误判。"""
    raw = text.strip()
    if not raw:
        return None

    labels = question.options
    if not labels:
        return None

    # 去除 "答案：" / "正确答案" / "选择" 等常见前缀。
    raw = re.sub(r"^(?:正确答案|答案|对/错|选择)[：:\s]*", "", raw).strip()
    if not raw:
        return None

    # 判断题(A. 对 B. 错)且答案以非选项字母开头（如 C. 对）→ 按语义回引选项。
    if len(labels) == 2:
        tf_in_text = re.search(r"(对|正确|错|错误)", raw)
        if tf_in_text:
            word = tf_in_text.group(1)
            is_true = word in ("对", "正确")
            for letter, content in labels.items():
                content_l = content.casefold().strip()
                if is_true and content_l in ("对", "正确", "是", "true", "yes"):
                    return letter
                if not is_true and content_l in ("错", "错误", "否", "false", "no"):
                    return letter

    letter = normalize_answer(raw, question.options)
    if letter:
        # 仅接受题干真实存在的选项字母（如对/错题只含 A/B，C/D 视为无效）。
        if all(ch in labels for ch in letter):
            return letter
        return None

    # 答案直接是选项文本 → 回映射到对应字母。
    compact = re.sub(r"[\s,，、/。；;:：]+", "", raw).casefold()
    for label, content in labels.items():
        if re.sub(r"[\s,，、/。；;:：]+", "", content).casefold() == compact:
            return label
    return None


def sync_answers(
    questions: list[QuestionRecord],
    ai_answers: list[AnswerRecord],
    gold_answers: list[AnswerRecord],
) -> tuple[list[SyncDecision], dict]:
    """执行同步纠错，返回 (决策列表, 统计)。"""
    questions_by_number = {q.number: q for q in questions}
    ai_by_number = {a.number: a for a in ai_answers}
    gold_by_number = {g.number: g for g in gold_answers}

    decisions: list[SyncDecision] = []
    for number in sorted(questions_by_number):
        question = questions_by_number[number]
        ai = ai_by_number.get(number)
        ai_raw = ai.raw_answer if ai else ""
        ai_norm = normalize(ai_raw, question) if ai else None

        gold = gold_by_number.get(number)
        gold_raw = gold.raw_answer if gold else ""
        gold_norm = normalize(gold_raw, question) if gold else None

        if gold_norm is not None:
            final = gold_norm
            if ai_norm == gold_norm:
                status, note = "correct", "AI 与权威一致"
            elif ai_norm is not None:
                status, note = "wrong", "AI 有误，已按权威纠错"
            else:
                status, note = "corrected", "AI 无效，已按权威补齐"
        elif ai_norm is not None:
            final, status, note = ai_norm, "untouched", "仅规范化，无权威答案比照"
        else:
            final, status, note = "", "unsupported", "无权威答案且 AI 答案无效"

        decisions.append(
            SyncDecision(
                number=number,
                question_type=question.question_type,
                ai_answer=ai_raw,
                ai_normalized=ai_norm,
                gold_answer=gold_raw,
                gold_normalized=gold_norm,
                final_answer=final,
                status=status,
                note=note,
            )
        )

    counts = {status: 0 for status in ("correct", "wrong", "corrected", "unsupported", "untouched")}
    for d in decisions:
        counts[d.status] += 1

    summary = {
        "questions": len(questions),
        "ai_answers": len(ai_answers),
        "gold_answers": len(gold_answers),
        "forced_corrections": counts["wrong"] + counts["corrected"],
        "with_gold": bool(gold_answers),
        "accuracy": None if not gold_answers else round(counts["correct"] / max(counts["correct"] + counts["wrong"], 1), 4),
        **counts,
    }
    return decisions, summary


def write_outputs(
    decisions: list[SyncDecision],
    output_dir: Path,
    report_name: str,
    text_name: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 可直接用于提交的答案（仅保留有效项）。
    text_lines = [
        f"{d.number}. {d.final_answer}"
        for d in decisions
        if d.final_answer
    ]
    text_path = output_dir / text_name
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    payload = {
        "schema_version": 2,
        "summary": {
            "questions": len(decisions),
            "final_answers": sum(1 for d in decisions if d.final_answer),
            "corrected": sum(1 for d in decisions if d.status == "corrected"),
            "wrong_corrected": sum(1 for d in decisions if d.status == "wrong"),
        },
        "results": [d.__dict__ for d in decisions],
    }
    report_path = output_dir / report_name
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"answers": text_path, "report": report_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线答案同步纠错：AI 答案与权威答案校对并生成可提交结果")
    parser.add_argument("questions", type=Path, help="题目 TXT")
    parser.add_argument("--answers", type=Path, default=None, help="AI/候选答案文件（每行: 题号=答案，如 1=A）")
    parser.add_argument("--gold", type=Path, default=None, help="权威正确答案文件（可选，同上格式；提供后作批改与纠错基准）")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--report-name", default="answer_sync_report.json")
    parser.add_argument("--output-name", default="corrected_answers.txt")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        questions, _ = parse_questions(args.questions)
        if not questions:
            raise AnswerSyncError("没有解析出题目")
        answers: list[AnswerRecord] = []
        gold: list[AnswerRecord] = []
        if args.answers:
            answers, _ = parse_answers(args.answers)
        if args.gold:
            gold, _ = parse_answers(args.gold)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    decisions, summary = sync_answers(questions, answers, gold)
    paths = write_outputs(decisions, args.output_dir, args.report_name, args.output_name)

    print(f"题目: {summary['questions']} | AI答案: {summary['ai_answers']} | 权威答案: {summary['gold_answers']}")
    if summary.get("with_gold"):
        print(f"准确率: {summary['accuracy'] * 100:.1f}% | "
              f"一致: {summary['correct']} | 有误已纠: {summary['wrong']} | 补齐: {summary['corrected']}")
    else:
        print(f"规范化纠错: {summary['untouched']} | 待检(无效): {summary['unsupported']}")
    print(f"纠错后答案已写入: {paths['answers']}")
    print(f"明细报告已写入: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())