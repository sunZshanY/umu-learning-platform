# -*- coding: utf-8 -*-
"""测试 answer_sync 离线答案同步纠错的核心逻辑（不联网）。"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from answer_sync import normalize, parse_questions, parse_answers, sync_answers

TF_Q = """Q1. 商品标题中可以多次出现同一个词以增加权重。( )（单选，1分）
A.对
B.错
Q2. 上传了3:4的主图视频，必须上传＿的主图。( )（单选，1分）
A.1:1
B.3:4
C.4:3
D.16:9
Q3. 店铺招牌的尺寸大小是____。（单选，1分）
A.950像素×150像素
B.720像素×120像素
C.950像素×120像素
D.750像素×150像素
"""


def write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_normalize_tf_reference():
    """对/错题非选项字母 C 应被拒绝（无法自纠，留给权威）。"""
    tmp = Path(tempfile.mkdtemp())
    qpath = write_text(tmp / "q.txt", TF_Q)
    questions, _ = parse_questions(qpath)
    tf = questions[0]
    assert tf.options == {"A": "对", "B": "错"}
    assert normalize("C", tf) is None
    assert normalize("答案：A", tf) == "A"
    assert normalize("对", tf) == "A"


def test_normalize_option_text():
    """答案直接是选项文本（填空题带选项）应回映射到字母。"""
    tmp = Path(tempfile.mkdtemp())
    qpath = write_text(tmp / "q.txt", TF_Q)
    questions, _ = parse_questions(qpath)
    q3 = questions[2]
    assert normalize("950像素×150像素", q3) == "A"
    assert normalize("B", q3) == "B"
    assert normalize("不存在的内容", q3) is None


def test_sync_with_gold():
    """有权威答案时：纠正错误 AI 答案并标记一致性。"""
    tmp = Path(tempfile.mkdtemp())
    qpath = write_text(tmp / "q.txt", TF_Q)
    questions, _ = parse_questions(qpath)

    ai = write_text(tmp / "ai.txt", "1=B\n2=D\n3=B\n")
    gold = write_text(tmp / "gold.txt", "1=A\n2=D\n3=B\n")
    answers, _ = parse_answers(ai)
    gold_answers, _ = parse_answers(gold)

    decisions, summary = sync_answers(questions, answers, gold_answers)
    by_number = {d.number: d for d in decisions}

    assert by_number[1].status == "wrong"
    assert by_number[1].final_answer == "A"
    assert by_number[2].status == "correct"
    # Q3: AI=B 与权威 B 一致 → correct
    assert by_number[3].status == "correct"
    assert summary["with_gold"] is True
    assert summary["correct"] == 2
    assert summary["wrong"] == 1
    assert summary["corrected"] == 0


def test_sync_without_gold():
    """无权威文件时仅规范化，不伪造答案。"""
    tmp = Path(tempfile.mkdtemp())
    qpath = write_text(tmp / "q.txt", TF_Q)
    questions, _ = parse_questions(qpath)
    ai = write_text(tmp / "ai.txt", "1=A\n2=D\n3=D\n")
    answers, _ = parse_answers(ai)

    decisions, summary = sync_answers(questions, answers, [])
    by_number = {d.number: d for d in decisions}
    assert by_number[1].status == "untouched"
    assert by_number[1].final_answer == "A"
    assert by_number[3].status == "untouched"
    assert summary["ai_answers"] == 3
    assert summary["gold_answers"] == 0


if __name__ == "__main__":
    test_normalize_tf_reference()
    test_normalize_option_text()
    test_sync_with_gold()
    test_sync_without_gold()
    print("所有 answer_sync 测试通过")