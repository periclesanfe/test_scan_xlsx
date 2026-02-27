#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass

from utils import process_scan_image


@dataclass
class FakeQuestion:
    id: int
    answer_type: str
    custom_options: str

    def get_options(self):
        if self.answer_type == "yes_no":
            return ["Sim", "Não"]
        if self.answer_type == "likert":
            return ["1", "2", "3", "4", "5"]
        return [o.strip() for o in self.custom_options.split(",") if o.strip()]


@dataclass
class FakeTestQuestion:
    question_id: int
    question: FakeQuestion


@dataclass
class FakeTest:
    questions: list


def build_fake_test(spec):
    test_questions = []
    for row in spec["questions"]:
        q = FakeQuestion(
            id=int(row["question_id"]),
            answer_type=row["answer_type"],
            custom_options=row.get("custom_options", ""),
        )
        test_questions.append(FakeTestQuestion(question_id=q.id, question=q))
    return FakeTest(questions=test_questions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMR accuracy against a golden dataset.")
    parser.add_argument("dataset", help="Path to dataset JSON")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(args.dataset))
    test = build_fake_test(dataset["test_spec"])

    total_expected = 0
    total_correct = 0
    per_question = {}

    for sample in dataset["samples"]:
        img_path = sample["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(base_dir, img_path)

        detected = process_scan_image(img_path, test)
        expected = {int(k): v for k, v in sample["expected_answers"].items()}

        for qid, exp in expected.items():
            got = detected.get(qid)
            total_expected += 1
            row = per_question.setdefault(qid, {"total": 0, "correct": 0})
            row["total"] += 1
            if got == exp:
                total_correct += 1
                row["correct"] += 1

    accuracy = (total_correct / total_expected) * 100 if total_expected else 0.0
    print(f"samples={len(dataset['samples'])}")
    print(f"answers_total={total_expected}")
    print(f"answers_correct={total_correct}")
    print(f"accuracy={accuracy:.2f}%")
    print("per_question:")
    for qid in sorted(per_question.keys()):
        row = per_question[qid]
        pct = (row["correct"] / row["total"]) * 100 if row["total"] else 0
        print(f"  q{qid}: {row['correct']}/{row['total']} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
