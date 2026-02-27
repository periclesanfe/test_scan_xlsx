#!/usr/bin/env python3
import argparse
import json
import os
import random

import cv2
import numpy as np


def draw_sheet(output_path, answers, rotate_deg=0.0, add_noise=False):
    w, h = 1654, 2339
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Fiducials (squares)
    size = 24
    for x, y in [(50, 50), (w - 50, 50), (50, h - 50), (w - 50, h - 50)]:
        cv2.rectangle(img, (x - size // 2, y - size // 2), (x + size // 2, y + size // 2), (0, 0, 0), -1)

    start_y = 300
    row_gap = 180
    start_x = 250
    col_gap = 180
    radius = 28

    for q_index, option_index in enumerate(answers, start=1):
        y = start_y + (q_index - 1) * row_gap
        for opt_idx in range(5):
            x = start_x + opt_idx * col_gap
            cv2.circle(img, (x, y), radius, (0, 0, 0), 2)
            if opt_idx == option_index:
                cv2.circle(img, (x, y), radius - 4, (0, 0, 0), -1)

    if rotate_deg:
        m = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_deg, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))

    if add_noise:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = noisy

    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OMR dataset")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    test_spec = {
        "questions": [
            {"question_id": 1, "answer_type": "likert", "custom_options": ""},
            {"question_id": 2, "answer_type": "likert", "custom_options": ""},
            {"question_id": 3, "answer_type": "likert", "custom_options": ""},
            {"question_id": 4, "answer_type": "likert", "custom_options": ""},
            {"question_id": 5, "answer_type": "likert", "custom_options": ""},
        ]
    }

    samples = []
    for i in range(args.samples):
        answers_idx = [random.randint(0, 4) for _ in range(5)]
        rotate_deg = random.choice([0.0, -2.0, 2.0, -4.0, 4.0])
        add_noise = random.choice([False, True])

        img_name = f"sample_{i:03d}.png"
        img_path = os.path.join(images_dir, img_name)
        draw_sheet(img_path, answers_idx, rotate_deg=rotate_deg, add_noise=add_noise)

        expected = {str(j + 1): str(answers_idx[j] + 1) for j in range(5)}
        samples.append({
            "image_path": os.path.join("images", img_name),
            "expected_answers": expected,
        })

    dataset = {
        "test_spec": test_spec,
        "samples": samples,
    }
    with open(os.path.join(args.out_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"dataset saved at {os.path.join(args.out_dir, 'dataset.json')}")


if __name__ == "__main__":
    main()
