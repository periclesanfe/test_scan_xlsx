#!/usr/bin/env python3
import argparse
import json
import os
import random

import cv2
import numpy as np

A4_WIDTH_PX = 1654
A4_HEIGHT_PX = 2339
QUESTION_COUNT = 5
OPTION_COUNT = 5

# Geometry tuned to the Hough-based fallback in utils.process_scan_image.
START_Y = 460
ROW_GAP = 220
START_X = 220
COL_GAP = 115
BUBBLE_RADIUS = 16


def draw_sheet(output_path, answers, rotate_deg=0.0, add_noise=False):
    img = np.full((A4_HEIGHT_PX, A4_WIDTH_PX, 3), 255, dtype=np.uint8)

    for q_index, option_index in enumerate(answers):
        y = START_Y + (q_index * ROW_GAP)
        for opt_idx in range(OPTION_COUNT):
            x = START_X + (opt_idx * COL_GAP)
            cv2.circle(img, (x, y), BUBBLE_RADIUS, (0, 0, 0), 2)
            if opt_idx == option_index:
                cv2.circle(img, (x, y), BUBBLE_RADIUS - 3, (0, 0, 0), -1)

    if rotate_deg:
        matrix = cv2.getRotationMatrix2D((A4_WIDTH_PX // 2, A4_HEIGHT_PX // 2), rotate_deg, 1.0)
        img = cv2.warpAffine(img, matrix, (A4_WIDTH_PX, A4_HEIGHT_PX), borderValue=(255, 255, 255))

    if add_noise:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OMR dataset")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    test_spec = {
        "questions": [
            {"question_id": i + 1, "answer_type": "likert", "custom_options": ""}
            for i in range(QUESTION_COUNT)
        ]
    }

    samples = []
    for i in range(args.samples):
        answers_idx = [random.randint(0, OPTION_COUNT - 1) for _ in range(QUESTION_COUNT)]
        rotate_deg = random.choice([0.0, -2.0, 2.0, -4.0, 4.0])
        add_noise = random.random() < 0.4

        img_name = f"sample_{i:03d}.png"
        img_path = os.path.join(images_dir, img_name)
        draw_sheet(img_path, answers_idx, rotate_deg=rotate_deg, add_noise=add_noise)

        expected = {str(j + 1): str(answers_idx[j] + 1) for j in range(QUESTION_COUNT)}
        samples.append(
            {
                "image_path": os.path.join("images", img_name),
                "expected_answers": expected,
            }
        )

    dataset = {
        "test_spec": test_spec,
        "samples": samples,
    }
    with open(os.path.join(args.out_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"dataset saved at {os.path.join(args.out_dir, 'dataset.json')}")


if __name__ == "__main__":
    main()
