# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores


def rlcr_score(
    reward_inputs: list[dict[str, Any]], brier_weight: float = 0.5, format_weight: float = 0.5
) -> list[dict[str, float]]:
    """
    Compute RLCR reward based on three components:
      1. Accuracy indicator I (0 or 1)
      2. Brier term: -(c - I)^2, where c is parsed confidence
      3. Format score for correct output structure (<think> + \boxed{} + <confidence>)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for math reward function.")

    scores = []

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        gt = reward_input["ground_truth"]

        # 1️⃣ Accuracy part (示性函数)
        I = 1.0 if accuracy_reward(response, gt) > 0.5 else 0.0  # or directly compute I

        # 2️⃣ Parse confidence from response
        conf_match = re.search(r"<confidence>(\d+(?:\.\d+)?)</confidence>", response)
        c = float(conf_match.group(1)) / 10.0 if conf_match else 0.5  # normalize to [0,1]

        # 3️⃣ Format score
        format_score = format_reward(response)

        # 4️⃣ Brier score term
        brier_score = -((c - I) ** 2)

        # 5️⃣ Combine overall score
        overall = I + brier_weight * brier_score + format_weight * format_score

        scores.append(
            {"overall": overall, "indicator": I, "brier": brier_score, "confidence": c, "format": format_score}
        )

    return scores


def rlcr_passk_score(
    reward_inputs: list[dict[str, Any]],
    brier_weight: float = 0.5,
    format_weight: float = 0.5,
) -> list[dict[str, float]]:
    """
    Compute RLCR reward based on:
      1️⃣ Accuracy indicator I (0 or 1)
      2️⃣ Brier term: -(c - mean_I)^2, where mean_I = average accuracy over batch
      3️⃣ Format score (<think> + \\boxed{} + <confidence>)
    """

    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for math reward function.")

    # First pass: compute all individual I values
    I_values = []
    responses = []
    ground_truths = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        gt = reward_input["ground_truth"]
        I = 1.0 if accuracy_reward(response, gt) > 0.5 else 0.0
        I_values.append(I)
        responses.append(response)
        ground_truths.append(gt)

    # Compute group average accuracy
    mean_I = sum(I_values) / len(I_values) if I_values else 0.0

    # Second pass: compute final scores using mean_I
    scores = []
    for i, response in enumerate(responses):
        I = I_values[i]

        # Parse confidence
        conf_match = re.search(r"<confidence>(\d+(?:\.\d+)?)</confidence>", response)
        c = float(conf_match.group(1)) / 10.0 if conf_match else 0.5  # normalize to [0,1]

        # Format score
        format_score = format_reward(response)

        # Brier term uses group-level mean accuracy
        brier_score = -((c - mean_I) ** 2)

        # Combine total
        overall = I + brier_weight * brier_score + format_weight * format_score

        scores.append(
            {
                "overall": overall,
                "indicator": I,
                "brier": brier_score,
                "confidence": c,
                "format": format_score,
                "mean_accuracy": mean_I,
            }
        )

    return scores
