import math
import re
import statistics
from typing import Any

import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer
from sklearn.metrics import roc_auc_score


def extract_confidence(response: str) -> float:
    """从 <confidence> 标签中提取置信度（1–10），并归一化到 [0,1]"""
    match = re.search(r"<confidence>(\d+(?:\.\d+)?)</confidence>", response)
    if match:
        c = float(match.group(1))
        return max(0.0, min((c / 10.0) ** 1.3, 1.0))  # 限制到 [0,1]
    return 0.5  # 若未提供则视为中性置信度


def format_reward(response: str) -> float:
    """
    检测是否包含且仅包含：
      <think>...</think> + \boxed{...} + <think>...</think> + <confidence>score</confidence>
    - 顺序必须正确
    - 每种标签只能出现一次
    - 允许前后和中间有任意文本
    - confidence ∈ [1, 10]
    """

    # 匹配出四个核心部分（按顺序）
    pattern = re.compile(
        r"<think>.*?</think>.*?"      # 第一个 think
        r"\\boxed\{.*?\}.*?"          # boxed
        r"<think>.*?</think>.*?"      # 第二个 think
        r"<confidence>(.*?)</confidence>",  # confidence 值
        re.DOTALL
    )

    match = pattern.search(response)  # ✅ 用 search 而不是 fullmatch
    if not match:
        print("❌ 未按要求顺序出现四个部分")
        return 0.0

    # 检查数量是否唯一
    think_count = len(re.findall(r"<think>.*?</think>", response, re.DOTALL))
    boxed_count = len(re.findall(r"\\boxed\{.*?\}", response))
    conf_count = len(re.findall(r"<confidence>.*?</confidence>", response, re.DOTALL))

    if not (think_count == 2 and boxed_count == 1 and conf_count == 1):
        print(f"❌ 数量不对: think={think_count}, boxed={boxed_count}, confidence={conf_count}")
        return 0.0

    # 提取 confidence 内容并验证数值范围
    try:
        conf_str = match.group(1).strip()
        conf_value = float(conf_str)
        if 1.0 <= conf_value <= 10.0:
            return 1.0  # 格式正确 + 合法范围
        else:
            return 0.5  # 格式正确但数值越界
    except ValueError:
        return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """提取第一个 boxed 内容并判定是否正确"""
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    brier_weight: float = 0.8,
    format_weight: float = 0.4,
) -> list[dict[str, float]]:
    """
    计算总 reward：
      total = I + brier_weight * ( - (c - I)^2 ) + format_weight * format_score
    并计算 batch 级别置信度指标（方差、AUROC、ECE 等）
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    c_list, I_list, format_list, processed_responses = [], [], [], []

    # === 提取各项指标 ===
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        gt = reward_input["ground_truth"]
        processed_responses.append((response, gt))
        c_list.append(extract_confidence(response))
        I_list.append(accuracy_reward(response, gt))
        format_list.append(format_reward(response))

    # === batch级别统计 ===
    if len(c_list) > 1:
        c_variance = statistics.variance(c_list)
    else:
        c_variance = 0.0

    # --- AUROC ---
    try:
        auroc = roc_auc_score(I_list, c_list)
    except ValueError:
        auroc = float("nan")  # 若只有一个类别则无法计算

    # --- ECE ---
    def compute_ece(confidences, accuracies, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            start, end = bins[i], bins[i + 1]
            idx = [j for j, c in enumerate(confidences) if start <= c < end]
            if not idx:
                continue
            avg_conf = np.mean([confidences[j] for j in idx])
            avg_acc = np.mean([accuracies[j] for j in idx])
            ece += abs(avg_conf - avg_acc) * len(idx) / len(confidences)
        return ece

    ece = compute_ece(c_list, I_list)

    # --- Confidence 统计信息 ---
    c_max = max(c_list)
    c_min = min(c_list)
    c_mean = statistics.mean(c_list)
    c_std = statistics.pstdev(c_list)
    c_entropy = -sum(c * math.log(c + 1e-8) + (1 - c) * math.log(1 - c + 1e-8) for c in c_list) / len(c_list)

    # === 每个样本的score ===
    for idx, (response, gt) in enumerate(processed_responses):
        I = I_list[idx]
        c = c_list[idx]
        format_score = format_list[idx]
        brier_score = -((c - I) ** 2)
        overall = I + brier_weight * brier_score + format_weight * format_score

        scores.append(
            {
                "overall": overall,
                "accuracy": I,
                "brier": brier_score,
                "confidence": c,
                "format": format_score,
                "batch_conf_variance": c_variance,
                "batch_conf_auroc": auroc,
                "batch_conf_ece": ece,
                "batch_conf_max": c_max,
                "batch_conf_min": c_min,
                "batch_conf_mean": c_mean,
                "batch_conf_std": c_std,
                "batch_conf_entropy": c_entropy,
            }
        )

    return scores


def rlcr_passk_score(
    reward_inputs: list[dict[str, Any]],
    brier_weight: float = 0.5,
    format_weight: float = 0.5,
    var_weight: float = 0.0,  # 可选项：如果你想让方差参与overall，可以调这个
) -> list[dict[str, float]]:
    """
    Compute RLCR reward based on:
      1️⃣ Accuracy indicator I (0 or 1)
      2️⃣ Brier term: -(c - mean_I)^2, where mean_I = average accuracy over batch
      3️⃣ Format score (<think> + \\boxed{} + <confidence>)
      4️⃣ Confidence variance term (用于分析或可加入overall)
    """

    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for math reward function.")

    # ---- First pass: compute individual accuracy I and collect confidences ----
    I_values, responses, confidences = [], [], []

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        gt = reward_input["ground_truth"]
        I = 1.0 if accuracy_reward(response, gt) > 0.5 else 0.0
        I_values.append(I)
        responses.append(response)

        # 提取confidence
        conf_match = re.search(r"<confidence>(\d+(?:\.\d+)?)</confidence>", response)
        c = float(conf_match.group(1)) / 10.0 if conf_match else 0.5
        confidences.append(c)

    # ---- Group statistics ----
    mean_I = sum(I_values) / len(I_values) if I_values else 0.0
    mean_c = sum(confidences) / len(confidences) if confidences else 0.0
    var_c = sum((c - mean_c) ** 2 for c in confidences) / len(confidences) if confidences else 0.0

    # ---- Second pass: compute final scores ----
    scores = []
    for i, response in enumerate(responses):
        I = I_values[i]
        c = confidences[i]

        format_score = format_reward(response)
        brier_score = -((c - mean_I) ** 2)

        # 如果想惩罚confidence方差，可以加进去
        overall = I + brier_weight * brier_score + format_weight * format_score - var_weight * var_c

        scores.append(
            {
                "overall": overall,
                "indicator": I,
                "brier": brier_score,
                "confidence": c,
                "format": format_score,
                "mean_accuracy": mean_I,
                "mean_confidence": mean_c,
                "confidence_variance": var_c,
            }
        )

    return scores
