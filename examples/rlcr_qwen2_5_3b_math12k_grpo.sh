#!/bin/bash

set -x

MODEL_PATH=/root/autodl-tmp/model/Qwen2.5-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/rlcr_llm_config.yaml \
    data.train_files=/root/autodl-tmp/data/math12k/data/train-00000-of-00001.parquet \
    data.val_files=/root/autodl-tmp/data/math12k/data/test-00000-of-00001.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_3b_math12k_grpo \
    trainer.n_gpus_per_node=2 \
    data.format_prompt=./examples/format_prompt/math12k_rlcr.jinja \
    worker.rollout.tensor_parallel_size=2 \
    worker.reward.reward_function=./examples/reward_function/rlcr_llm.py:compute_score
