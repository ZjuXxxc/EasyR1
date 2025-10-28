#!/bin/bash

set -x

MODEL_PATH=/root/autodl-tmp/model/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/rlcr_vlm_config.yaml \
    data.train_files=/root/autodl-tmp/data/ViRL39K/train_subset.parquet \
    data.val_files=/root/autodl-tmp/data/ViRL39K/val_subset.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=1028_vl_3b_virl39k_1 \
    trainer.n_gpus_per_node=2 \
    data.format_prompt=./examples/format_prompt/geo3k_rlcr.jinja \
    worker.rollout.tensor_parallel_size=2 \
    worker.reward.reward_function=./examples/reward_function/rlcr_vlm.py:compute_score
