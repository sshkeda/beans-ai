# accelerate launch grpo_cracked.py
# this will not run bc i only have 2x 3090s :(

from typing import Dict, List
from trl import GRPOConfig, GRPOTrainer
import os
from utils.reward_function import calculate_reward
import json
from accelerate import Accelerator

accelerator = Accelerator()

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="DEBUG")

BASE_MODEL = "../../../LLaMA-Factory/models/beans000-1.5B"
OUTPUT_DIR = "models/beans001-1.5B-GRPO"
DATASET_PATH = "datasets/beans001_grpo_train.json"
MERGE_OUTPUT_DIR = "models/beans001-1.5B"

MAX_COMPLETION_LENGTH = 1024
MAX_PROMPT_LENGTH = 401
MAX_SEQ_LENGTH = MAX_COMPLETION_LENGTH + MAX_PROMPT_LENGTH
GPU_MEMORY_UTILIZATION = 0.2

PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_GENERATIONS = 8
SAVE_STEPS = 50


def load_training_data() -> List[Dict]:
    raw_data = json.load(open(DATASET_PATH))

    formatted_data = []
    for item in raw_data:
        formatted_item = {
            "prompt": [{"role": "user", "content": item["prompt"]}],
            "target": item["target"],
            "question_type": item["question_type"],
            "fen": item["fen"],
            "id": item["id"],
        }
        formatted_data.append(formatted_item)

    logger.info(f"Dataset loaded and formatted with {len(formatted_data)} examples")
    return formatted_data


def create_training_args() -> GRPOConfig:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        bf16=True,
        gradient_checkpointing=True,
        learning_rate=1e-6,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=1,
        max_steps=-1,
        save_steps=SAVE_STEPS,
        report_to="tensorboard",
        resume_from_checkpoint=True,
        output_dir=OUTPUT_DIR,
    )

    logger.info(f"Training configuration created. Output directory: {OUTPUT_DIR}")
    return training_args


def beans_reward_func(completions, target, question_type, fen, **kwargs) -> list[float]:
    responses = ["<think>\n" + completion[0]["content"] for completion in completions]

    return [
        calculate_reward(
            llm_response=r,
            target=t,
            question_type=q_t,
            fen=f,
        )
        for r, t, q_t, f in zip(responses, target, question_type, fen)
    ]


def main():
    dataset = load_training_data()
    training_args = create_training_args()

    logger.info("Initializing GRPO Trainer")

    trainer = GRPOTrainer(
        model=BASE_MODEL,
        reward_funcs=[beans_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")

    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
