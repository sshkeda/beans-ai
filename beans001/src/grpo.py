# ============================================================ #

CHECKPOINT_PATH = "models/beans002-1.5B-GRPO/checkpoint-1700"

# BASE_MODEL = "../../../LLaMA-Factory/models/beans000-1.5B"
BASE_MODEL = CHECKPOINT_PATH

NEW_MODEL = "beans003"

OUTPUT_DIR = f"models/{NEW_MODEL}-1.5B-GRPO"
DATASET_PATH = f"datasets/{NEW_MODEL}_train.json"
MERGE_OUTPUT_DIR = f"models/{NEW_MODEL}-1.5B"


MAX_COMPLETION_LENGTH = 2048
MAX_PROMPT_LENGTH = 99
MAX_SEQ_LENGTH = MAX_COMPLETION_LENGTH + MAX_PROMPT_LENGTH
GPU_MEMORY_UTILIZATION = 0.7

PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
NUM_GENERATIONS = 4
LEARNING_RATE = 2.5e-6
TEMPERATURE = 1.0
SAVE_STEPS = 50

LORA_RANK = 256


# ============================================================ #
from typing import Dict, Any, Tuple, List
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import os
from utils.reward_function import calculate_reward
import json


def setup_model() -> Tuple[Any, Any]:
    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=LORA_RANK,
    #     target_modules=[
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "o_proj",
    #         "gate_proj",
    #         "up_proj",
    #         "down_proj",
    #     ],
    #     lora_alpha=LORA_RANK*2,
    #     use_gradient_checkpointing="unsloth",
    #     random_state=42,
    # )

    print(f"Model loaded and configured with LoRA rank: {LORA_RANK}")
    return model, tokenizer


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

    print(f"Dataset loaded and formatted with {len(formatted_data)} examples")
    return formatted_data


def create_training_args() -> GRPOConfig:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=1,
        max_steps=-1,
        save_steps=SAVE_STEPS,
        max_grad_norm=0.1,
        report_to="tensorboard",
        resume_from_checkpoint=True,
        output_dir=OUTPUT_DIR,
    )

    print(f"Training configuration created. Output directory: {OUTPUT_DIR}")
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
    model, tokenizer = setup_model()
    dataset = load_training_data()
    training_args = create_training_args()

    print("Initializing GRPO Trainer")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[beans_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training")
    trainer.train()
    print("Training completed")

    model.save_pretrained_merged(
        MERGE_OUTPUT_DIR,
        tokenizer,
        save_method="lora",
    )
    print("Model saved")


if __name__ == "__main__":
    main()
