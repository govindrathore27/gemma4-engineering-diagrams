# shared/model/train.py
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class TrainConfig:
    project: str
    data_path: str
    output_dir: str
    base_model: str = "google/gemma-4-e4b-it"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    max_seq_len: int = 1024
    lora_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

    def __post_init__(self) -> None:
        if not self.data_path or not self.data_path.strip():
            raise ValueError("TrainConfig.data_path must be a non-empty string")
        if not self.output_dir or not self.output_dir.strip():
            raise ValueError("TrainConfig.output_dir must be a non-empty string")


def _format_row(row: dict) -> str:
    return (
        f"<start_of_turn>user\n{row['instruction']}\n{row['input']}<end_of_turn>\n"
        f"<start_of_turn>model\n{row['output']}<end_of_turn>"
    )


def _patch_quantization_flags(model) -> None:
    """Clear bnb quantization flags so accelerate allows multi-device models.

    Unsloth always loads the pre-quantized 4-bit variant regardless of
    load_in_4bit=False. accelerate refuses to prepare quantized models that
    span multiple devices, but the per-layer bnb ops still work correctly
    on their assigned device once the guard is removed.
    """
    for obj in [model, getattr(model, "model", None), getattr(model, "base_model", None)]:
        if obj is None:
            continue
        for attr in ("is_loaded_in_4bit", "is_loaded_in_8bit"):
            if getattr(obj, attr, False):
                setattr(obj, attr, False)


def train(cfg: TrainConfig) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = num_gpus > 1
    device_map = "balanced" if use_multi_gpu else {"": 0}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=True,
        device_map=device_map,
    )

    if use_multi_gpu:
        _patch_quantization_flags(model)

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    if use_multi_gpu:
        _patch_quantization_flags(model)

    torch.cuda.empty_cache()

    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {cfg.data_path!r}. "
            "Run the data pipeline first to generate the JSONL file."
        )

    with open(data_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    dataset = Dataset.from_dict({"text": [_format_row(r) for r in rows]})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_len,
        args=TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
            optim="paged_adamw_8bit",
            ddp_find_unused_parameters=False if use_multi_gpu else None,
        ),
    )
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Adapter saved to {cfg.output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    train(TrainConfig(project=args.project, data_path=args.data, output_dir=args.output))
