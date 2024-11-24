from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import DatasetDict
from typing import Any, Callable
import wandb


def get_data_collator(tokenizer: AutoTokenizer) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def data_collator(features):
        inputs = [f"{example['problem']} {tokenizer.sep_token} {example['solution']}" for example in features]

        batch = tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = batch["input_ids"].clone()

        for i, example in enumerate(features):
            problem = f"{example['problem']} {tokenizer.sep_token}"
            problem_length = len(tokenizer(problem, add_special_tokens=False)["input_ids"])

            labels[i, :problem_length] = -100

        batch["labels"] = labels
        return batch

    return data_collator


DEFAULT_TRAINING_ARGS = {
    "output_dir": "./codegen-causal-lm",
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    "save_steps": 1,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "logging_dir": "./logs",
    "logging_steps": 1,
    "logging_strategy": "steps",
    "evaluation_strategy": "epoch",
    "eval_steps": 1,
    "report_to": "wandb",
    "run_name": "codegen",
    "remove_unused_columns": False,
    "load_best_model_at_end": True
}


def finetune(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: DatasetDict,
    project_name: str = "dataset_filtration",
    **training_args: Any,
) -> AutoModelForCausalLM:
    wandb.init(project=project_name)

    combined_training_args = {**DEFAULT_TRAINING_ARGS, **training_args}
    wandb.config.update(combined_training_args)
    training_args = TrainingArguments(**combined_training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        data_collator=get_data_collator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    wandb.finish()

    return model
