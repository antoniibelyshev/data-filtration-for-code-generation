from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn.functional import cross_entropy
import numpy as np


def compute_perplexity(
    samples: dict[str, list[str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, torch.Tensor]:
    input_texts = [problem + " " + solution for problem, solution in zip(samples["problem"], samples["solution"])]
    
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    logits = logits[:, :-1, :]
    labels = inputs["input_ids"][:, 1:]

    perplexity = cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none',
    ).reshape(logits.size(0), -1).mean(dim=1).exp()
    
    return {"perplexity": perplexity}


def compute_solution_perplexity(
    samples: dict[str, list[str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    problems = samples["problem"]
    solutions = samples["solution"]

    if max_length is None:
        max_length = model.config.n_positions if hasattr(model.config, "n_positions") else 512

    problem_tokens = tokenizer(problems, return_tensors="pt", padding=True, truncation=True, max_length=max_length // 2).to(model.device)
    solution_tokens = tokenizer(solutions, return_tensors="pt", padding=True, truncation=True, max_length=max_length // 2).to(model.device)
    
    input_ids = torch.cat([problem_tokens["input_ids"], solution_tokens["input_ids"]], dim=-1)
    
    labels = input_ids.clone()
    problem_length = problem_tokens["input_ids"].size(1)
    labels[:, :problem_length] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    
    logits = logits[:, problem_length:]
    labels = input_ids[:, problem_length:]

    perplexity = cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none',
    ).reshape(logits.size(0), -1).mean(dim=1).exp()
    
    return {"solution_perplexity": perplexity}
