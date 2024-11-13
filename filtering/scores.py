import os
from collections import Counter

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import openai
from torch.nn.functional import cross_entropy


client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

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

def compute_ice_score(samples: dict[str, list[str]], aspect: str = "correctness", model = "gpt-4o-mini") -> dict[str, torch.Tensor]:
    ice_scores = []
    for problem, solution in zip(samples["problem"], samples["solution"]):
        ice_score = compute_ice_score_sample(problem, solution, aspect, model)
        ice_scores.append(ice_score)
    return {"ice_score": torch.tensor(ice_scores)}

def compute_ice_score_sample(problem: str, output: str, aspect: str = "correctness", model: str = "gpt-4o-mini") -> int:
    with open(f"./filtering/prompts_template/system_prompt_ice_score_{aspect}.txt", "r") as f:
        prompt = f.read()
    prompt = prompt.replace("{{PROBLEM}}", problem).replace("{{OUTPUT}}", output)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
    )
    raw_content = response.choices[0].message.content
    ice_score = get_gpt_answer(raw_content=raw_content, aspect=aspect)

    return ice_score

# parsing function from ICE-score paper
def get_gpt_answer(raw_content: str, aspect: str) -> int:
    """
    Extracts the GPT answer from the raw content.

    Args:
        raw_content (str): The raw content from GPT response.
        aspect (str): The evaluation aspect.

    Returns:
        int: The extracted answer as an integer.
    """
    try:
        return int(raw_content)
    except ValueError:
        try:
            return process_raw_content(raw_content, aspect)
        except:
            return 0


# parsing function from ICE-score paper
def process_raw_content(content: str, aspect: str) -> int:
    """
    Processes the raw content to extract the answer.

    Args:
        content (str): The raw content from GPT response.
        aspect (str): The evaluation aspect.

    Returns:
        int: The extracted answer as an integer.
    """
    # Normalize content: lowercase, remove parentheses, and split into lines
    splits = content.lower().replace("(", "").replace(")", "").split("\n")

    # Extract lines containing "score", remove dots, and replace "out of" and "/4"
    ls = [
        ll.strip(".").replace("out of ", "/").replace("/4", "")
        for l in splits
        for ll in l.lstrip('0123456789. ').split(". ")
        if any(item in ll for item in ["score"] + aspect.split())
    ]

    # Extract all numeric characters in each line and store them in a list
    ans = [ll for l in ls for ll in l.split() if ll.isnumeric()]

    # If there are multiple answers, return the most common one
    if len(set(ans)) != 1 and len(ans) > 1:
        return int(Counter(ans).most_common(1)[0][0])

    # Handle special cases where there are no answers or multiple non-numeric answers
    if len(set(ans)) != 1:
        if "N/A" in content:
            return 0

    # Return the single numeric answer
    return int(ans[0])

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
