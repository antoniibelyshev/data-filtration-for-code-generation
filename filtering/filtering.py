from datasets import Dataset


def filter_dataset(
    dataset: Dataset,
    score: str,
    p: float,
    part: str,
) -> Dataset:
    if part not in {"low", "mid", "high"}:
        raise ValueError("part must be one of 'low', 'mid', or 'high'")
    if not (0 < p <= 1):
        raise ValueError("p must be a float between 0 and 1")
    
    sorted_dataset = dataset.sort(score)
    num_samples = len(dataset)
    num_to_select = int(p * num_samples)

    if part == "low":
        start_idx = 0
        end_idx = num_to_select
    elif part == "mid":
        start_idx = (num_samples - num_to_select) // 2
        end_idx = start_idx + num_to_select
    elif part == "high":
        start_idx = num_samples - num_to_select
        end_idx = num_samples
    
    filtered_dataset = sorted_dataset.select(range(start_idx, end_idx))
    
    return filtered_dataset
