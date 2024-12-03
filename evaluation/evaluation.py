import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm

def evaluate_humaneval_chrf(
    checkpoint_path: str,
    dataset_name: str,
    tokenizer_checkpoint: str | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    print("Downloading model...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint or checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
    print("Downloading dataset...")
    dataset = load_dataset(dataset_name)
    test_cases = dataset['test']
    chrf_metric = evaluate.load('chrf')

    model.eval()

    predictions = []
    expected_outputs = []
    with torch.no_grad():
        for i, example in enumerate(tqdm(test_cases)):
            code_input = example['prompt']
            expected_output = example['canonical_solution']

            tokenized_input = tokenizer.encode_plus(code_input, return_tensors='pt')
            inputs = tokenized_input['input_ids'].to(device)
            attention_mask = tokenized_input['attention_mask'].to(device)
            outputs = model.generate(
                inputs,
                attention_mask = attention_mask,
                # max_length=len(inputs[0]) + 50
                max_new_tokens = 50,
                pad_token_id = tokenizer.__dict__.get("pad_token_id", tokenizer.eos_token_id)
            )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(output_text)
            expected_outputs.append([expected_output])

    chrf = chrf_metric.compute(predictions=predictions, references=expected_outputs)

    print(f'ChrF Score: {chrf}')


if __name__ == "__main__":
    evaluate_humaneval_chrf("Salesforce/codegen-350M-mono", "openai_humaneval")
