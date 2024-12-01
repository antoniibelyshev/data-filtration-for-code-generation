import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm

def evaluate_humaneval_chrf(checkpoint_path: str, dataset_name: str):
    print("Downloading model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    print("Downloading dataset...")
    dataset = load_dataset(dataset_name)
    test_cases = dataset['test']
    chrf_metric = evaluate.load('chrf')

    model.eval()

    predictions = []
    expected_outputs = []
    with torch.no_grad():
        for i, example in tqdm(enumerate(test_cases)):
            code_input = example['prompt']
            expected_output = example['canonical_solution']

            inputs = tokenizer.encode(code_input, return_tensors='pt')
            outputs = model.generate(inputs, max_length=len(inputs[0]) + 50)

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(output_text)
            expected_outputs.append([expected_output])

    chrf = chrf_metric.compute(predictions=predictions, references=expected_outputs)

    print(f'ChrF Score: {chrf}')


if __name__ == "__main__":
    evaluate_humaneval_chrf("Salesforce/codegen-350M-mono", "openai_humaneval")
