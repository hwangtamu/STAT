import torch
import json
import re
from tqdm import tqdm
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from peft import PeftModel
import os
import sys

import os
import json
from tqdm import tqdm
from transformers import GenerationConfig
import torch
# The answer parser and grader live in eval/, whose modules import each other by bare name
# (e.g. `from utils import *`), so that directory itself has to be on the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eval')))
import parser
import grader

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Llama 3 model with SFT and QLoRA.")

    parser.add_argument("--base_model_id", type=str, default="./models/Llama-3.2-3B-Instruct")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft")
    parser.add_argument("--output_filename", type=str, default="")
    parser.add_argument("--base_output_filename", type=str, default="")
    parser.add_argument("--run_base_model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1, help="generation temperature")
    parser.add_argument("--num_generated_responses", type=int, default=1, help="Number of responses to generate per prompt (for pass@k evaluation).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")

    return parser.parse_args()

class EvalConfig:
    # Model and tokenizer names
    BASE_MODEL_ID = "./models/Llama-3.2-3B-Instruct"
    # BASE_MODEL_ID = "./models/Qwen2.5-1.5B-Instruct"

    DATA_NAME = "original_train_multi_sol_9.5k"
    ADAPTER_PATH = f"./sft/finetuned_adapters/Llama-3.2-3B-Instruct/sft_{DATA_NAME}_5e-6_Llama-3.2-3B-Instruct/final_adapter"
    # OUTPUT_FILENAME = "eval_results_sft_7.5k_5e-06_Llama-3.2-3B-Instruct_lr5e-06_ep3.jsonl"
    OUTPUT_FILENAME = f"eval_results_math500_sft_{DATA_NAME}_5e-06_llama3.2_3b_instruct.jsonl"
    BASE_OUTPUT_FILENAME = "base_model_results_math.jsonl"
    # BASE_OUTPUT_FILENAME = "base_model_results_math500.jsonl"
    # Test dataset
    # TEST_DATA_PATH = "./sft/eval_data/amc23.jsonl"
    # TEST_DATA_PATH = "/scratch/gpfs/yh0068/slm-math/RLHF-Reward-Modeling/pipeline_outputs/skill_labeled/math/Llama-3.2-3B-Instruct/skills_labeled.jsonl"  
    TEST_DATA_PATH = "/scratch/gpfs/yh0068/slm-math/RLHF-Reward-Modeling/evaluation/data/math/test.jsonl"
    # Output directory for saving results
    OUTPUT_DIR = "./sft/evaluation_results"
    NUM_TEST_SAMPLES = 500
    NUM_GENERATED_RESPONSES = 1

    # Fixed instruction for all test questions
    INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}"
    
    # Generation parameters
    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.1 # Use low temperature for more deterministic, factual answers
    TOP_P = 0.9
    DO_SAMPLE = True

    # print(f"ADAPTER PATH: {ADAPTER_PATH}")

# ---------------------------------------------------------------------------------
# 3. Main Evaluation Logic
# ---------------------------------------------------------------------------------

def format_prompt(tokenizer, instruction, question):
    """
    Formats the prompt for a base model, consistent with fine-tuning.
    This should reflect how you structured your training data (e.g., "Instruction: X\nQuestion: Y\nAnswer:").
    """
    # Assuming your fine-tuning data was structured like:
    # "Instruction: {instruction}\nQuestion: {question}\nAnswer:"
    # This will make the model complete the "Answer:" part.
    # Adjust this string precisely to match your fine-tuning input format.
    prompt_text = f"{instruction}\n{question}" 
    # Or, if your fine-tuning data was just "Instruction here. Question here. Output here."
    # then perhaps: f"{instruction} {question} "
    
    return prompt_text # No chat template for base models


def get_question(example):
    """Test files name the question field either 'question' or 'problem'."""
    return example["question"] if "question" in example else example["problem"]


def evaluate_model(model, tokenizer, dataset, config, output_filename=None, overwrite=False):
    """
    Runs evaluation. If results file exists, it re-grades from the file.
    Otherwise, it runs model generation and saves the results.
    """
    # if not output_filename:
    #     output_filename = config.OUTPUT_FILENAME
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)
    
    # --- Check if results file already exists ---
    if os.path.exists(output_path):
        print(f"\n--- Found existing results at {output_path} ---")
        print("--- Re-running answer extraction and grading only ---")

        difficult_question = []
        question_to_answer = {}

        # print(dataset[0])
        for example in dataset:
            # if not example['parsed_output']['judge_pred'] or True not in example['score']:
            question = get_question(example)
            question_to_answer[question] = example["answer"]
            if 'score' in example and True not in example['score']:
                difficult_question.append(question)
        
        results = []
        with open(output_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        correct_predictions = 0
        total_predictions = 0
        total_correct_responses = 0
        total_responses = 0
        for result in tqdm(results, desc="Re-grading"):
            # if difficult_question and result['question'] not in difficult_question:
            #     continue
            if "gsm8k" in config.TEST_DATA_PATH:
                question = result['question']
                solution = question_to_answer[question]
                ground_truth_answer = str(parser.extract_answer(solution, 'gsm8k'))
                result['ground_truth'] = ground_truth_answer
                data_name = 'gsm8k'
            else:
                data_name = 'math'
            if "boxed" in result['ground_truth']:
                    result['ground_truth'] = str(parser.extract_answer(result['ground_truth'], 'math'))

            # Single-sample results hold one output per question; multi-sample results hold a list.
            responses = result['generated_responses'] if 'generated_responses' in result else [result]
            question_is_correct = False
            for response in responses:
                # Re-run the answer extraction on the stored model output
                extracted_answer = parser.extract_answer(response['full_model_output'], data_name)
                # Compare with the ground truth using the grader
                is_correct = grader.math_equal(extracted_answer, result['ground_truth'])
                # Update the result dictionary and count correct predictions
                response['extracted_answer'] = extracted_answer
                response['is_correct'] = is_correct
                if is_correct:
                    total_correct_responses += 1
                    question_is_correct = True
                total_responses += 1
            if 'generated_responses' in result:
                result['is_correct_any'] = question_is_correct
            if question_is_correct:
                correct_predictions += 1
            total_predictions += 1
        
        # Calculate and print the new accuracy
        accuracy = (correct_predictions / total_predictions) * 100
        if total_responses == total_predictions:
            print(f"Re-grading finished. New Accuracy: {accuracy:.2f}%")
        else:
            k = total_responses // total_predictions
            average_accuracy = (total_correct_responses / total_responses) * 100
            print(f"Re-grading finished. pass@{k}: {accuracy:.2f}% | average@{k}: {average_accuracy:.2f}%")
        
        # Overwrite the old file with the re-graded results
        if overwrite:
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
            print("Updated results file with new grading.")
        
        return accuracy

    # --- If file does not exist, run the full generation and evaluation ---
    else:
        print(f"\n--- Starting new evaluation for: {output_filename} ---")
        
        results = []
        correct_predictions = 0
        pass_k_correct_questions = 0
        total_correct_responses = 0
        total_responses_generated = 0
        
        generation_config = GenerationConfig(
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=config.NUM_GENERATED_RESPONSES,
        )
        
        for example in tqdm(dataset, desc="Evaluating"):
            question = get_question(example)
            
            if "gsm8k" in config.TEST_DATA_PATH:
                ground_truth_answer = str(parser.extract_answer(example["answer"], 'gsm8k'))
            elif "gt" in example:
                ground_truth_answer = str(example["gt"])
            elif "answer" in example:
                ground_truth_answer = str(example["answer"])
            if "boxed" in ground_truth_answer:
                ground_truth_answer = str(parser.extract_answer(ground_truth_answer, 'math'))
            prompt = format_prompt(tokenizer, config.INSTRUCTION, question)
            print("Prompt: ", prompt)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)
            
            # full_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # model_output = full_prediction[len(prompt.replace(tokenizer.bos_token, '')):]
            if config.NUM_GENERATED_RESPONSES == 1:
                input_token_len = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_token_len:]
                model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                print("Model output: ", model_output)
                
                extracted_answer = parser.extract_answer(model_output, 'math')
                
                # Note: Using math_equal directly, as requested in your original script.
                # If you have a 'grader' object, you'd use grader.math_equal here.
                is_correct = grader.math_equal(extracted_answer, ground_truth_answer) 

                print("Is_correct: ", is_correct)
                if not is_correct:
                    print(f"correct answer: {ground_truth_answer}, model answer: {extracted_answer}")
                if is_correct:
                    correct_predictions += 1
                    
                results.append({
                    "question": question,
                    "prompt": prompt,
                    "ground_truth": ground_truth_answer,
                    "full_model_output": model_output,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct
                })
            else:
                input_token_len = inputs.input_ids.shape[1]
                model_outputs = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)

                generated_responses_list = []
                question_is_correct_for_pass_k = False

                for model_output in model_outputs:
                    total_responses_generated += 1
                    extracted_answer = parser.extract_answer(model_output, 'math')
                    is_correct = grader.math_equal(extracted_answer, ground_truth_answer)
                    
                    if is_correct:
                        total_correct_responses += 1
                        question_is_correct_for_pass_k = True
                    
                    generated_responses_list.append({
                        "full_model_output": model_output,
                        "extracted_answer": extracted_answer,
                        "is_correct": is_correct
                    })
                
                if question_is_correct_for_pass_k:
                    pass_k_correct_questions += 1
                    
                results.append({
                    "question": question,
                    "prompt": prompt,
                    "ground_truth": ground_truth_answer,
                    "generated_responses": generated_responses_list,
                    "is_correct_any": question_is_correct_for_pass_k
                })
            
        if config.NUM_GENERATED_RESPONSES == 1:
            accuracy = (correct_predictions / len(dataset)) * 100
            print(f"Evaluation finished. Accuracy: {accuracy:.2f}%")
        else:
            k = config.NUM_GENERATED_RESPONSES
            accuracy = (pass_k_correct_questions / len(dataset)) * 100
            average_accuracy = (total_correct_responses / total_responses_generated) * 100
            print(f"Evaluation finished. pass@{k}: {accuracy:.2f}% | average@{k}: {average_accuracy:.2f}%")
        
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"Results saved to {output_path}")
        
        return accuracy

def main():
    args = get_args()
    config = EvalConfig()

    config.BASE_MODEL_ID = args.base_model_id
    config.DATA_NAME = args.data_name
    config.ADAPTER_PATH = args.adapter_path
    config.TEST_DATA_PATH = args.test_data_path

    config.OUTPUT_DIR = args.output_dir
    config.OUTPUT_FILENAME = args.output_filename
    config.BASE_OUTPUT_FILENAME = args.base_output_filename

    config.NUM_TEST_SAMPLES = args.num_test_samples
    config.NUM_GENERATED_RESPONSES = args.num_generated_responses
    config.TEMPERATURE = args.temperature

    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load the test dataset
    test_dataset = load_dataset("json", data_files=config.TEST_DATA_PATH, split="train")
    if config.NUM_TEST_SAMPLES != -1:
        test_dataset = test_dataset.select(range(min(len(test_dataset), config.NUM_TEST_SAMPLES)))

    # Load tokenizer (same for both models)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load only the model that is being evaluated ---
    # Generation is skipped (and no model is loaded) when the results file already exists;
    # in that case the stored outputs are just re-graded.
    output_filename = config.BASE_OUTPUT_FILENAME if args.run_base_model else config.OUTPUT_FILENAME
    model = None
    if not os.path.exists(os.path.join(config.OUTPUT_DIR, output_filename)):
        print("\nLoading Base Model...")
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # The adapter is merged into the base model's weights in place, so it must not be
        # loaded when the base model itself is the one being evaluated.
        if not args.run_base_model:
            if not os.path.isdir(config.ADAPTER_PATH):
                raise FileNotFoundError(
                    f"No adapter found at {config.ADAPTER_PATH}. Fine-tune the model first, "
                    "or pass --run_base_model to evaluate the base model."
                )
            print(f"\nMerging adapter weights from {config.ADAPTER_PATH} into base model...")
            model = PeftModel.from_pretrained(model, config.ADAPTER_PATH).merge_and_unload()

    evaluate_model(model, tokenizer, test_dataset, config, output_filename, args.overwrite)

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
