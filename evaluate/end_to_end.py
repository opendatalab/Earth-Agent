#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import os
from typing import Dict, List, Any, Tuple

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON file data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_answer_from_text(text: str) -> str:
    """Extract answer from text, supporting multiple formats"""
    if not text:
        return "FAIL"
    
    # Handle FAIL cases
    if "FAIL" in text:
        return "FAIL"
    
    # Extract <Answer>X<Answer> format answers
    pattern = r'<Answer>([A-F])<Answer>'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Extract <Answer>X</Answer> format answers
    pattern = r'<Answer>([A-F])</Answer>'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Find single A, B, C, D
    pattern = r'\b([A-F])\b'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].upper()  # Take the last match
    
    return "UNKNOWN"

def count_tool_calls(data: Dict) -> int:
    """Count tool calls"""
    tool_calls = data.get("tool_calls", [])
    return len(tool_calls)

def calculate_accuracy(ground_truth_data: List[Dict], predicted_data: List[Dict]) -> Dict:
    """Calculate accuracy"""
    # Create index mapping
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    pred_dict = {}
    for item in predicted_data:
        key = item["question_id"]
        # If prediction data key is pure number, convert to "questionX" format
        if key.isdigit():
            key = f"question{key}"
        pred_dict[key] = item
    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "correct_answers": 0,
        "fail_answers": 0,
        "unknown_answers": 0,
        "missing_predictions": [],
        "accuracy": 0.0,
        "detailed_results": []
    }

    for question_index, gt_item in list(gt_dict.items())[188:]:
        gt_answer = gt_item.get("final_answer", "")
        if gt_answer is None:
            gt_answer = ""
        gt_answer = gt_answer.strip()
        
        if question_index not in pred_dict:
            results["missing_predictions"].append(question_index)
            results["detailed_results"].append({
                "question_index": question_index,
                "ground_truth": gt_answer,
                "predicted": "MISSING",
                "correct": False,
                "status": "missing"
            })
            continue
        
        results["evaluated_questions"] += 1
        pred_item = pred_dict[question_index]
        # Try final_answer first, then fall back to polished_answer
        pred_answer_text = pred_item.get("final_answer") or pred_item.get("polished_answer", "")
        pred_answer = extract_answer_from_text(pred_answer_text)
        # Judge correctness
        is_correct = False
        status = "incorrect"
        
        if pred_answer == "FAIL":
            results["fail_answers"] += 1
            status = "fail"
        elif pred_answer == "UNKNOWN":
            results["unknown_answers"] += 1
            status = "unknown"
        
        elif pred_answer == gt_answer:
            results["correct_answers"] += 1
            is_correct = True
            status = "correct"
        
        results["detailed_results"].append({
            "question_index": question_index,
            "ground_truth": gt_answer,
            "predicted": pred_answer,
            "correct": is_correct,
            "status": status
        })
    
    # Calculate accuracy
    if results["evaluated_questions"] > 0:
        results["accuracy"] = results["correct_answers"] / results["evaluated_questions"]
    
    return results

def load_model_tool_calls(extracted_tool_calls_path: str) -> Dict:
    """Load model tool calls data"""
    try:
        data = load_json_data(extracted_tool_calls_path)
        # Convert to dictionary format, key as question_index
        tool_calls_dict = {}
        for item in data:
            key = item["question_index"]
            if key.isdigit():
                key = f"question{key}"
            tool_calls_dict[key] = item
        return tool_calls_dict
    except FileNotFoundError:
        print(f"Warning: Tool calls file not found: {extracted_tool_calls_path}")
        return {}

def calculate_efficiency_with_tool_calls(ground_truth_data: List[Dict], 
                                       model_tool_calls_data: Dict) -> Dict:
    """
    Calculate efficiency: model tool calls count / ground truth tool calls count
    Value > 1 means model used more tools than needed
    Value < 1 means model used fewer tools than expected
    """
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    
    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "efficiency_scores": [],
        "average_efficiency": 0.0,
        "detailed_results": []
    }

    for question_index, gt_item in list(gt_dict.items())[188:]:
        gt_tool_count = count_tool_calls(gt_item)
        
        if question_index not in model_tool_calls_data:
            results["detailed_results"].append({
                "question_index": question_index,
                "gt_tool_count": gt_tool_count,
                "model_tool_count": 0,
                "efficiency": 0.0,
                "status": "missing"
            })
            continue
        
        results["evaluated_questions"] += 1
        model_item = model_tool_calls_data[question_index]
        model_tool_count = count_tool_calls(model_item)
        
        # Calculate efficiency ratio
        if gt_tool_count == 0:
            efficiency = 1.0 if model_tool_count == 0 else float('inf')  # If ground truth has no tools but model has tools, efficiency is infinity
        else:
            # Efficiency = model tools / ground truth tools
            # If model uses more tools to complete task, efficiency > 1
            # If model uses fewer tools to complete task, efficiency < 1
            efficiency = model_tool_count / gt_tool_count
        
        results["efficiency_scores"].append(efficiency)
        results["detailed_results"].append({
            "question_index": question_index,
            "gt_tool_count": gt_tool_count,
            "model_tool_count": model_tool_count,
            "efficiency": efficiency,
            "status": "evaluated"
        })
    
    # Calculate average efficiency
    if results["efficiency_scores"]:
        results["average_efficiency"] = sum(results["efficiency_scores"]) / len(results["efficiency_scores"])
    
    return results

def run_end_to_end_evaluation(ground_truth_file: str, 
                            predicted_answers_file: str,
                            model_tool_calls_file: str = None) -> Dict:
    """Run complete end-to-end evaluation"""
    print("Loading data")
    ground_truth_data = load_json_data(ground_truth_file)
    predicted_data = load_json_data(predicted_answers_file)
    
    print("Calculating accuracy")
    accuracy_results = calculate_accuracy(ground_truth_data, predicted_data)
    
    efficiency_results = {}
    if model_tool_calls_file:
        print("Loading tool calls data")
        model_tool_calls_data = load_model_tool_calls(model_tool_calls_file)
        print("Calculating efficiency")
        efficiency_results = calculate_efficiency_with_tool_calls(ground_truth_data, model_tool_calls_data)
    
    return {
        "accuracy": accuracy_results,
        "efficiency": efficiency_results,
        "summary": {
            "total_questions": accuracy_results["total_questions"],
            "accuracy_rate": accuracy_results["accuracy"],
            "average_efficiency": efficiency_results.get("average_efficiency", 0.0)
        }
    }

def print_evaluation_summary(results: Dict, model_name: str = ""):
    """Print evaluation summary"""
    accuracy = results["accuracy"]
    efficiency = results["efficiency"]
    summary = results["summary"]

    print("=" * 60)
    if model_name:
        print(f"End-to-End Evaluation Summary - {model_name}")
    else:
        print("End-to-End Evaluation Summary")
    print("=" * 60)
    print(f"Total questions: {accuracy['total_questions']}")
    print(f"Evaluated questions: {accuracy['evaluated_questions']}")
    print(f"Missing predictions: {len(accuracy['missing_predictions'])}")

    print("\nAccuracy statistics:")
    print("-" * 40)
    print(f"Correct answers: {accuracy['correct_answers']}")
    print(f"Incorrect answers: {accuracy['evaluated_questions'] - accuracy['correct_answers'] - accuracy['fail_answers'] - accuracy['unknown_answers']}")
    print(f"FAIL answers: {accuracy['fail_answers']}")
    print(f"Unknown answers: {accuracy['unknown_answers']}")
    print(f"Accuracy rate: {accuracy['accuracy']:.4f} ({accuracy['accuracy']*100:.2f}%)")

    if efficiency:
        print("\nEfficiency statistics:")
        print("-" * 40)
        print(f"Average efficiency: {efficiency['average_efficiency']:.4f}")
        print(f"Efficiency evaluated questions: {efficiency['evaluated_questions']}")

    if accuracy['missing_predictions']:
        print(f"\nMissing questions: {accuracy['missing_predictions'][:10]}{'...' if len(accuracy['missing_predictions']) > 10 else ''}")

    print("=" * 60)

def find_model_directories(root_dir: str) -> List[str]:
    """Find all model directories that contain the required files"""
    model_dirs = []

    if not os.path.exists(root_dir):
        print(f"Error: Root directory does not exist: {root_dir}")
        return model_dirs

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Check if both required files exist
            results_file = os.path.join(item_path, "results_summary_polished.json")
            tool_calls_file = os.path.join(item_path, "extracted_tool_calls.json")

            if os.path.exists(results_file) and os.path.exists(tool_calls_file):
                model_dirs.append(item_path)
                print(f"Found model directory: {item}")
            else:
                missing_files = []
                if not os.path.exists(results_file):
                    missing_files.append("results_summary_polished.json")
                if not os.path.exists(tool_calls_file):
                    missing_files.append("extracted_tool_calls.json")
                print(f"Skipping {item}: missing {', '.join(missing_files)}")

    return sorted(model_dirs)

def run_batch_evaluation(root_dir: str, ground_truth_file: str) -> Dict[str, Dict]:
    """Run evaluation for all models in the root directory"""
    model_dirs = find_model_directories(root_dir)

    if not model_dirs:
        print("No valid model directories found")
        return {}

    print(f"\nFound {len(model_dirs)} model directories to evaluate")
    print("-" * 60)

    all_results = {}
    summary_results = []

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"\nEvaluating model: {model_name}")
        print("-" * 40)

        predicted_answers_file = os.path.join(model_dir, "results_summary_polished.json")
        model_tool_calls_file = os.path.join(model_dir, "extracted_tool_calls.json")

        try:
            # Run evaluation for this model
            results = run_end_to_end_evaluation(
                ground_truth_file,
                predicted_answers_file,
                model_tool_calls_file
            )

            all_results[model_name] = results

            # Print summary for this model
            print_evaluation_summary(results, model_name)

            # Save individual results
            output_file = os.path.join(model_dir, "end_to_end_evaluation_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to: {output_file}")

            # Collect summary for final comparison
            summary_results.append({
                "model_name": model_name,
                "accuracy": results["summary"]["accuracy_rate"],
                "efficiency": results["summary"]["average_efficiency"],
                "total_questions": results["summary"]["total_questions"]
            })

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

    # Print final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON - ALL MODELS")
    print("=" * 70)
    print(f"{'Model Name':<30} {'Efficiency':<12} {'Accuracy':<10}")
    print("-" * 70)

    for summary in sorted(summary_results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{summary['model_name']:<30} {summary['efficiency']:.4f}      {summary['accuracy']*100:.2f}%")

    print("=" * 70)

    return all_results

def main():
    """Main function - run end-to-end evaluation for all models"""
    # Configuration
    root_dir = "./evaluate_langchain"
    ground_truth_file = "./extracted_tool_calls_GT.json"

    print("Starting batch end-to-end evaluation")
    print(f"Root directory: {root_dir}")
    print(f"Ground truth file: {ground_truth_file}")

    try:
        # Run batch evaluation
        all_results = run_batch_evaluation(root_dir, ground_truth_file)

        if all_results:
            # Save combined results
            combined_output_file = "./evaluate/evaluate/batch_evaluation_results.json"
            with open(combined_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            print("Batch evaluation completed")
            print(f"Combined results saved to: {combined_output_file}")
            print(f"Individual results saved in each model directory")
        else:
            print("No models were successfully evaluated")

        return all_results

    except Exception as e:
        print(f"Error during batch evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_single_model():
    """Alternative function to evaluate a single model (original functionality)"""
    # File paths for single model evaluation
    ground_truth_file = "./extracted_tool_calls_GT.json/extracted_tool_calls_GT.json"
    predicted_answers_file = "./evaluate_langchain/deepseek-V3_1_AP/results_summary_polished.json"
    model_tool_calls_file = "./evaluate_langchain/deepseek-V3_1_AP/extracted_tool_calls.json"

    print("Starting single model evaluation")

    try:
        # Run evaluation
        results = run_end_to_end_evaluation(
            ground_truth_file,
            predicted_answers_file,
            model_tool_calls_file
        )

        # Print summary
        print_evaluation_summary(results)

        # Save detailed results
        output_file = "./evaluate/end_to_end_evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nDetailed evaluation results saved to: {output_file}")

        return results

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()