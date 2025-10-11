#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
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

    for question_index, gt_item in list(gt_dict.items())[:]:
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

    for question_index, gt_item in list(gt_dict.items())[:]:
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
            "average_efficiency": efficiency_results.get("average_efficiency", 0.0),
            "fail_rate": accuracy_results["fail_answers"] / accuracy_results["evaluated_questions"] if accuracy_results["evaluated_questions"] > 0 else 0.0
        }
    }

def print_evaluation_summary(results: Dict):
    """Print evaluation summary"""
    accuracy = results["accuracy"]
    efficiency = results["efficiency"]
    summary = results["summary"]
    
    print("=" * 60)
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
    print(f"Fail rate: {summary['fail_rate']:.4f} ({summary['fail_rate']*100:.2f}%)")
    
    if efficiency:
        print("\nEfficiency statistics:")
        print("-" * 40)
        print(f"Average efficiency: {efficiency['average_efficiency']:.4f}")
        print(f"Efficiency evaluated questions: {efficiency['evaluated_questions']}")
    
    if accuracy['missing_predictions']:
        print(f"\nMissing questions: {accuracy['missing_predictions'][:10]}{'...' if len(accuracy['missing_predictions']) > 10 else ''}")

    print("=" * 60)

def print_detailed_results(results: Dict):
    """Print detailed results for each question"""
    accuracy = results["accuracy"]
    efficiency = results["efficiency"]

    print("\n" + "=" * 70)
    print("Detailed Results for Each Question")
    print("=" * 70)
    print(f"{'Question':<12} {'GT_Answer':<10} {'Pred_Answer':<10} {'Accuracy':<10} {'GT_Tools':<10} {'Model_Tools':<10} {'Efficiency':<12}")
    print("-" * 70)

    # Create efficiency results index
    efficiency_dict = {}
    if efficiency and efficiency.get("detailed_results"):
        for eff_result in efficiency["detailed_results"]:
            efficiency_dict[eff_result["question_index"]] = eff_result

    # Traverse accuracy results
    for acc_result in accuracy["detailed_results"]:
        question_index = acc_result["question_index"]
        gt_answer = acc_result["ground_truth"]
        pred_answer = acc_result["predicted"]
        is_correct = "Correct" if acc_result["correct"] else "Incorrect"

        # Get efficiency data
        eff_result = efficiency_dict.get(question_index, {})
        gt_tools = eff_result.get("gt_tool_count", 0)
        model_tools = eff_result.get("model_tool_count", 0)
        efficiency_val = eff_result.get("efficiency", 0.0)

        if efficiency_val == float('inf'):
            efficiency_str = "inf"
        else:
            efficiency_str = f"{efficiency_val:.4f}"

        print(f"{question_index:<12} {gt_answer:<10} {pred_answer:<10} {is_correct:<10} {gt_tools:<10} {model_tools:<10} {efficiency_str:<12}")

    print("=" * 70)

def main():
    """Main function - run end-to-end evaluation"""
    # Specify model directory
    model_dir = "./evaluate_langchain/Kimik2_IF_25-09-21_07-08"

    # File paths
    ground_truth_file = "./extracted_tool_calls_GT.json"
    predicted_answers_file = f"{model_dir}/results_summary_polished.json"
    model_tool_calls_file = f"{model_dir}/extracted_tool_calls.json"
    
    print("Starting end-to-end evaluation")
    
    try:
        # Run evaluation
        results = run_end_to_end_evaluation(
            ground_truth_file, 
            predicted_answers_file,
            model_tool_calls_file
        )
        
        # Print summary
        print_evaluation_summary(results)

        # Print detailed results for each question
        print_detailed_results(results)
        
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