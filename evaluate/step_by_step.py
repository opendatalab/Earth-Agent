import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON file data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_tool_names_from_calls(tool_calls: List[Dict]) -> List[str]:
    """Extract tool names from tool calls list"""
    return [call.get("name", "") for call in tool_calls]

def find_tool_calls_from_data(data: Dict) -> List[str]:
    """Extract tool call name list from data"""
    tool_calls = data.get("tool_calls", [])
    return extract_tool_names_from_calls(tool_calls)

def contains_all_tool_calls_any_order(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """
    Check if all expected tool calls are contained (order not considered) - soft scoring version
    """
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)
    
    if not expected_tools:
        return {
            "score": 1.0,
            "key": "contains_all_tool_calls_any_order",
            "expected": expected_tools,
            "actual": actual_tools,
            "details": {"matched_tools": 0, "total_expected": 0}
        }
    
    # Calculate the number of matched tools
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    matched_tools = expected_set.intersection(actual_set)
    
    score = len(matched_tools) / len(expected_set)
    
    return {
        "score": score, 
        "key": "contains_all_tool_calls_any_order",
        "expected": expected_tools,
        "actual": actual_tools,
        "details": {
            "matched_tools": len(matched_tools),
            "total_expected": len(expected_set),
            "matched_tool_names": list(matched_tools)
        }
    }

def contains_all_tool_calls_in_order(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """
    Check if all expected tool calls are contained (considering order, but allowing other tools in between) - soft scoring version
    """
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)
    
    if not expected_tools:
        return {
            "score": 1.0,
            "key": "contains_all_tool_calls_in_order",
            "expected": expected_tools,
            "actual": actual_tools,
            "details": {"matched_in_order": 0, "total_expected": 0}
        }
    
    # Calculate the number of tools matched in order
    matched_count = 0
    actual_iter = iter(actual_tools)
    
    for expected_tool in expected_tools:
        # Try to find this tool in the remaining actual_tools
        found = False
        for actual_tool in actual_iter:
            if actual_tool == expected_tool:
                matched_count += 1
                found = True
                break
        if not found:
            # If not found, break the loop (strict order requirement)
            break
    
    score = matched_count / len(expected_tools)
    
    return {
        "score": score, 
        "key": "contains_all_tool_calls_in_order",
        "expected": expected_tools,
        "actual": actual_tools,
        "details": {
            "matched_in_order": matched_count,
            "total_expected": len(expected_tools)
        }
    }

def trajectory_step_wise_score(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """
    Calculate trajectory step-wise score (strict order matching)
    """
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)

    if not expected_tools:
        return {"score": 0, "key": "trajectory_step_wise"}

    # Calculate the number of correctly matched steps
    correct_steps = 0
    min_length = min(len(expected_tools), len(actual_tools))

    for i in range(min_length):
        if expected_tools[i] == actual_tools[i]:
            correct_steps += 1
        else:
            break  # Stop as soon as there's a mismatch (strict order)

    score = correct_steps / len(expected_tools) if len(expected_tools) > 0 else 0

    return {
        "score": score,
        "key": "trajectory_step_wise",
        "details": {
            "correct_steps": correct_steps,
            "total_expected": len(expected_tools),
            "expected_sequence": expected_tools,
            "actual_sequence": actual_tools[:len(expected_tools)]
        }
    }

def check_parameter_accuracy(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """
    Check the accuracy of tool call parameters - soft scoring version
    Match step by step, give i/total_steps score when matched to step i
    """
    predicted_calls = predicted_data.get("tool_calls", [])
    expected_calls = ground_truth_data.get("tool_calls", [])
    
    if not expected_calls:
        return {
            "score": 1.0,
            "key": "parameter_accuracy",
            "details": {
                "reason": "No expected tool calls",
                "expected_count": 0,
                "actual_count": len(predicted_calls),
                "matched_steps": 0,
                "total_expected_steps": 0
            }
        }
    
    total_expected_steps = len(expected_calls)
    matched_steps = 0
    parameter_details = []
    
    # Match step by step, stop scoring as soon as there's a mismatch
    min_length = min(len(predicted_calls), len(expected_calls))
    
    for i in range(min_length):
        pred_call = predicted_calls[i]
        exp_call = expected_calls[i]
        
        call_detail = {
            "step": i + 1,
            "expected_tool_name": exp_call.get("name", ""),
            "actual_tool_name": pred_call.get("name", ""),
            "name_match": pred_call.get("name") == exp_call.get("name"),
            "input_match": pred_call.get("input") == exp_call.get("input"),
            "is_correct": False
        }
        
        # Both tool name and parameters must match to be considered correct
        if (pred_call.get("name") == exp_call.get("name") and 
            pred_call.get("input") == exp_call.get("input")):
            matched_steps += 1
            call_detail["is_correct"] = True
            parameter_details.append(call_detail)
        else:
            # Stop scoring as soon as there's a mismatch
            call_detail["is_correct"] = False
            parameter_details.append(call_detail)
            break
    
    # If predicted tool calls are fewer than expected, record missing steps
    if len(predicted_calls) < len(expected_calls):
        for i in range(len(predicted_calls), len(expected_calls)):
            exp_call = expected_calls[i]
            parameter_details.append({
                "step": i + 1,
                "expected_tool_name": exp_call.get("name", ""),
                "actual_tool_name": None,
                "name_match": False,
                "input_match": False,
                "is_correct": False,
                "reason": "Missing step"
            })
    
    # Soft scoring: matched steps / total expected steps
    score = matched_steps / total_expected_steps
    
    return {
        "score": score,
        "key": "parameter_accuracy",
        "details": {
            "matched_steps": matched_steps,
            "total_expected_steps": total_expected_steps,
            "expected_count": len(expected_calls),
            "actual_count": len(predicted_calls),
            "call_details": parameter_details,
            "scoring_method": "soft_scoring_step_by_step"
        }
    }

def evaluate_single_question(predicted_data: Dict, ground_truth_data: Dict) -> Dict:
    """
    Perform complete evaluation for a single question
    """
    results = {}

    # Execute evaluation metrics (removed exact_match)
    results["contains_all_tool_calls_any_order"] = contains_all_tool_calls_any_order(predicted_data, ground_truth_data)
    results["contains_all_tool_calls_in_order"] = contains_all_tool_calls_in_order(predicted_data, ground_truth_data)
    results["trajectory_step_wise_score"] = trajectory_step_wise_score(predicted_data, ground_truth_data)
    results["parameter_accuracy"] = check_parameter_accuracy(predicted_data, ground_truth_data)

    return results

def run_step_by_step_evaluation(predicted_file: str, ground_truth_file: str) -> Dict:
    """
    Run complete step-by-step evaluation
    """
    # Load data
    predicted_data = load_json_data(predicted_file)
    ground_truth_data = load_json_data(ground_truth_file)
    
    # Create index mapping (based on question_index)
    # Handle different index formats: ground truth uses "question1", prediction uses "1"
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    pred_dict = {}
    for item in predicted_data:
        key = item["question_index"]
        # If the prediction data key is a pure number, convert to "questionX" format
        if key.isdigit():
            key = f"question{key}"
        pred_dict[key] = item
    
    all_results = {}
    summary_stats = {
        "total_questions": 0,
        "evaluated_questions": 0,
        "missing_predictions": [],
        "metrics_summary": {
            "contains_all_tool_calls_any_order": {"total_score": 0, "count": 0},
            "contains_all_tool_calls_in_order": {"total_score": 0, "count": 0},
            "trajectory_step_wise_score": {"total_score": 0, "count": 0},
            "parameter_accuracy": {"total_score": 0, "count": 0}
        }
    }
    
    # Traverse the first 100 questions
    for question_index, gt_item in list(gt_dict.items())[188:]:
        summary_stats["total_questions"] += 1
        
        if question_index not in pred_dict:
            summary_stats["missing_predictions"].append(question_index)
            continue
        
        pred_item = pred_dict[question_index]
        summary_stats["evaluated_questions"] += 1
        
        # Evaluate single question
        question_results = evaluate_single_question(pred_item, gt_item)
        all_results[question_index] = question_results
        
        # Update summary statistics
        for metric_name, metric_result in question_results.items():
            if metric_name in summary_stats["metrics_summary"]:
                summary_stats["metrics_summary"][metric_name]["total_score"] += metric_result["score"]
                summary_stats["metrics_summary"][metric_name]["count"] += 1
    
    # Calculate average scores
    for metric_name, metric_stats in summary_stats["metrics_summary"].items():
        if metric_stats["count"] > 0:
            metric_stats["average_score"] = metric_stats["total_score"] / metric_stats["count"]
        else:
            metric_stats["average_score"] = 0
    
    return {
        "individual_results": all_results,
        "summary": summary_stats
    }

def print_evaluation_summary(results: Dict, model_name: str = ""):
    """
    Print evaluation results summary
    """
    summary = results["summary"]
    print("=" * 60)
    if model_name:
        print(f"Step-by-Step Evaluation Results Summary - {model_name}")
    else:
        print("Step-by-Step Evaluation Results Summary")
    print("=" * 60)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Evaluated questions: {summary['evaluated_questions']}")
    print(f"Missing predictions: {len(summary['missing_predictions'])}")

    if summary['missing_predictions']:
        print(f"Missing questions: {summary['missing_predictions']}")

    print("\nAverage scores for each metric:")
    print("-" * 60)
    for metric_name, metric_stats in summary["metrics_summary"].items():
        avg_score = metric_stats.get("average_score", 0)
        count = metric_stats.get("count", 0)
        print(f"{metric_name}: {avg_score:.4f} (based on {count} questions)")

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
            # Check if required file exists
            tool_calls_file = os.path.join(item_path, "extracted_tool_calls.json")

            if os.path.exists(tool_calls_file):
                model_dirs.append(item_path)
                print(f"Found model directory: {item}")
            else:
                print(f"Skipping {item}: missing extracted_tool_calls.json")

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

        predicted_file = os.path.join(model_dir, "extracted_tool_calls.json")

        try:
            # Run evaluation for this model
            results = run_step_by_step_evaluation(predicted_file, ground_truth_file)

            all_results[model_name] = results

            # Print summary for this model
            print_evaluation_summary(results, model_name)

            # Save individual results
            output_file = os.path.join(model_dir, "step_by_step_evaluation_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to: {output_file}")

            # Collect summary for final comparison
            metrics_summary = results["summary"]["metrics_summary"]
            summary_results.append({
                "model_name": model_name,
                "contains_all_tool_calls_any_order": metrics_summary["contains_all_tool_calls_any_order"]["average_score"],
                "contains_all_tool_calls_in_order": metrics_summary["contains_all_tool_calls_in_order"]["average_score"],
                "trajectory_step_wise_score": metrics_summary["trajectory_step_wise_score"]["average_score"],
                "parameter_accuracy": metrics_summary["parameter_accuracy"]["average_score"],
                "total_questions": results["summary"]["total_questions"]
            })

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

    # Print final comparison
    print("\n" + "=" * 100)
    print("FINAL COMPARISON - ALL MODELS")
    print("=" * 100)
    print(f"{'Model Name':<25} {'Tool_Any_Order':<15} {'Tool_In_Order':<15} {'Tool_Exact_Match':<18} {'Parameter':<12}")
    print("-" * 100)

    for summary in sorted(summary_results, key=lambda x: x["parameter_accuracy"], reverse=True):
        print(f"{summary['model_name']:<25} "
              f"{summary['contains_all_tool_calls_any_order']:.4f}          "
              f"{summary['contains_all_tool_calls_in_order']:.4f}          "
              f"{summary['trajectory_step_wise_score']:.4f}             "
              f"{summary['parameter_accuracy']:.4f}")

    print("=" * 100)

    return all_results

def main():
    """
    Main function - run batch evaluation
    """
    # Configuration
    root_dir = "./evaluate_langchain"
    ground_truth_file = "./extracted_tool_calls_GT.json"

    print("Starting batch step-by-step evaluation")
    print(f"Root directory: {root_dir}")
    print(f"Ground truth file: {ground_truth_file}")

    try:
        # Run batch evaluation
        all_results = run_batch_evaluation(root_dir, ground_truth_file)

        if all_results:
            # Save combined results
            combined_output_file = "./evaluate/batch_step_by_step_results.json"
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
    # File paths
    predicted_file = r"./evaluate_langchain/qwen3max_IF_25-09-06_02-14/extracted_tool_calls.json"
    ground_truth_file = r"./extracted_tool_calls_GT.json"

    print("Starting step-by-step evaluation")

    try:
        # Run evaluation
        results = run_step_by_step_evaluation(predicted_file, ground_truth_file)

        # Print summary
        print_evaluation_summary(results)

        # Save detailed results
        output_file = r"./evaluate/step_by_step_evaluation_results.json"
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