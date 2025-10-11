import json
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

def contains_all_tool_calls_in_order_exact_match(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    """
    Check if tool calls are completely matched (both order and content are exactly the same) - soft scoring version
    """
    expected_tools = find_tool_calls_from_data(ground_truth_data)
    actual_tools = find_tool_calls_from_data(predicted_data)
    
    if not expected_tools:
        return {
            "score": 1.0 if not actual_tools else 0.0,
            "key": "contains_all_tool_calls_in_order_exact_match",
            "expected": expected_tools,
            "actual": actual_tools,
            "details": {"exact_matches": 0, "total_expected": 0}
        }
    
    # Calculate the number of exact matches by position
    min_length = min(len(expected_tools), len(actual_tools))
    exact_matches = 0
    
    for i in range(min_length):
        if expected_tools[i] == actual_tools[i]:
            exact_matches += 1
        else:
            break  # Stop as soon as there's a mismatch (requiring strict order)
    
    # If lengths are different, calculate score based on the longer length
    total_positions = max(len(expected_tools), len(actual_tools))
    score = exact_matches / total_positions if total_positions > 0 else 1.0
    
    return {
        "score": score, 
        "key": "contains_all_tool_calls_in_order_exact_match",
        "expected": expected_tools,
        "actual": actual_tools,
        "details": {
            "exact_matches": exact_matches,
            "total_expected": len(expected_tools),
            "total_actual": len(actual_tools),
            "total_positions": total_positions
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
    
    # Execute all evaluation metrics
    results["contains_all_tool_calls_any_order"] = contains_all_tool_calls_any_order(predicted_data, ground_truth_data)
    results["contains_all_tool_calls_in_order"] = contains_all_tool_calls_in_order(predicted_data, ground_truth_data)
    results["contains_all_tool_calls_in_order_exact_match"] = contains_all_tool_calls_in_order_exact_match(predicted_data, ground_truth_data)
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
            "contains_all_tool_calls_in_order_exact_match": {"total_score": 0, "count": 0},
            "trajectory_step_wise_score": {"total_score": 0, "count": 0},
            "parameter_accuracy": {"total_score": 0, "count": 0}
        }
    }
    
    # Traverse the first 100 questions
    for question_index, gt_item in list(gt_dict.items())[:]:
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

def print_evaluation_summary(results: Dict):
    """
    Print evaluation results summary
    """
    summary = results["summary"]
    print("=" * 60)
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

def print_detailed_results(results: Dict):
    """
    Print detailed results for each question
    """
    individual_results = results["individual_results"]

    print("\n" + "=" * 80)
    print("Detailed Results for Each Question")
    print("=" * 80)
    print(f"{'Question':<12} {'Tool_Any_Order':<15} {'Tool_In_Order':<15} {'Tool_Exact_Match':<17} {'Parameter':<15}")
    print("-" * 80)

    for question_index, question_results in individual_results.items():
        any_order_score = question_results["contains_all_tool_calls_any_order"]["score"]
        in_order_score = question_results["contains_all_tool_calls_in_order"]["score"]
        trajectory_score = question_results["trajectory_step_wise_score"]["score"]
        parameter_score = question_results["parameter_accuracy"]["score"]

        print(f"{question_index:<12} {any_order_score:<15.4f} {in_order_score:<15.4f} {trajectory_score:<17.4f} {parameter_score:<15.4f}")

    print("=" * 80)

def main():
    """
    Main function - run evaluation
    """
    # Specify model directory
    model_dir = "./evaluate_langchain/Kimik2_IF_25-09-21_07-08"

    # File paths
    predicted_file = f"{model_dir}/extracted_tool_calls.json"
    ground_truth_file = "./extracted_tool_calls_GT.json"
    
    print("Starting step-by-step evaluation")
    
    try:
        # Run evaluation
        results = run_step_by_step_evaluation(predicted_file, ground_truth_file)
        
        # Print summary
        print_evaluation_summary(results)

        # Print detailed results for each question
        print_detailed_results(results)
        
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