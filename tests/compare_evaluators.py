"""Compare reference normalization on published runs without invoking any tools."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import step_by_step_single as metrics
from evaluate.reference_trajectory import (
    InvalidReferenceError,
    normalize_reference_trajectory,
)

METRICS = {
    "contains_all_tool_calls_any_order": metrics.contains_all_tool_calls_any_order,
    "contains_all_tool_calls_in_order": metrics.contains_all_tool_calls_in_order,
    "contains_all_tool_calls_in_order_exact_match": metrics.contains_all_tool_calls_in_order_exact_match,
    "trajectory_step_wise_score": metrics.trajectory_step_wise_score,
    "parameter_accuracy": metrics.check_parameter_accuracy,
}


def git_text(revision: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True
    )


def verify_original_metrics(revision: str) -> None:
    """Use shared metric functions only if their baseline bodies are unchanged."""
    original = ast.parse(git_text(revision, "evaluate/step_by_step_single.py"))
    current = ast.parse(Path(metrics.__file__).read_text())

    def function_bodies(tree):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, ast.FunctionDef)
        }

    before, after = function_bodies(original), function_bodies(current)
    names = [function.__name__ for function in METRICS.values()]
    names.extend(("extract_tool_names_from_calls", "find_tool_calls_from_data"))
    for name in names:
        if name not in before or name not in after or before[name] != after[name]:
            raise ValueError(f"Metric {name} differs from the requested baseline")


def scores(prediction: dict, reference: dict) -> dict:
    result = {
        name: function(prediction, reference)["score"]
        for name, function in METRICS.items()
    }
    expected = len(reference.get("tool_calls", []))
    actual = len(prediction.get("tool_calls", []))
    result["call_ratio"] = (
        actual / expected if expected else (1.0 if actual == 0 else float("inf"))
    )
    return result


def averages(rows: list[dict]) -> dict:
    return {
        name: sum(row[name] for row in rows) / len(rows) if rows else None
        for name in (*METRICS, "call_ratio")
    }


def compare(baseline: str) -> dict:
    verify_original_metrics(baseline)
    original = {
        record["question_index"]: record
        for record in json.loads(git_text(baseline, "extracted_tool_calls_GT.json"))
    }
    references = json.loads((ROOT / "extracted_tool_calls_GT.json").read_text())
    normalized, invalid, changed = {}, [], []
    tool_families = set()
    for reference in references:
        question = reference["question_index"]
        try:
            view = normalize_reference_trajectory(reference)
        except InvalidReferenceError as error:
            invalid.append({"question_index": question, "reason": str(error)})
            continue
        normalized[question] = view
        if view != reference:
            changed.append(question)
            before = Counter(call["name"] for call in reference["tool_calls"])
            after = Counter(call["name"] for call in view["tool_calls"])
            tool_families.update(
                name for name in after if after[name] != before[name]
            )

    paired = controls = affected = excluded = 0
    comparisons, models = [], []
    paths = sorted((ROOT / "evaluate_langchain").glob("*/extracted_tool_calls.json"))
    if not paths:
        raise FileNotFoundError("No published saved runs found in evaluate_langchain")
    for path in paths:
        predictions = {
            "question" + str(record["question_index"]).removeprefix("question"): record
            for record in json.loads(path.read_text())
        }
        before_rows, after_rows, missing = [], [], []
        for reference in references:
            question = reference["question_index"]
            if question not in predictions:
                if question in normalized:
                    missing.append(question)
                continue
            paired += 1
            if question not in normalized:
                excluded += 1
                continue
            before = scores(predictions[question], original[question])
            after = scores(predictions[question], normalized[question])
            before_rows.append(before)
            after_rows.append(after)
            if normalized[question] == original[question]:
                if before != after:
                    raise AssertionError(f"Unchanged reference changed scores: {path.parent.name}/{question}")
                controls += 1
            else:
                affected += 1
                comparisons.append({
                    "model": path.parent.name, "question": question,
                    "actual_calls": len(predictions[question]["tool_calls"]),
                    "reference_calls_before": len(original[question]["tool_calls"]),
                    "reference_calls_after": len(normalized[question]["tool_calls"]),
                    "before": before, "after": after,
                })
        models.append({
            "model": path.parent.name,
            "eligible_questions": len(normalized),
            "evaluated_questions": len(after_rows),
            "missing_valid_predictions": missing,
            "before_on_same_eligible_pool": averages(before_rows),
            "after_on_same_eligible_pool": averages(after_rows),
        })

    return {
        "summary": {
            "baseline": baseline,
            "metric_implementations_unchanged": True,
            "models": len(models),
            "selected_references": len(references),
            "eligible_references": len(normalized),
            "normalized_reference_questions": changed,
            "batch_tool_families": sorted(tool_families),
            "real_question_run_pairs": paired,
            "unchanged_control_pairs": controls,
            "normalization_affected_pairs": affected,
            "unscorable_pairs": excluded,
            "invalid_references": invalid,
            "scope": "Saved tool-call replay; no agent execution or scientific answer verification.",
        },
        "models": models,
        "comparisons": comparisons,
    }


def main() -> None:
    manifest = json.loads((ROOT / "tests/fixtures/reference_cases.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=manifest["upstream_commit"])
    parser.add_argument("--output", type=Path, help="Save full comparison JSON at this path")
    args = parser.parse_args()
    baseline = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline}^{{commit}}"],
        cwd=ROOT, text=True,
    ).strip()
    report = compare(baseline)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
