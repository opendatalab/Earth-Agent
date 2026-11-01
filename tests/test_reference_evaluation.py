"""Regression cases replayed from the published Earth-Agent reference and runs."""

import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluate"))

STEP_MODULES = [
    importlib.import_module("step_by_step"),
    importlib.import_module("step_by_step_single"),
]
EFFICIENCY_MODULES = [
    importlib.import_module("end_to_end"),
    importlib.import_module("end_to_end_single"),
]
MANIFEST = json.loads(
    (Path(__file__).parent / "fixtures" / "reference_cases.json").read_text()
)
REFERENCES = {
    record["question_index"]: record
    for record in json.loads((ROOT / "extracted_tool_calls_GT.json").read_text())
}
RUNS = {
    path: {
        "question" + str(record["question_index"]).removeprefix("question"): record
        for record in json.loads((ROOT / path).read_text())
    }
    for path in {case["prediction_source"] for case in MANIFEST["cases"].values()}
}
FIXTURE = {"cases": {
    name: {
        "reference": REFERENCES[case["question_index"]],
        "prediction": RUNS[case["prediction_source"]][case["question_index"]],
    }
    for name, case in MANIFEST["cases"].items()
}}


def historical_question1_reference():
    """Recreate the released reference defect for invalid-input regression tests."""
    reference = copy.deepcopy(REFERENCES["question1"])
    stale_path = "benchmark/data/question1/Xinjiang_2022-10-16_LST.tif"
    paths = reference["tool_calls"][1]["input"]["lst_path"]
    if stale_path not in paths:
        paths.append(stale_path)
        paths.sort()
    return reference


def efficiency(module, reference, prediction):
    # The batch script selects RGB by default. Put the real case at the selected
    # position without changing its evaluator or the real case's contents.
    references = [reference]
    if module.__name__ == "end_to_end":
        references = [
            {"question_index": f"padding{i}", "tool_calls": []}
            for i in range(188)
        ] + references
    result = module.calculate_efficiency_with_tool_calls(
        references, {reference["question_index"]: prediction}
    )
    return result["detailed_results"][-1]["efficiency"]


class TVDIReferenceEvaluationTests(unittest.TestCase):
    def test_question5_real_run_matches_all_thirteen_tool_steps(self):
        case = FIXTURE["cases"]["question5_gpt5_IF"]
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                result = module.evaluate_single_question(case["prediction"], case["reference"])
                self.assertEqual(result["trajectory_step_wise_score"]["score"], 1.0)
                self.assertEqual(
                    result["trajectory_step_wise_score"]["details"]["total_expected"], 13
                )

    def test_question5_real_run_has_thirteen_actual_and_reference_calls(self):
        case = FIXTURE["cases"]["question5_gpt5_IF"]
        self.assertEqual(len(case["prediction"]["tool_calls"]), 13)
        for module in EFFICIENCY_MODULES:
            with self.subTest(evaluator=module.__name__):
                self.assertEqual(
                    efficiency(module, case["reference"], case["prediction"]), 1.0
                )

    def test_question5_remaining_parameter_difference_is_not_forgiven(self):
        case = FIXTURE["cases"]["question5_gpt5_IF"]
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                result = module.evaluate_single_question(case["prediction"], case["reference"])
                # The final call still differs in paths and explicit defaults.
                self.assertEqual(result["parameter_accuracy"]["details"]["matched_steps"], 12)
                self.assertAlmostEqual(result["parameter_accuracy"]["score"], 12 / 13)

    def test_question3_extra_and_different_calls_still_reduce_scores(self):
        case = FIXTURE["cases"]["question3_gpt5_IF"]
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                result = module.evaluate_single_question(case["prediction"], case["reference"])
                self.assertAlmostEqual(result["trajectory_step_wise_score"]["score"], 0.9)
                self.assertAlmostEqual(result["parameter_accuracy"]["score"], 0.8)
        for module in EFFICIENCY_MODULES:
            with self.subTest(evaluator=module.__name__):
                self.assertAlmostEqual(
                    efficiency(module, case["reference"], case["prediction"]), 1.8
                )

    def test_scalar_tvdi_control_remains_unchanged(self):
        case = FIXTURE["cases"]["question2_gpt5_IF"]
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                result = module.evaluate_single_question(case["prediction"], case["reference"])
                self.assertEqual(result["trajectory_step_wise_score"]["score"], 1.0)
                self.assertAlmostEqual(result["parameter_accuracy"]["score"], 2 / 3)
        for module in EFFICIENCY_MODULES:
            with self.subTest(evaluator=module.__name__):
                self.assertEqual(efficiency(module, case["reference"], case["prediction"]), 1.0)

    def test_rgb_control_remains_unchanged(self):
        case = FIXTURE["cases"]["question190_deepseek_AP"]
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                result = module.evaluate_single_question(case["prediction"], case["reference"])
                self.assertTrue(all(metric["score"] == 1.0 for metric in result.values()))
        for module in EFFICIENCY_MODULES:
            with self.subTest(evaluator=module.__name__):
                self.assertEqual(efficiency(module, case["reference"], case["prediction"]), 1.0)

    def test_question1_unequal_real_lists_require_review(self):
        case = {
            "reference": historical_question1_reference(),
            "prediction": FIXTURE["cases"]["question1_deepseek_IF"]["prediction"],
        }
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                with self.assertRaisesRegex(ValueError, "question1: compute_tvdi step 2") as error:
                    module.evaluate_single_question(case["prediction"], case["reference"])
                self.assertIn("ndvi_path=91", str(error.exception))
                self.assertIn("lst_path=92", str(error.exception))
        for module in EFFICIENCY_MODULES:
            with self.subTest(evaluator=module.__name__):
                self.assertIsNone(efficiency(module, case["reference"], case["prediction"]))


    def test_incomplete_real_run_gets_lower_parameter_credit(self):
        case = FIXTURE["cases"]["question69_deepseek_IF"]
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                result = module.evaluate_single_question(case["prediction"], case["reference"])
                self.assertAlmostEqual(result["parameter_accuracy"]["score"], 1 / 24)
                self.assertAlmostEqual(result["trajectory_step_wise_score"]["score"], 2 / 24)
                self.assertAlmostEqual(result["contains_all_tool_calls_in_order"]["score"], 4 / 24)

    def test_legitimate_list_tools_and_separate_years_are_not_merged(self):
        from reference_trajectory import normalize_reference_trajectory
        case = FIXTURE["cases"]["question1_deepseek_IF"]
        reference = {
            "question_index": "question1",
            "tool_calls": [
                call for call in case["reference"]["tool_calls"]
                if call["name"] == "calculate_tif_average"
            ],
        }
        self.assertEqual(len(reference["tool_calls"]), 4)
        self.assertEqual(normalize_reference_trajectory(reference), reference)

    def test_question1_reference_matches_published_and_observed_file_inventory(self):
        from reference_trajectory import normalize_reference_trajectory

        reference = REFERENCES["question1"]
        published = MANIFEST["question1_dataset"]
        # Actual get_filelist output, rather than model-generated paths.
        saved = RUNS["evaluate_langchain/gpt5_IF/extracted_tool_calls.json"]["question1"]
        observation = next(
            call["output"] for call in saved["tool_calls"]
            if call["name"] == "get_filelist"
        )
        observed_files = set(json.loads(observation))
        self.assertEqual(len(observed_files), published["file_count"])
        self.assertNotIn(published["removed_reference_filename"], observed_files)
        self.assertEqual(set(reference["tool_calls"][0]["output"]), observed_files)

        paths = reference["tool_calls"][1]["input"]
        def dates(values):
            return [Path(path).stem.split("_")[-1] for path in values]

        def input_dates(values):
            # NDVI/LST names end in the modality; compare their dated stem.
            return [
                Path(path).stem.removeprefix("Xinjiang_").rsplit("_", 1)[0]
                for path in values
            ]

        self.assertEqual(len(paths["ndvi_path"]), published["ndvi_count"])
        self.assertEqual(len(paths["lst_path"]), published["lst_count"])
        self.assertEqual(input_dates(paths["ndvi_path"]), input_dates(paths["lst_path"]))
        self.assertEqual(input_dates(paths["ndvi_path"]), dates(paths["output_path"]))
        input_files = {
            Path(path).name for key in ("ndvi_path", "lst_path") for path in paths[key]
        }
        self.assertEqual(input_files, observed_files)
        annual_inputs = {
            path for call in reference["tool_calls"]
            if call["name"] == "calculate_tif_average"
            for path in call["input"]["file_list"]
        }
        # Verify the existing annual steps cover exactly the paired dates.
        # Output-directory differences remain subject to normal parameter scoring.
        self.assertEqual(set(dates(annual_inputs)), set(dates(paths["output_path"])))
        self.assertEqual(len(normalize_reference_trajectory(reference)["tool_calls"]), 98)

        source = json.loads((ROOT / "benchmark/question.json").read_text())["1"]
        source_calls = [
            call["function"] for dialog in source["dialogs"]
            for call in dialog.get("tool_calls", [])
        ]
        source_tvdi = next(call for call in source_calls if call["name"] == "compute_tvdi")
        self.assertEqual(source_tvdi["arguments"], paths)
        source_files = next(
            dialog["content"]["content"] for dialog in source["dialogs"]
            if dialog.get("role") == "tool" and dialog.get("name") == "get_filelist"
        )
        self.assertEqual(set(source_files), observed_files)

    def test_invalid_batch_variants_are_rejected_without_guessing(self):
        from reference_trajectory import normalize_reference_trajectory
        case = FIXTURE["cases"]["question5_gpt5_IF"]
        variants = []
        missing = copy.deepcopy(case["reference"])
        del missing["tool_calls"][1]["input"]["lst_path"]
        variants.append(missing)
        empty = copy.deepcopy(case["reference"])
        for key in ("ndvi_path", "lst_path", "output_path"):
            empty["tool_calls"][1]["input"][key] = []
        variants.append(empty)
        non_string = copy.deepcopy(case["reference"])
        non_string["tool_calls"][1]["input"]["ndvi_path"][0] = None
        variants.append(non_string)
        for reference in variants:
            with self.subTest(reference=reference["tool_calls"][1]["input"]):
                with self.assertRaises(ValueError):
                    normalize_reference_trajectory(reference)

    def test_evaluation_preserves_original_reference_and_saved_run(self):
        case = copy.deepcopy(FIXTURE["cases"]["question5_gpt5_IF"])
        original = copy.deepcopy(case)
        for module in STEP_MODULES:
            module.evaluate_single_question(case["prediction"], case["reference"])
        for module in EFFICIENCY_MODULES:
            efficiency(module, case["reference"], case["prediction"])
        self.assertEqual(case, original)

    def test_reference_self_replay_scores_one_after_expansion(self):
        case = FIXTURE["cases"]["question5_gpt5_IF"]
        # The first twelve recorded calls are exactly the expanded reference
        # prefix; the thirteenth difference is covered separately above.
        prefix = {
            "question_index": "question5",
            "tool_calls": case["reference"]["tool_calls"][:2],
        }
        prediction = {"tool_calls": case["prediction"]["tool_calls"][:12]}
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                result = module.evaluate_single_question(prediction, prefix)
                self.assertTrue(all(metric["score"] == 1.0 for metric in result.values()))


class GeneralReferenceEvaluationTests(unittest.TestCase):
    def test_real_batches_in_six_other_tools_use_the_same_rule(self):
        expected = {
            # Reference calls after expansion, actual recorded agent calls.
            "question7": (17, 18),
            "question13": (171, 2),
            "question17": (8, 10),
            "question32": (80, 93),
            "question36": (33, 0),
            "question42": (33, 38),
        }
        from reference_trajectory import normalize_reference_trajectory

        for question, (expected_count, actual_count) in expected.items():
            case = FIXTURE["cases"][question + "_gpt5_IF"]
            original = copy.deepcopy(case)
            with self.subTest(question=question):
                normalized = normalize_reference_trajectory(case["reference"])
                self.assertEqual(len(normalized["tool_calls"]), expected_count)
                self.assertEqual(len(case["prediction"]["tool_calls"]), actual_count)
                self.assertEqual(normalize_reference_trajectory(normalized), normalized)
                for module in STEP_MODULES:
                    result = module.evaluate_single_question(
                        case["prediction"], case["reference"]
                    )
                    self.assertEqual(
                        result["trajectory_step_wise_score"]["details"]["total_expected"],
                        expected_count,
                    )
                for module in EFFICIENCY_MODULES:
                    self.assertAlmostEqual(
                        efficiency(module, case["reference"], case["prediction"]),
                        actual_count / expected_count,
                    )
                self.assertEqual(case, original)

    def test_real_scalar_settings_are_shared_and_list_inputs_remain_single_calls(self):
        from reference_trajectory import normalize_reference_trajectory

        reference = REFERENCES["question32"]
        normalized = normalize_reference_trajectory(reference)
        calculations = [
            call for call in normalized["tool_calls"] if call["name"] == "split_window"
        ]
        self.assertEqual(len(calculations), 78)
        self.assertTrue(all(call["input"]["parameter"] == "LST" for call in calculations))
        # This tool accepts a string OR a list of image paths as one argument.
        self.assertEqual(normalized["tool_calls"][-1], reference["tool_calls"][-1])
        # Batch statistics and regression consume real lists in single calls.
        reference = REFERENCES["question13"]
        normalized = normalize_reference_trajectory(reference)
        self.assertEqual(normalized["tool_calls"][-13:], reference["tool_calls"][-13:])
        # Nested list conditions and untyped bounding-box arrays stay intact.
        for question in ("question21", "question213", "question226", "question238"):
            with self.subTest(question=question):
                self.assertEqual(
                    normalize_reference_trajectory(REFERENCES[question]),
                    REFERENCES[question],
                )

    def test_new_tool_is_discovered_from_its_contract_without_name_exceptions(self):
        import reference_trajectory

        # Unit-only source fixture. A top-level failure proves it is not executed.
        source = (
            'raise RuntimeError("do not import tool servers")\n'
            '@mcp.tool()\n'
            'def previously_unseen(image: str, threshold: float, weights: list[float], '
            'method: str = "mean", label: str | None = None):\n'
            '    raise RuntimeError("do not execute tools")\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new_tool.py"
            path.write_text(source)
            contracts = reference_trajectory._read_tool_contracts(Path(directory))
        reference = {
            "question_index": "unit-contract",
            "tool_calls": [{
                "name": "previously_unseen",
                "input": {
                    "image": ["a.tif", "b.tif"],
                    "threshold": [1, 2.5],
                    "weights": [0.2, 0.8],
                    "method": "median",
                    "label": None,
                },
            }],
        }
        original = copy.deepcopy(reference)
        with patch.object(reference_trajectory, "_tool_contracts", return_value=contracts):
            normalized = reference_trajectory.normalize_reference_trajectory(reference)
        calls = normalized["tool_calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual([call["input"]["image"] for call in calls], ["a.tif", "b.tif"])
        self.assertEqual([call["input"]["threshold"] for call in calls], [1, 2.5])
        for call in calls:
            self.assertEqual(call["input"]["weights"], [0.2, 0.8])
            self.assertEqual(call["input"]["method"], "median")
            self.assertIsNone(call["input"]["label"])
        self.assertEqual(reference, original)

    def test_missing_empty_and_conflicting_tool_definitions_stop_evaluation(self):
        import reference_trajectory

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                reference_trajectory._read_tool_contracts(directory / "missing")
            with self.assertRaisesRegex(ValueError, "No exported MCP tool"):
                reference_trajectory._read_tool_contracts(directory)
            (directory / "a.py").write_text(
                "@mcp.tool()\ndef duplicate(path: str): pass\n"
            )
            (directory / "b.py").write_text(
                "@mcp.tool()\ndef duplicate(path: list[str]): pass\n"
            )
            with self.assertRaisesRegex(ValueError, "Ambiguous MCP tool name"):
                reference_trajectory._read_tool_contracts(directory)

    def test_nullable_scalars_expand_but_scalar_list_unions_do_not(self):
        import reference_trajectory

        source = (
            "import typing\n"
            "@mcp.tool()\n"
            "def example(path: typing.Optional[str], "
            "values: typing.Union[str, list[str]]): pass\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "tool.py").write_text(source)
            contracts = reference_trajectory._read_tool_contracts(directory)
        reference = {"tool_calls": [{
            "name": "example",
            "input": {"path": ["a.tif", None], "values": ["band1", "band2"]},
        }]}
        with patch.object(reference_trajectory, "_tool_contracts", return_value=contracts):
            result = reference_trajectory.normalize_reference_trajectory(reference)
        self.assertEqual(
            [call["input"]["path"] for call in result["tool_calls"]], ["a.tif", None]
        )
        self.assertTrue(all(
            call["input"]["values"] == ["band1", "band2"]
            for call in result["tool_calls"]
        ))

    def test_unknown_and_untyped_inputs_are_not_guessed_to_be_batches(self):
        from reference_trajectory import normalize_reference_trajectory

        reference = {
            "tool_calls": [{
                "name": "unknown_tool", "input": {"data": [1, 2, 3]},
            }],
        }
        self.assertEqual(normalize_reference_trajectory(reference), reference)

    def test_unequal_real_batches_are_reported_for_all_affected_tools(self):
        from reference_trajectory import normalize_reference_trajectory

        expected = {
            "question6": "lst_single_channel",
            "question12": "lst_multi_channel",
            "question18": "band_ratio",
            "question34": "split_window",
            "question44": "ATI",
            "question57": "lst_multi_channel",
            "question88": "ATI",
        }
        self.assertEqual(list(expected), MANIFEST["invalid_reference_questions"])
        for question, tool in expected.items():
            original = copy.deepcopy(REFERENCES[question])
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, question + ": " + tool):
                    normalize_reference_trajectory(REFERENCES[question])
                self.assertEqual(REFERENCES[question], original)


class EvaluationWorkflowTests(unittest.TestCase):
    def test_complete_saved_run_workflows_continue_with_reported_coverage(self):
        reference_path = str(ROOT / "extracted_tool_calls_GT.json")
        run = ROOT / "evaluate_langchain" / "gpt5_IF"
        for module in STEP_MODULES:
            with self.subTest(evaluator=module.__name__):
                result = module.run_step_by_step_evaluation(
                    str(run / "extracted_tool_calls.json"), reference_path
                )
                summary = result["summary"]
                expected = 59 if module.__name__ == "step_by_step" else 240
                self.assertEqual(summary["eligible_questions"], expected)
                self.assertEqual(summary["evaluated_questions"], expected)
                for metric in summary["metrics_summary"].values():
                    self.assertEqual(metric["count"], expected)
                if module.__name__.endswith("_single"):
                    self.assertEqual(summary["total_questions"], 247)
                    self.assertEqual(
                        [item["question_index"] for item in summary["invalid_references"]],
                        MANIFEST["invalid_reference_questions"],
                    )
                    self.assertIn("question1", result["individual_results"])
                    self.assertIn("question247", result["individual_results"])
                else:
                    self.assertEqual(summary["invalid_references"], [])
                with contextlib.redirect_stdout(io.StringIO()):
                    module.print_evaluation_summary(result)
                    if hasattr(module, "print_detailed_results"):
                        module.print_detailed_results(result)
        for module in EFFICIENCY_MODULES:
            with self.subTest(evaluator=module.__name__):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = module.run_end_to_end_evaluation(
                        reference_path, str(run / "results_summary_polished.json"),
                        str(run / "extracted_tool_calls.json"),
                    )
                    module.print_evaluation_summary(result)
                    if hasattr(module, "print_detailed_results"):
                        module.print_detailed_results(result)
                expected = 59 if module.__name__ == "end_to_end" else 240
                efficiency_result = result["efficiency"]
                self.assertEqual(efficiency_result["eligible_questions"], expected)
                self.assertEqual(efficiency_result["evaluated_questions"], expected)
                self.assertEqual(len(efficiency_result["efficiency_scores"]), expected)
                if module.__name__.endswith("_single"):
                    row = efficiency_result["detailed_results"][0]
                    self.assertEqual(row["question_index"], "question1")
                    self.assertEqual(row["status"], "evaluated")
                    self.assertEqual(row["gt_tool_count"], 98)
                    self.assertEqual(row["model_tool_count"], 102)
                    self.assertAlmostEqual(row["efficiency"], 102 / 98)
                    self.assertEqual(
                        [item["question_index"] for item in efficiency_result["invalid_references"]],
                        MANIFEST["invalid_reference_questions"],
                    )
                    self.assertEqual(result["accuracy"]["evaluated_questions"], 247)
                    self.assertEqual(
                        result["accuracy"]["detailed_results"][0]["question_index"],
                        "question1",
                    )

    def selected_references(self, module, references):
        if module.__name__ in ("step_by_step", "end_to_end"):
            # Preserve the scripts' RGB offset while exercising real references.
            # Prefix entries are outside the selected range and never scored.
            excluded = {reference["question_index"] for reference in references}
            prefix = [
                reference for question, reference in REFERENCES.items()
                if question not in excluded
            ][:188]
            return prefix + references
        return references

    def test_invalid_reference_pool_is_independent_of_model_predictions(self):
        invalid = historical_question1_reference()
        valid = REFERENCES["question5"]
        missing = REFERENCES["question3"]
        case = FIXTURE["cases"]["question5_gpt5_IF"]
        for module in STEP_MODULES + EFFICIENCY_MODULES:
            references = self.selected_references(module, [invalid, valid, missing])
            for include_invalid_prediction in (False, True):
                with self.subTest(evaluator=module.__name__,
                                  invalid_prediction=include_invalid_prediction):
                    predictions = {"question5": case["prediction"]}
                    if include_invalid_prediction:
                        predictions["question1"] = FIXTURE["cases"]["question1_deepseek_IF"]["prediction"]
                    if module in STEP_MODULES:
                        with patch.object(module, "load_json_data",
                                          side_effect=[list(predictions.values()), references]):
                            result = module.run_step_by_step_evaluation("saved-run", "references")
                        summary = result["summary"]
                        self.assertEqual(summary["missing_predictions"], ["question3"])
                        for metric in summary["metrics_summary"].values():
                            self.assertEqual(metric["count"], 1)
                    else:
                        summary = module.calculate_efficiency_with_tool_calls(
                            references, predictions
                        )
                        self.assertEqual(summary["efficiency_scores"], [1.0])
                        self.assertEqual(summary["average_efficiency"], 1.0)
                    self.assertEqual(summary["total_questions"], 3)
                    self.assertEqual(summary["eligible_questions"], 2)
                    self.assertEqual(summary["evaluated_questions"], 1)
                    self.assertEqual(
                        [item["question_index"] for item in summary["invalid_references"]],
                        ["question1"],
                    )
                    self.assertIn("lst_path=92", summary["invalid_references"][0]["reason"])

    def test_contract_failures_propagate_instead_of_becoming_invalid_references(self):
        case = FIXTURE["cases"]["question5_gpt5_IF"]
        for module in STEP_MODULES + EFFICIENCY_MODULES:
            references = self.selected_references(module, [case["reference"]])
            with self.subTest(evaluator=module.__name__):
                with patch.object(
                    module, "normalize_reference_trajectory",
                    side_effect=ValueError("tool-contract loading failed"),
                ):
                    with self.assertRaisesRegex(ValueError, "tool-contract loading failed"):
                        if module in STEP_MODULES:
                            with patch.object(
                                module, "load_json_data",
                                side_effect=[[case["prediction"]], references],
                            ):
                                module.run_step_by_step_evaluation("run", "references")
                        else:
                            module.calculate_efficiency_with_tool_calls(
                                references, {"question5": case["prediction"]}
                            )

    def test_batch_reports_and_saved_json_handle_unscorable_and_missing_runs(self):
        references = self.selected_references(
            STEP_MODULES[0], [REFERENCES["question6"], REFERENCES["question5"]]
        )
        prediction = FIXTURE["cases"]["question5_gpt5_IF"]["prediction"]
        answers = json.loads(
            (ROOT / "evaluate_langchain/gpt5_IF/results_summary_polished.json").read_text()
        )
        answer = next(record for record in answers if record["question_id"] == "5")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference_path = directory / "references.json"
            reference_path.write_text(json.dumps(references))
            for name, calls, model_answers in (
                ("saved_run", [prediction], [answer]),
                ("missing_run", [], []),
            ):
                model = directory / name
                model.mkdir()
                (model / "extracted_tool_calls.json").write_text(json.dumps(calls))
                (model / "results_summary_polished.json").write_text(json.dumps(model_answers))
            for module in (STEP_MODULES[0], EFFICIENCY_MODULES[0]):
                with self.subTest(evaluator=module.__name__):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        results = module.run_batch_evaluation(
                            str(directory), str(reference_path)
                        )
                    self.assertEqual(set(results), {"saved_run", "missing_run"})
                    self.assertIn("N/A", output.getvalue())
                    self.assertNotIn("Error evaluating", output.getvalue())
                    for name in results:
                        if module in STEP_MODULES:
                            coverage = results[name]["summary"]
                            average = coverage["metrics_summary"]["parameter_accuracy"]["average_score"]
                            output_name = "step_by_step_evaluation_results.json"
                        else:
                            coverage = results[name]["efficiency"]
                            average = coverage["average_efficiency"]
                            output_name = "end_to_end_evaluation_results.json"
                        self.assertEqual(coverage["eligible_questions"], 1)
                        self.assertEqual(
                            coverage["evaluated_questions"],
                            1 if name == "saved_run" else 0,
                        )
                        self.assertEqual(
                            [item["question_index"] for item in coverage["invalid_references"]],
                            ["question6"],
                        )
                        if name == "missing_run":
                            self.assertIsNone(average)
                        else:
                            self.assertIsNotNone(average)
                        saved = json.loads((directory / name / output_name).read_text())
                        self.assertEqual(saved, results[name])

    def test_all_invalid_references_produce_no_numeric_average(self):
        for module in STEP_MODULES:
            references = self.selected_references(module, [historical_question1_reference()])
            with self.subTest(evaluator=module.__name__):
                with patch.object(module, "load_json_data", side_effect=[[], references]):
                    result = module.run_step_by_step_evaluation("saved-run", "references")
                self.assertEqual(result["summary"]["eligible_questions"], 0)
                self.assertEqual(result["summary"]["evaluated_questions"], 0)
                self.assertFalse(result["individual_results"])
                for metric in result["summary"]["metrics_summary"].values():
                    self.assertEqual(metric["count"], 0)
                    self.assertIsNone(metric["average_score"])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    module.print_evaluation_summary(result)
                self.assertIn("N/A", output.getvalue())
                self.assertIn("question1", output.getvalue())
        for module in EFFICIENCY_MODULES:
            references = self.selected_references(module, [historical_question1_reference()])
            with self.subTest(evaluator=module.__name__):
                with patch.object(module, "load_json_data", side_effect=[references, []]):
                    with patch.object(module, "load_model_tool_calls", return_value={}):
                        with contextlib.redirect_stdout(io.StringIO()):
                            result = module.run_end_to_end_evaluation(
                                "references", "saved-answers", "saved-tool-calls"
                            )
                efficiency_result = result["efficiency"]
                self.assertEqual(efficiency_result["eligible_questions"], 0)
                self.assertEqual(efficiency_result["evaluated_questions"], 0)
                self.assertEqual(efficiency_result["efficiency_scores"], [])
                self.assertIsNone(efficiency_result["average_efficiency"])
                self.assertIsNone(result["summary"]["average_efficiency"])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    module.print_evaluation_summary(result)
                    if hasattr(module, "print_detailed_results"):
                        module.print_detailed_results(result)
                self.assertIn("N/A", output.getvalue())
                self.assertIn("question1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
