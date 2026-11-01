# Evaluator regression tests

Run from the repository root:

```bash
python -B -m unittest discover -s tests -v
```

The tests read `extracted_tool_calls_GT.json`, `benchmark/question.json`, and
archived `evaluate_langchain/*/extracted_tool_calls.json` files directly.
The small `fixtures/reference_cases.json` manifest records source identifiers
and revisions. No LLM keys, running tool servers, imagery downloads, or
geospatial packages are required.

## Reproduce the baseline comparison

Run the original and normalized references against every published saved run:

```bash
python -B tests/compare_evaluators.py --output /tmp/earth-agent-comparison.json
```

The baseline commit defaults to the revision in the fixture manifest and can
be overridden with `--baseline`. Git must contain that revision. The command
checks that the underlying metric functions are unchanged, then compares both
representations on the same eligible reference pool. It saves model-level
coverage, before/after averages, and per-question differences to the requested
JSON file. Archived model outputs are never rewritten.

## General reference-batch normalization

Some released references group several calls into parallel arrays even though
the tool accepts one scalar value per parameter. The evaluator reads exported
MCP function signatures from `agent/tools/*.py` using Python's AST; it never
imports or executes those files. The same normalization rule applies to every
tool with known scalar parameter types, with no tool-name or question-number
exceptions.

For example, a list supplied to a parameter declared `str` is expanded into
one call per string. Parallel scalar-parameter arrays must have equal, nonzero
lengths. Fixed scalar parameters are shared by those calls. Required parameters
must be present, and scalar values must match their declared primitive types.
Existing scoring formulas then operate on the normalized reference.

A parameter declared `list[str]` already accepts an entire list in one call.
It is preserved, as are scalar/list unions, nested list inputs, and unknown or
untyped parameters. The normalizer does not infer types from parameter names or
guess date pairings. Recorded agent calls keep their actual count and order,
including retries.

The tool-definition directory must exist and contain exported tools. An empty
inventory or duplicate tool names stop evaluation. Tool-loading and implementation failures propagate
to the caller; only explicit invalid-reference errors enter the exclusions.

Real regression cases cover TVDI, single-channel and multi-channel temperature
retrieval, band ratios, split-window retrieval, MODIS day/night temperature, and
ATI. List-valued statistics, annual averages, regression, and bounding-box
inputs are preserved. A small unit fixture checks discovery of a newly declared
tool without executing its source.

| Real saved run | Metric | Original | With normalization |
| --- | --- | ---: | ---: |
| Question 5, GPT-5 IF | Model/reference call-count ratio | 4.3333 | 1.0000 |
| Question 5, GPT-5 IF | Matching tool-sequence prefix | 0.6667 | 1.0000 |
| Question 5, GPT-5 IF | Parameter score | 0.3333 | 0.9231 |
| Question 7, GPT-5 IF | Model/reference call-count ratio | 6.0000 | 1.0588 |
| Question 17, GPT-5 IF | Parameter score | 0.1429 | 0.3750 |
| Question 32, GPT-5 IF | Model/reference call-count ratio | 31.0000 | 1.1625 |
| Question 42, GPT-5 IF | Model/reference call-count ratio | 12.6667 | 1.1515 |
| Question 69, DeepSeek IF | Parameter score | 0.3333 | 0.0417 |

Incomplete or different tool sequences can receive lower scores after
normalization. Extra calls and parameter differences remain penalized under
the original metrics. A call-count ratio of one alone does not demonstrate
successful completion. Scalar TVDI (question 2) and RGB (question 190) are
unchanged controls.

## Invalid references and coverage

Questions 6, 12, 18, 34, 44, 57, and 88 contain unequal arrays for scalar tool
parameters. They are reported in `invalid_references` and excluded from tool-use
averages. The single-model workflows have 247 selected references and 240
eligible references; model-specific missing predictions are reported separately.
The default RGB selection still contains 59 questions.

Direct single-question scoring raises a reference error for invalid batches.
Complete workflows continue with the other references. Efficiency details use
`status: "invalid_reference"` with `null` for the reference count and ratio.
An average with no scored questions is `null`, printed as `N/A`. Tests verify
that eligibility is independent of model predictions. Final-answer accuracy
remains independent of tool-use reference errors.

## Question 1 annotation correction

Question 1's released reference contained an extra
`Xinjiang_2022-10-16_LST.tif` entry. The
[published file inventory](https://huggingface.co/api/datasets/Sssunset/Earth-Bench/tree/49ce6b784c038b63b0817cbfa8211eb053b47e37/question1?limit=1000)
contains 91 LST and 91 NDVI files, with neither modality present for that date.
The same 182-file inventory is recorded in 16 archived runs. Removing the stale
entry from `benchmark/question.json` and `extracted_tool_calls_GT.json` aligns
all 91 input pairs with the existing output dates and annual-average date
coverage. Question 1 now has 98 reference calls and is eligible for scoring.
Its saved GPT-5 IF run has 102 calls, producing a ratio of 102/98 (1.0408).
A historical malformed-reference test restores the stale input to verify
rejection of that defect.

These checks replay saved records and verify evaluator behaviour. They do not
rerun an agent or independently verify scientific results from satellite data.
Existing output-path differences and other benchmark annotation issues remain
subject to the original scoring rules.
