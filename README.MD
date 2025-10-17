# Earth Agent Evaluation Framework

This repository contains the evaluation framework for Earth Agent: Unlocking the Full Landscape of Earth Observation with Agents

## 📦 Data Preparation

### 1. Download Dataset from Hugging Face

Download the benchmark dataset from Hugging Face:

<!-- ```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download the dataset
huggingface-cli download wqjklej/safkhj --local-dir ./benchmark/data --repo-type dataset
``` -->

<!-- Alternatively, you can download manually:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wqjklej/safkhj",
    repo_type="dataset",
    local_dir="./benchmark/data"
)
``` -->

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download the dataset
huggingface-cli download Sssunset/Earth-Bench --local-dir ./benchmark/data --repo-type dataset
``` -->

<!-- Alternatively, you can download manually:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Sssunset/Earth-Bench",
    repo_type="dataset",
    local_dir="./benchmark/data"
)
```

### 2. Dataset Structure

After downloading, your data directory should have the following structure:

```
Earth_Agent/benchmark/
            └── data/
                ├── question1/
                │   ├── image1
                |   |── image2
                |   |── ...
                |   |    
                ├── question2/
                │   ├── image1
                |   |── image2
                |   |── ...
                |   |  
                ├── ...
                └── question248/
                    ├── image1
                    ├── image2
                    └── ...
```

## 🔧 Configuration

### 1. API Keys Setup

Before running evaluations, configure your model API keys in the configuration files:

```bash
# Edit the configuration files in agent/ directory
# Set your API keys for the models you want to evaluate
cp agent/config.json.example agent/config.json
# Edit agent/config.json and add your API keys
```

### 2. Model Configuration Files

The framework supports multiple models. Configuration files are located in `agent/` directory:
- `config_gpt5.json` - GPT-5 configuration
- `config_deepseek.json` - DeepSeek configuration
- `config_kimik2.json` - Kimik2 configuration
- `config_gemini2_5.json` - Gemini 2.5 configuration
- And more...

## 🚀 Running Evaluations

### 1. Single Model Evaluation

Run evaluation for a single model:

```bash
# Example: Evaluate GPT-5 model
python main.py --config agent/config_gpt5.json --mode evaluation

# Example: Evaluate DeepSeek model
python main.py --config agent/config_deepseek.json --mode evaluation
```

### 2. Batch Model Evaluation

Run evaluation for multiple models:

```bash
# Run all configured models
python batch_evaluate.py --config_dir agent/ --output_dir ./evaluate_langchain
```

## 📊 Evaluation Metrics

The framework provides comprehensive evaluation across multiple dimensions:

### 1. Tool-Use Evaluation (Step-by-Step Analysis)

Run step-by-step evaluation:

```bash
python evaluate/step_by_step.py
```

**Metrics calculated:**
- **Tool-Any-Order**: Measures if all required tools are used (order-independent)
- **Tool-In-Order**: Measures if tools are used in the correct sequence
- **Tool-Exact-Match**: Strict step-by-step matching of tool usage
- **Parameter**: Accuracy of tool parameters and arguments

### 2. End-to-End Evaluation

Run end-to-end evaluation:

```bash
python evaluate/end_to_end.py
```

**Metrics calculated:**
- **Efficiency**: Tool usage efficiency (model tools / ground truth tools)
- **Accuracy**: Final answer accuracy percentage

### 3. Evaluation Results

Results will be saved in the following locations:

```
evaluate_langchain/
├── [model_name]/
│   ├── results_summary_polished.json    # Final answers
│   ├── extracted_tool_calls.json        # Tool usage data
│   ├── step_by_step_evaluation_results.json
│   └── end_to_end_evaluation_results.json
└── ...
```

### 4. Batch Evaluation Results

Combined results for all models:

```
evaluate/
├── batch_step_by_step_results.json      # Tool-use metrics for all models
└── batch_evaluation_results.json        # End-to-end metrics for all models
```

## 📈 Understanding the Results

### Tool-Use Metrics (0.0 - 1.0 scale)
- **Higher is better** for all tool-use metrics
- **Tool-Any-Order**: 1.0 means all required tools were used
- **Tool-In-Order**: 1.0 means perfect sequential tool usage
- **Tool-Exact-Match**: 1.0 means perfect step-by-step execution
- **Parameter**: 1.0 means perfect parameter accuracy

### End-to-End Metrics
- **Efficiency**: Lower values indicate more efficient tool usage
  - 1.0 = Perfect efficiency (same number of tools as ground truth)
  - \>1.0 = Used more tools than necessary
  - <1.0 = Used fewer tools than ground truth
- **Accuracy**: Percentage of correctly answered questions (0-100%)

### Sample Output

```
====================================================================================================
Model Name                Tool_Any_Order  Tool_In_Order   Tool_Exact_Match   Parameter
----------------------------------------------------------------------------------------------------
deepseek-V3_1_IF          0.8921          0.8764          0.7405             0.5722
gpt5_AP                   0.7661          0.7504          0.5960             0.4615
kimik2_IF                 0.8062          0.7990          0.6332             0.5219
...
====================================================================================================

======================================================================
Model Name                     Efficiency   Accuracy
----------------------------------------------------------------------
gpt5_AP                        1.5312      59.32%
kimik2_IF                      1.4104      62.71%
deepseek-V3_1_AP               1.6895      55.93%
...
======================================================================
```

## 🔍 Advanced Usage

### Custom Evaluation Range

Modify the evaluation range by editing the slice in evaluation files:

```python
# In evaluate/step_by_step.py and evaluate/end_to_end.py
# Evaluate RGB Modality
for question_index, gt_item in list(gt_dict.items())[188:]:

# Evaluate Spectrum Modality
for question_index, gt_item in list(gt_dict.items())[0:100]:

# Evaluate Products Modality
for question_index, gt_item in list(gt_dict.items())[100:188]:
```

### Ground Truth Data

The ground truth file `extracted_tool_calls_GT.json` contains reference tool usage patterns and correct answers for comparison.

## 📝 File Descriptions

- `main.py` - Main evaluation script for single models
- `evaluate/step_by_step.py` - Tool-use evaluation metrics
- `evaluate/end_to_end.py` - End-to-end evaluation metrics
- `evaluate/merge.py` - Tool call merging utilities
- `agent/` - Model configuration files
- `benchmark/` - Benchmark dataset and questions
- `tools/` - Tool implementations for the agent system

## 📚 Citation
```bibtex
@article{feng2025earth,
  title={Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents},
  author={Feng, Peilin and Lv, Zhutao and Ye, Junyan and Wang, Xiaolei and Huo, Xinjie and Yu, Jinhua and Xu, Wanghan and Zhang, Wenlong and Bai, Lei and He, Conghui and others},
  journal={arXiv preprint arXiv:2509.23141},
  year={2025}
}
```