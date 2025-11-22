"""
Configuration classes for Training-Free GRPO
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json


@dataclass
class PracticeArguments:
    """Arguments for practice process"""
    # Training loop settings
    epochs: int = 1
    batch_size: int = 10
    grpo_n: int = 3  # Number of rollouts per query

    # Rollout settings
    rollout_concurrency: int = 16
    rollout_temperature: float = 0.7
    task_timeout: int = 3600

    # Experience extraction settings
    num_experiences_per_query: int = 1
    given_ground_truth: bool = True

    # Evaluation settings
    do_eval: bool = False
    eval_strategy: str = "epoch"  # "epoch" or "steps"
    eval_steps: int = 10
    eval_data_truncate: Optional[int] = None

    # Data settings
    shuffle_data: bool = True
    rollout_data_truncate: Optional[int] = None

    # Restart behavior
    restart_step: Optional[int] = None  # None: use cache, 0: restart all, N: restart from step N

    # Objectives
    agent_objective: str = ""
    learning_objective: str = ""


@dataclass
class ModelConfig:
    """LLM Model Configuration"""
    model_name: str = "gpt5"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 8192
    timeout: int = 120


@dataclass
class TrainingFreeGRPOConfig:
    """Unified configuration for Training-Free GRPO"""
    exp_id: str = "earth_agent_practice"

    # Practice settings
    practice: PracticeArguments = field(default_factory=PracticeArguments)

    # Model configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    judge_model: Optional[ModelConfig] = None

    # Dataset paths
    practice_dataset_path: str = "./benchmark/question.json"
    eval_dataset_path: Optional[str] = None

    # Question filtering
    question_ids: Optional[List[str]] = None  # If None, use all questions

    # Paths
    output_dir: str = "./training_free_results"
    log_dir: str = "./training_free_logs"

    # Verification
    verify_module: str = "earth_science"  # Module name in verify/ directory

    # LangChain agent config
    langchain_config_path: str = "./agent/config_gpt5.json"
    use_autoplanning: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingFreeGRPOConfig':
        """Load configuration from dictionary"""
        practice_dict = config_dict.pop('practice', {})
        practice = PracticeArguments(**practice_dict)

        model_dict = config_dict.pop('model', {})
        model = ModelConfig(**model_dict)

        judge_model_dict = config_dict.pop('judge_model', None)
        judge_model = ModelConfig(**judge_model_dict) if judge_model_dict else None

        return cls(
            practice=practice,
            model=model,
            judge_model=judge_model,
            **config_dict
        )

    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingFreeGRPOConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'exp_id': self.exp_id,
            'practice': self.practice.__dict__,
            'model': self.model.__dict__,
            'judge_model': self.judge_model.__dict__ if self.judge_model else None,
            'practice_dataset_path': self.practice_dataset_path,
            'eval_dataset_path': self.eval_dataset_path,
            'question_ids': self.question_ids,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir,
            'verify_module': self.verify_module,
            'langchain_config_path': self.langchain_config_path,
            'use_autoplanning': self.use_autoplanning
        }

    def save(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
