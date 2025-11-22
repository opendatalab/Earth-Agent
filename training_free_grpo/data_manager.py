"""
Data Manager for Training-Free GRPO
Handles dataset loading, preprocessing, and batch management
"""
import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class EarthAgentSample:
    """Single evaluation sample for Earth-Agent"""
    # Basic info
    question_id: str
    question: str
    data_path: str
    choices: Optional[List[str]] = None
    correct_answer: Optional[str] = None

    # Processing stages
    stage: str = "init"  # init -> rollout -> judged

    # Rollout results
    trajectory: Optional[List[Dict]] = None  # Agent execution trajectory
    response: Optional[str] = None  # Final answer

    # Verification results
    reward: float = 0.0  # 0.0 to 1.0
    reasoning: Optional[str] = None  # Verification reasoning

    # Metadata
    epoch: int = 0
    batch_idx: int = 0
    rollout_idx: int = 0  # Which rollout in the group (0 to grpo_n-1)
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EarthAgentSample':
        """Create from dictionary"""
        return cls(**data)

    def update(self, **kwargs):
        """Update fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class DataManager:
    """
    Manages Earth-Agent dataset for Training-Free GRPO
    """

    def __init__(self, config):
        """
        Args:
            config: TrainingFreeGRPOConfig object
        """
        self.config = config
        self.practice_dataset_path = Path(config.practice_dataset_path)
        self.eval_dataset_path = Path(config.eval_dataset_path) if config.eval_dataset_path else None

        # Load datasets
        self.practice_data = self._load_dataset(self.practice_dataset_path, config.question_ids)
        self.eval_data = self._load_dataset(self.eval_dataset_path, None) if self.eval_dataset_path else []

        # In-memory cache for current epoch
        self.current_epoch_data: List[EarthAgentSample] = []
        self.current_epoch: int = -1

    def _load_dataset(self, dataset_path: Path, question_ids: Optional[List[str]] = None) -> List[Dict]:
        """Load Earth-Agent benchmark dataset"""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        # Convert to list format
        dataset = []
        for qid, question_info in full_data.items():
            # Filter by question_ids if provided
            if question_ids and qid not in question_ids:
                continue

            # Determine which question format to use (autoplanning vs instructed)
            ap_index = 0 if question_info['evaluation'][0]['type'] == 'autonomous planning' else 1
            data_path = question_info['evaluation'][ap_index].get('data', None)
            data_path = question_info['evaluation'][1 - ap_index].get('data', None) if data_path is None else data_path

            if data_path is None:
                continue

            question_text = (
                question_info['evaluation'][ap_index]['question']
                if self.config.use_autoplanning
                else question_info['evaluation'][1 - ap_index]['question']
            )

            # Extract ground truth answer from evaluation field
            eval_idx = ap_index if self.config.use_autoplanning else (1 - ap_index)
            gt_answer_obj = question_info['evaluation'][eval_idx].get('gt_answer', {})
            correct_answer = gt_answer_obj.get('whitelist', None) if isinstance(gt_answer_obj, dict) else None

            dataset.append({
                'question_id': qid,
                'question': question_text,
                'data_path': data_path,
                'choices': question_info.get('choices', None),
                'correct_answer': correct_answer  # Ground truth from evaluation.gt_answer.whitelist
            })

        print(f"Loaded {len(dataset)} questions from {dataset_path}")
        return dataset

    def load_epoch_data(self, epoch: int, shuffle: bool = True, truncate: Optional[int] = None) -> List[EarthAgentSample]:
        """
        Prepare data for a specific epoch

        Args:
            epoch: Epoch number
            shuffle: Whether to shuffle the data
            truncate: Maximum number of samples to use (for debugging)

        Returns:
            List of EarthAgentSample objects for this epoch
        """
        # Use cached data if already loaded for this epoch
        if self.current_epoch == epoch and self.current_epoch_data:
            return self.current_epoch_data

        # Create samples
        samples = []
        dataset = self.practice_data[:truncate] if truncate else self.practice_data

        if shuffle:
            dataset = random.sample(dataset, len(dataset))

        # Replicate each question grpo_n times for multiple rollouts
        grpo_n = self.config.practice.grpo_n
        for data_item in dataset:
            for rollout_idx in range(grpo_n):
                sample = EarthAgentSample(
                    question_id=data_item['question_id'],
                    question=data_item['question'],
                    data_path=data_item['data_path'],
                    choices=data_item['choices'],
                    correct_answer=data_item['correct_answer'],
                    epoch=epoch,
                    rollout_idx=rollout_idx,
                    metadata={'original_data': data_item}
                )
                samples.append(sample)

        # Cache for this epoch
        self.current_epoch = epoch
        self.current_epoch_data = samples

        print(f"Prepared {len(samples)} samples for epoch {epoch} ({len(dataset)} unique questions × {grpo_n} rollouts)")
        return samples

    def get_batch_samples(
        self,
        epoch: int,
        batch_idx: Optional[int] = None,
        stage: Optional[str] = None,
        batch_size: int = 10
    ) -> List[EarthAgentSample]:
        """
        Get samples for a specific batch

        Args:
            epoch: Epoch number
            batch_idx: Batch index (None = all batches)
            stage: Filter by stage (None = all stages)
            batch_size: Size of each batch

        Returns:
            List of samples matching the criteria
        """
        # Ensure epoch data is loaded
        if self.current_epoch != epoch:
            self.load_epoch_data(epoch)

        samples = self.current_epoch_data

        # Filter by batch
        if batch_idx is not None:
            start_idx = batch_idx * batch_size * self.config.practice.grpo_n
            end_idx = start_idx + batch_size * self.config.practice.grpo_n
            samples = samples[start_idx:end_idx]

        # Filter by stage
        if stage is not None:
            samples = [s for s in samples if s.stage == stage]

        return samples

    def update_sample(self, sample: EarthAgentSample):
        """Update a sample in the current epoch data"""
        for i, s in enumerate(self.current_epoch_data):
            if (s.question_id == sample.question_id and
                s.epoch == sample.epoch and
                s.rollout_idx == sample.rollout_idx):
                self.current_epoch_data[i] = sample
                return

        # If not found, append
        self.current_epoch_data.append(sample)

    def save_checkpoint(self, checkpoint_path: Path):
        """Save current state to checkpoint"""
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            'current_epoch': self.current_epoch,
            'samples': [s.to_dict() for s in self.current_epoch_data]
        }

        with open(checkpoint_path / f"epoch_{self.current_epoch}.json", 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self, checkpoint_path: Path, epoch: int):
        """Load state from checkpoint"""
        checkpoint_file = checkpoint_path / f"epoch_{epoch}.json"

        if not checkpoint_file.exists():
            return False

        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        self.current_epoch = checkpoint_data['current_epoch']
        self.current_epoch_data = [
            EarthAgentSample.from_dict(s) for s in checkpoint_data['samples']
        ]

        print(f"Loaded checkpoint for epoch {epoch}")
        return True
