"""
Main Training-Free GRPO Orchestrator for Earth-Agent
"""
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from langchain_openai import ChatOpenAI

from .config import TrainingFreeGRPOConfig
from .data_manager import DataManager
from .rollout_manager import RolloutManager
from .experience_updater import ExperienceUpdater
from .verify.earth_science_enhanced import verify_earth_science_answer


class TrainingFreeGRPO:
    """
    Main orchestrator for Training-Free GRPO on Earth-Agent

    Pipeline:
    1. Load dataset and configuration
    2. For each epoch and batch:
        a. Run N rollouts per question (with higher temperature)
        b. Verify each rollout with rewards
        c. Extract experiences using group advantage
        d. Update experience pool
    3. Generate enhanced agent config with experiences
    """

    def __init__(self, config: TrainingFreeGRPOConfig):
        """
        Args:
            config: TrainingFreeGRPOConfig object
        """
        self.config = config
        self.data_manager = DataManager(config)
        self.rollout_manager = RolloutManager(config, self.data_manager)

        # Initialize judge LLM
        self.judge_llm = self._create_judge_llm()
        self.experience_updater = ExperienceUpdater(config, self.judge_llm)

        # Experience pool
        self.experiences: Dict[str, str] = {}

        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.log_dir = Path(config.log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Temp directory for agent outputs
        self.temp_dir = self.output_dir / f"temp_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _create_judge_llm(self) -> ChatOpenAI:
        """Create judge LLM for verification and experience extraction"""
        judge_config = self.config.judge_model or self.config.model

        return ChatOpenAI(
            model=judge_config.model_name,
            api_key=judge_config.api_key,
            base_url=judge_config.base_url,
            temperature=judge_config.temperature,
            max_tokens=judge_config.max_tokens,
            timeout=judge_config.timeout
        )

    async def run(self) -> Path:
        """
        Run the complete Training-Free GRPO pipeline

        Returns:
            Path to the enhanced agent configuration file
        """
        print("="*80)
        print(f"Starting Training-Free GRPO for Earth-Agent")
        print(f"Experiment ID: {self.config.exp_id}")
        print(f"Epochs: {self.config.practice.epochs}")
        print(f"Batch size: {self.config.practice.batch_size}")
        print(f"GRPO-N (rollouts per question): {self.config.practice.grpo_n}")
        print("="*80)

        try:
            # Main training loop
            for epoch in range(self.config.practice.epochs):
                print(f"\n{'='*80}")
                print(f"EPOCH {epoch + 1}/{self.config.practice.epochs}")
                print(f"{'='*80}")

                await self._run_epoch(epoch)

                # Save checkpoint
                self._save_checkpoint(epoch)

            # Generate final enhanced agent config
            enhanced_config_path = self._generate_enhanced_config()

            print(f"\n{'='*80}")
            print(f"Training-Free GRPO completed!")
            print(f"Enhanced agent config saved to: {enhanced_config_path}")
            print(f"Total experiences learned: {len(self.experiences)}")
            print(f"{'='*80}")

            return enhanced_config_path

        except Exception as e:
            print(f"Error in Training-Free GRPO: {e}")
            raise

        finally:
            # Cleanup
            await self.rollout_manager.cleanup()

    async def _run_epoch(self, epoch: int):
        """Run one epoch of training"""
        # Load data for this epoch
        self.data_manager.load_epoch_data(
            epoch,
            shuffle=self.config.practice.shuffle_data,
            truncate=self.config.practice.rollout_data_truncate
        )

        # Calculate number of batches
        total_samples = len(self.data_manager.current_epoch_data)
        batch_size = self.config.practice.batch_size * self.config.practice.grpo_n
        num_batches = total_samples // batch_size

        print(f"Processing {num_batches} batches ({total_samples} samples total)")

        # Process each batch
        for batch_idx in range(num_batches):
            print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
            await self._run_batch(epoch, batch_idx)

    async def _run_batch(self, epoch: int, batch_idx: int):
        """Run one batch of training"""
        step = epoch * 1000 + batch_idx  # Unique step identifier

        # Check if we should use cache
        use_cache = self._should_use_cache(step)

        # Stage 1: Run rollouts
        print("\n[1/3] Running rollouts...")
        rollouts = await self.rollout_manager.run_batch(
            batch_idx=batch_idx,
            temp_dir=self.temp_dir,
            use_cache=use_cache
        )

        if not rollouts:
            print("No rollouts completed, skipping batch")
            return

        # Compute statistics
        stats = self.rollout_manager.compute_batch_statistics(rollouts)
        print(f"Rollout stats: {stats}")

        # Stage 2: Verify rollouts
        print("\n[2/3] Verifying rollouts...")
        verified_rollouts = await self._verify_rollouts(rollouts)

        # Stage 3: Extract and update experiences
        print("\n[3/3] Extracting experiences...")
        new_experiences = await self.experience_updater.run(
            rollouts=verified_rollouts,
            current_experiences=self.experiences,
            concurrency=self.config.practice.rollout_concurrency
        )

        # Update experience pool
        self.experiences = new_experiences

        # Log batch results
        self._log_batch_results(epoch, batch_idx, stats, verified_rollouts)

    async def _verify_rollouts(self, rollouts: list):
        """Verify all rollouts and assign rewards"""
        print(f"Verifying {len(rollouts)} rollouts...")

        async def verify_one(sample):
            try:
                result = await verify_earth_science_answer(
                    sample=sample,
                    judge_llm=self.judge_llm,
                    timeout_score=0.0
                )

                sample.update(
                    stage="judged",
                    reward=result["reward"],
                    reasoning=result["reasoning"]
                )

                return sample

            except Exception as e:
                print(f"Error verifying {sample.question_id}: {e}")
                sample.update(
                    stage="judged",
                    reward=0.0,
                    reasoning=f"Verification error: {e}"
                )
                return sample

        # Verify in parallel
        semaphore = asyncio.Semaphore(self.config.practice.rollout_concurrency)

        async def verify_with_semaphore(sample):
            async with semaphore:
                return await verify_one(sample)

        tasks = [verify_with_semaphore(r) for r in rollouts]
        verified = await asyncio.gather(*tasks)

        # Compute reward statistics
        rewards = [r.reward for r in verified]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"Average reward: {avg_reward:.3f} (range: {min(rewards):.2f} - {max(rewards):.2f})")

        return verified

    def _should_use_cache(self, step: int) -> bool:
        """Determine if cache should be used for this step"""
        restart_step = self.config.practice.restart_step
        return restart_step is None or step < restart_step

    def _generate_enhanced_config(self) -> Path:
        """Generate enhanced agent configuration with experiences"""
        print("\nGenerating enhanced agent configuration...")

        # Format experiences for prompt
        if self.experiences:
            experience_text = "\n\nWhen solving Earth observation questions, you MUST carefully read and apply these learned experiences:\n\n"
            experience_text += "\n".join([
                f"[Experience {eid}] {exp}"
                for eid, exp in self.experiences.items()
            ])
        else:
            experience_text = ""

        # Create enhanced system prompt
        enhanced_sys_prompt = f'''{self.rollout_manager.sys_prompt}

{experience_text}'''

        # Save enhanced configuration
        enhanced_config = {
            "exp_id": f"{self.config.exp_id}_enhanced",
            "system_prompt": enhanced_sys_prompt,
            "experiences": self.experiences,
            "base_config": str(self.config.langchain_config_path),
            "metadata": {
                "num_experiences": len(self.experiences),
                "generated_at": datetime.now().isoformat(),
                "practice_config": self.config.to_dict()
            }
        }

        config_path = self.output_dir / f"{self.config.exp_id}_enhanced_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_config, f, indent=2, ensure_ascii=False)

        # Also save just the experiences
        exp_path = self.output_dir / f"{self.config.exp_id}_experiences.json"
        with open(exp_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiences, f, indent=2, ensure_ascii=False)

        print(f"Enhanced config saved to: {config_path}")
        print(f"Experiences saved to: {exp_path}")

        return config_path

    def _save_checkpoint(self, epoch: int):
        """Save checkpoint after each epoch"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "experiences": self.experiences,
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        print(f"Checkpoint saved: {checkpoint_path}")

    def _log_batch_results(self, epoch: int, batch_idx: int, stats: dict, rollouts: list):
        """Log batch results to file"""
        log_path = self.log_dir / f"batch_log_epoch{epoch}.jsonl"

        batch_log = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "stats": stats,
            "num_experiences": len(self.experiences),
            "sample_results": [
                {
                    "question_id": r.question_id,
                    "rollout_idx": r.rollout_idx,
                    "reward": r.reward,
                    "response": r.response[:200] if r.response else None  # Truncate for logging
                }
                for r in rollouts[:5]  # Log first 5 samples
            ],
            "timestamp": datetime.now().isoformat()
        }

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(batch_log, ensure_ascii=False) + '\n')

        # Save full rollout details
        rollouts_dir = self.output_dir / "rollouts"
        rollouts_dir.mkdir(exist_ok=True)

        rollouts_path = rollouts_dir / f"rollouts_epoch{epoch}_batch{batch_idx}.json"
        rollouts_data = []
        for r in rollouts:
            rollouts_data.append({
                "question_id": r.question_id,
                "rollout_idx": r.rollout_idx,
                "question": r.question,
                "response": r.response,
                "reward": r.reward,
                "correct_answer": r.correct_answer,
                "choices": r.choices,
                "trajectory": r.trajectory if hasattr(r, 'trajectory') else None,
                "timestamp": datetime.now().isoformat()
            })

        with open(rollouts_path, 'w', encoding='utf-8') as f:
            json.dump(rollouts_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(rollouts)} rollouts to: {rollouts_path}")

