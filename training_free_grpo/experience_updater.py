"""
Experience Updater for Training-Free GRPO
Extracts and distills high-quality experiences from agent rollouts
"""
import asyncio
import json
import re
from collections import defaultdict
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm
from langchain_openai import ChatOpenAI


class ExperienceUpdater:
    """
    Extracts experiential knowledge from agent rollouts using:
    1. Single rollout summary
    2. Group relative advantage (comparing good vs bad rollouts)
    3. Experience pool updating
    """

    def __init__(self, config, judge_llm: ChatOpenAI):
        """
        Args:
            config: TrainingFreeGRPOConfig
            judge_llm: ChatOpenAI instance for experience extraction
        """
        self.config = config
        self.llm = judge_llm

        # Load prompts
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates"""
        return {
            "SINGLE_ROLLOUT_SUMMARY": """Your goal is to analyze an agent's execution trajectory for Earth observation analysis and extract insights.

**Agent Objective:**
{agent_objective}

**Learning Objective:**
{learning_objective}

**Task:** Analyze the following trajectory and extract:
1. What tools were used and why
2. What information was extracted from tool results
3. What reasoning led to the final answer
4. Any missed opportunities or errors

**Input:**
- Question: {question}
- Ground Truth Answer: {answer}
- Agent's Trajectory:
{trajectory}

**Output Format:**
Provide a structured summary focusing on how the agent could improve its Earth observation analysis strategy.

Summary:
[Your analysis here]""",

            "GROUP_ADVANTAGE": """You are analyzing multiple attempts by an agent to answer an Earth observation question.

**Agent Objective:**
{agent_objective}

**Learning Objective:**
{learning_objective}

**Task:** Compare the attempts and extract up to {num_experiences} key experiences that would help future performance.

**Input:**
- Question: {question}
- Ground Truth: {answer}
- Multiple Attempts:
{trajectories}

**Instructions:**
1. Identify which responses performed better (higher rewards)
2. Compare successful vs unsuccessful strategies
3. Extract generalizable insights

**Output Format:**
<Experiences>
1. [First key experience/guideline]
2. [Second key experience/guideline]
...
</Experiences>

Provide your analysis:""",

            "EXPERIENCE_UPDATE": """You are managing a knowledge base of Earth observation analysis experiences.

**Current Experiences:**
{existing_experiences}

**New Experience to Consider:**
{new_experience}

**Task:** Decide what operation to perform:
- ADD: If this is entirely new information
- UPDATE: If this refines/improves an existing experience (specify which ID)
- DELETE: If this contradicts an existing experience (specify which ID)
- NONE: If this is redundant

**Output Format (JSON):**
{{
    "operation": "ADD|UPDATE|DELETE|NONE",
    "id": "experience_id or null",
    "content": "Updated experience text",
    "reasoning": "Why you made this decision"
}}

Your decision:""",

            "BATCH_MERGE": """You are consolidating multiple experience updates into a final knowledge base.

**Current Experiences:**
{existing_experiences}

**Proposed Updates:**
{proposed_updates}

**Task:**
1. Resolve any conflicts between proposed updates
2. Merge similar updates
3. Produce a final list of experiences

**Output Format (JSON array):**
[
    {{
        "operation": "ADD|UPDATE|DELETE",
        "id": "experience_id or null",
        "content": "Experience text"
    }},
    ...
]

Your consolidated updates:"""
        }

    async def run(
        self,
        rollouts: List,  # List of EarthAgentSample
        current_experiences: Dict[str, str],
        concurrency: int = 16
    ) -> Dict[str, str]:
        """
        Main pipeline for experience extraction

        Args:
            rollouts: List of EarthAgentSample with trajectories and rewards
            current_experiences: Current experience pool {id: experience_text}
            concurrency: Number of concurrent LLM calls

        Returns:
            Updated experience pool
        """
        print("\n=== Stage 1: Single Rollout Summary ===")
        summarized_rollouts = await self._summarize_rollouts(rollouts, concurrency)

        print("\n=== Stage 2: Group Advantage Analysis ===")
        new_experiences = await self._group_advantage(summarized_rollouts, concurrency)

        print("\n=== Stage 3: Experience Pool Update ===")
        updated_experiences = await self._update_experience_pool(
            current_experiences, new_experiences, concurrency
        )

        return updated_experiences

    async def _summarize_rollouts(
        self, rollouts: List, concurrency: int
    ) -> Dict[str, List[Dict]]:
        """Summarize each rollout's trajectory"""
        # Group by question
        question_to_rollouts = defaultdict(list)
        for rollout in rollouts:
            if rollout.trajectory and len(rollout.trajectory) > 0:
                question_to_rollouts[rollout.question_id].append(rollout)

        # Only process groups with partial success (some correct, some wrong)
        filtered_rollouts = []
        for qid, group_rollouts in question_to_rollouts.items():
            rewards = [r.reward for r in group_rollouts]
            avg_reward = sum(rewards) / len(rewards)

            # Learn from cases with at least some success
            # Changed from: 0 < avg_reward < 1.0 (only partial success)
            # To: avg_reward > 0 (any success)
            if avg_reward > 0:
                # Only include successful rollouts for learning
                filtered_rollouts.extend([r for r in group_rollouts if r.reward > 0])

        if not filtered_rollouts:
            print("No successful rollouts to summarize")
            return {}

        print(f"Summarizing {len(filtered_rollouts)} rollouts...")

        semaphore = asyncio.Semaphore(concurrency)

        async def summarize_one(rollout):
            async with semaphore:
                try:
                    prompt = self.prompts["SINGLE_ROLLOUT_SUMMARY"].format(
                        agent_objective=self.config.practice.agent_objective,
                        learning_objective=self.config.practice.learning_objective,
                        question=rollout.question,
                        answer=rollout.correct_answer or "[Not available]",
                        trajectory=json.dumps(rollout.trajectory, indent=2)
                    )

                    response = await self.llm.ainvoke(prompt)
                    return {
                        **rollout.to_dict(),
                        "summary": response.content
                    }
                except Exception as e:
                    print(f"Error summarizing rollout {rollout.question_id}: {e}")
                    return None

        tasks = [summarize_one(r) for r in filtered_rollouts]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing"):
            result = await coro
            if result:
                results.append(result)

        # Re-group by question
        question_to_summaries = defaultdict(list)
        for result in results:
            question_to_summaries[result['question_id']].append(result)

        return question_to_summaries

    async def _group_advantage(
        self, question_to_summaries: Dict[str, List[Dict]], concurrency: int
    ) -> List[str]:
        """Extract experiences by comparing good vs bad rollouts per question"""
        if not question_to_summaries:
            return []

        semaphore = asyncio.Semaphore(concurrency)
        num_exp_per_query = self.config.practice.num_experiences_per_query

        async def extract_experiences(qid, summaries):
            async with semaphore:
                try:
                    # Format trajectories with rewards
                    trajectory_text = "\n\n".join([
                        f"Attempt {i+1} (Reward: {s['reward']:.2f}):\n{s['summary']}"
                        for i, s in enumerate(summaries)
                    ])

                    prompt = self.prompts["GROUP_ADVANTAGE"].format(
                        agent_objective=self.config.practice.agent_objective,
                        learning_objective=self.config.practice.learning_objective,
                        num_experiences=num_exp_per_query,
                        question=summaries[0]['question'],
                        answer=summaries[0]['correct_answer'] or "[Not available]",
                        trajectories=trajectory_text
                    )

                    response = await self.llm.ainvoke(prompt)
                    response_text = response.content

                    # Extract experiences from <Experiences> tags
                    match = re.search(r'<Experiences>\s*(.*?)\s*</Experiences>', response_text, re.DOTALL)
                    if match:
                        experiences_text = match.group(1).strip()
                        # Parse numbered list
                        experiences = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|$)', experiences_text, re.DOTALL)
                        return [exp.strip() for exp in experiences]
                    else:
                        print(f"Could not extract experiences for question {qid}")
                        return []

                except Exception as e:
                    print(f"Error in group advantage for {qid}: {e}")
                    return []

        tasks = [extract_experiences(qid, summaries) for qid, summaries in question_to_summaries.items()]
        all_experiences = []

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Group advantage"):
            experiences = await coro
            all_experiences.extend(experiences)

        print(f"Extracted {len(all_experiences)} new experiences")
        return all_experiences

    async def _update_experience_pool(
        self, current_experiences: Dict[str, str], new_experiences: List[str], concurrency: int
    ) -> Dict[str, str]:
        """Update experience pool with new experiences"""
        if not new_experiences:
            return current_experiences

        # Stage 1: Decide operation for each new experience
        semaphore = asyncio.Semaphore(concurrency)
        operations = []

        async def decide_operation(exp):
            async with semaphore:
                try:
                    existing_text = "\n".join([
                        f"[{eid}] {text}" for eid, text in current_experiences.items()
                    ]) if current_experiences else "None"

                    prompt = self.prompts["EXPERIENCE_UPDATE"].format(
                        existing_experiences=existing_text,
                        new_experience=exp
                    )

                    response = await self.llm.ainvoke(prompt)
                    response_text = response.content

                    # Parse JSON
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        operation = json.loads(json_match.group(0))
                        return operation
                    else:
                        # Default to ADD if can't parse
                        return {
                            "operation": "ADD",
                            "id": None,
                            "content": exp,
                            "reasoning": "Failed to parse, defaulting to ADD"
                        }
                except Exception as e:
                    print(f"Error deciding operation: {e}")
                    return {
                        "operation": "ADD",
                        "id": None,
                        "content": exp,
                        "reasoning": f"Error: {e}"
                    }

        tasks = [decide_operation(exp) for exp in new_experiences]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Deciding operations"):
            op = await coro
            operations.append(op)

        # Stage 2: Batch merge all operations
        print("\nMerging all operations...")
        updated_experiences = await self._batch_merge_operations(current_experiences, operations)

        print(f"Experience pool updated: {len(current_experiences)} -> {len(updated_experiences)}")
        return updated_experiences

    async def _batch_merge_operations(
        self, current_experiences: Dict[str, str], operations: List[Dict]
    ) -> Dict[str, str]:
        """Merge all operations into final experience pool"""
        try:
            existing_text = "\n".join([
                f"[{eid}] {text}" for eid, text in current_experiences.items()
            ]) if current_experiences else "None"

            operations_text = json.dumps(operations, indent=2, ensure_ascii=False)

            prompt = self.prompts["BATCH_MERGE"].format(
                existing_experiences=existing_text,
                proposed_updates=operations_text
            )

            response = await self.llm.ainvoke(prompt)
            response_text = response.content

            # Parse JSON array
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                final_operations = json.loads(json_match.group(0))
            else:
                print("Could not parse batch merge result, using original operations")
                final_operations = operations

            # Apply operations
            updated = dict(current_experiences)
            max_id = max([int(k) for k in updated.keys()], default=-1) + 1

            for op in final_operations:
                op_type = op.get("operation", "ADD")
                content = op.get("content", "")
                eid = op.get("id")

                if op_type == "ADD" and content:
                    updated[str(max_id)] = content
                    max_id += 1
                elif op_type == "UPDATE" and eid in updated and content:
                    updated[eid] = content
                elif op_type == "DELETE" and eid in updated:
                    del updated[eid]

            return updated

        except Exception as e:
            print(f"Error in batch merge: {e}, keeping original experiences")
            return current_experiences
