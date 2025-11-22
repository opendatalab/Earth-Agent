"""
Rollout Manager for Training-Free GRPO
Executes agent rollouts and collects trajectories
"""
import asyncio
import json
from typing import List, Dict, Optional
from pathlib import Path
from tqdm.asyncio import tqdm
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .data_manager import DataManager, EarthAgentSample


class RolloutManager:
    """
    Manages agent rollouts for Training-Free GRPO
    Executes LangChain agent multiple times per question with higher temperature
    """

    def __init__(self, config, data_manager: DataManager):
        """
        Args:
            config: TrainingFreeGRPOConfig object
            data_manager: DataManager instance
        """
        self.config = config
        self.data_manager = data_manager
        self.agent = None
        self.client = None
        self.sys_prompt = '''You are a geoscientist, and you need to use tools to answer multiple-choice questions about Earth observation data analysis. Note that if a tool returns an error, you can only try again once. Ultimately, you only need to explicitly tell me the correct choice.
ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. For each question, you must provide the choice you think is most appropriate. Your final answer format must be:
<Answer>Your choice</Answer>'''

    async def initialize_agent(self, temp_dir: Path):
        """Initialize LangChain agent with specified configuration"""
        # Load LangChain configuration
        config_path = Path(self.config.langchain_config_path)
        with open(config_path, 'r') as f:
            langchain_config = json.load(f)

        # Setup model with rollout temperature
        model_config = langchain_config['models'][0]
        llm_kwargs = {
            'model': model_config['model_name'],
            'api_key': model_config['api_key'],
            'base_url': model_config['client_args']['base_url'],
            'temperature': self.config.practice.rollout_temperature,  # Use practice temperature
            'request_timeout': self.config.practice.task_timeout
        }

        if 'generate_args' in model_config:
            llm_kwargs['extra_body'] = model_config['generate_args']

        llm = ChatOpenAI(**llm_kwargs)

        # Prepare MCP servers
        mcp_servers = {}
        for server_name, server_config in langchain_config['mcpServers'].items():
            updated_args = []
            for arg in server_config['args']:
                if 'tmp/tmp/out' in arg:
                    updated_args.append(str(temp_dir / 'out'))
                elif arg.startswith('tools/'):
                    updated_args.append('agent/' + arg)
                else:
                    updated_args.append(arg)

            mcp_servers[server_name] = {
                "command": server_config['command'],
                "args": updated_args,
                "transport": "stdio"
            }

        # Create MCP client and agent
        self.client = MultiServerMCPClient(mcp_servers)
        tools = await self.client.get_tools()
        print(f"Loaded {len(tools)} tools for rollout")

        self.agent = create_react_agent(llm, tools)

    async def rollout_one(self, sample: EarthAgentSample) -> EarthAgentSample:
        """
        Execute one rollout for a single sample

        Args:
            sample: EarthAgentSample to process

        Returns:
            Updated sample with trajectory and response
        """
        try:
            # Prepare query
            query = sample.question + sample.data_path

            if sample.choices:
                query += '\n' + '\n'.join([
                    f'{chr(ord("A") + i)}.{choice}'
                    for i, choice in enumerate(sample.choices)
                ])

            full_query = f"{self.sys_prompt}\n\nQuestion: {query}"

            # Execute agent with timeout
            response = await asyncio.wait_for(
                self.agent.ainvoke(
                    {"messages": [HumanMessage(content=full_query)]},
                    config={
                        "recursion_limit": 50,
                        "max_execution_time": self.config.practice.task_timeout
                    }
                ),
                timeout=self.config.practice.task_timeout
            )

            # Extract trajectory
            trajectory = self._extract_trajectory(response)
            final_answer = self._extract_answer(response)

            # Update sample
            sample.update(
                stage="rollout",
                trajectory=trajectory,
                response=final_answer
            )

            return sample

        except asyncio.TimeoutError:
            print(f"Timeout for question {sample.question_id} rollout {sample.rollout_idx}")
            sample.update(
                stage="rollout",
                trajectory=[],
                response="TIMEOUT_ERROR",
                metadata={**sample.metadata, 'error': 'timeout'}
            )
            return sample

        except Exception as e:
            print(f"Error in rollout for question {sample.question_id}: {e}")
            sample.update(
                stage="rollout",
                trajectory=[],
                response=f"ERROR: {str(e)}",
                metadata={**sample.metadata, 'error': str(e)}
            )
            return sample

    def _extract_trajectory(self, response: Dict) -> List[Dict]:
        """Extract structured trajectory from agent response"""
        trajectory = []

        for message in response.get("messages", []):
            if not hasattr(message, 'type'):
                continue

            if message.type == 'human':
                trajectory.append({
                    "type": "user",
                    "content": message.content
                })

            elif message.type == 'ai':
                # Collect both thinking and tool calls
                step_data = {"type": "assistant", "content": []}

                # Add thinking text if present
                if message.content and message.content.strip():
                    step_data["content"].append({
                        "type": "text",
                        "text": message.content
                    })

                # Add tool calls
                if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                    for tool_call in message.additional_kwargs['tool_calls']:
                        try:
                            arguments = json.loads(tool_call['function']['arguments']) \
                                if isinstance(tool_call['function']['arguments'], str) \
                                else tool_call['function']['arguments']
                        except:
                            arguments = tool_call['function']['arguments']

                        step_data["content"].append({
                            "type": "tool_use",
                            "id": tool_call['id'],
                            "name": tool_call['function']['name'],
                            "input": arguments
                        })

                if step_data["content"]:
                    trajectory.append(step_data)

            elif message.type == 'tool':
                trajectory.append({
                    "type": "tool_result",
                    "tool_use_id": getattr(message, 'tool_call_id', 'unknown'),
                    "name": message.name,
                    "output": str(message.content)
                })

        return trajectory

    def _extract_answer(self, response: Dict) -> str:
        """Extract final answer from agent response"""
        messages = response.get("messages", [])

        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content
                if '<Answer>' in content and '</Answer>' in content:
                    start = content.find('<Answer>') + len('<Answer>')
                    end = content.find('</Answer>')
                    return content[start:end].strip()
                return content

        return "No answer found"

    async def run_batch(
        self,
        batch_idx: int,
        temp_dir: Path,
        use_cache: bool = True
    ) -> List[EarthAgentSample]:
        """
        Run rollouts for a specific batch

        Args:
            batch_idx: Batch index
            temp_dir: Temporary directory for agent outputs
            use_cache: Whether to use cached results

        Returns:
            List of processed samples
        """
        # Initialize agent if not already done
        if self.agent is None:
            await self.initialize_agent(temp_dir)

        # Get samples for this batch
        epoch = self.data_manager.current_epoch
        samples_to_process = self.data_manager.get_batch_samples(
            epoch=epoch,
            batch_idx=batch_idx,
            stage="init" if use_cache else None,
            batch_size=self.config.practice.batch_size
        )

        if not samples_to_process:
            print(f"No samples to process in batch {batch_idx}")
            return []

        print(f"Running {len(samples_to_process)} rollouts in batch {batch_idx}")

        # Run rollouts with concurrency control
        semaphore = asyncio.Semaphore(self.config.practice.rollout_concurrency)

        async def rollout_with_semaphore(sample: EarthAgentSample):
            async with semaphore:
                return await self.rollout_one(sample)

        # Execute rollouts in parallel
        tasks = [rollout_with_semaphore(sample) for sample in samples_to_process]
        results = []

        # Use tqdm for progress tracking
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Batch {batch_idx} rollouts"):
            result = await coro
            results.append(result)
            # Update in data manager
            self.data_manager.update_sample(result)

        print(f"Completed {len(results)} rollouts for batch {batch_idx}")
        return results

    async def cleanup(self):
        """Cleanup resources"""
        if self.client and hasattr(self.client, 'close'):
            await self.client.close()

    def compute_batch_statistics(self, samples: List[EarthAgentSample]) -> Dict:
        """Compute statistics for a batch of samples"""
        total = len(samples)
        if total == 0:
            return {}

        errors = sum(1 for s in samples if s.response and 'ERROR' in s.response)
        timeouts = sum(1 for s in samples if s.response == 'TIMEOUT_ERROR')
        successful = total - errors - timeouts

        return {
            "total_rollouts": total,
            "successful": successful,
            "errors": errors,
            "timeouts": timeouts,
            "success_rate": successful / total if total > 0 else 0
        }
