import os
os.environ["GTIFF_SRS_SOURCE"]="EPSG"
import json
import logging
import asyncio
from enum import auto
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from logging.handlers import RotatingFileHandler

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Pprint for debugging
from pprint import pprint

# Change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Global variables
logger = None
temp_dir_path = None

# Configuration
model_name = 'interns1'
autoplanning = False
sys_prompt = '''
You are a geoscientist, and you need to use tools to answer multiple-choice questions about Earth observation data analysis. Note that if a tool returns an error, you can only try again once. Ultimately, you only need to explicitly tell me the correct choice.
ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. For each question, you must provide the choice you think is most appropriate.Don't gibe me another format. Your final answer format must be:
<Answer>Your choice<Answer>
'''


def init_global_params():
    """Initialize global parameters and logging"""
    global temp_dir_path, logger
    
    if temp_dir_path is None:
        temp_dir_path = Path('./evaluate_langchain/{}_{}_{}'.format(
            model_name, 
            'AP' if autoplanning else "IF", 
            datetime.now().strftime('%y-%m-%d_%H-%M')
        )).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # Simplified logging compatible with original format
            log_record = {
                "question_index": record.args[0] if record.args else "unknown",
                "timestamp": self.formatTime(record, self.datefmt),
                "conversations": record.args[1] if len(record.args) > 1 else [],
                "final_answer": record.args[2] if len(record.args) > 2 else None
            }
            return json.dumps(log_record, ensure_ascii=False, indent=4)

    logger = logging.getLogger("text_logger")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        temp_dir_path / "{}_{}_langchain.log".format(
            model_name, 'AP' if autoplanning else "IF"
        )
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    
    return temp_dir_path, logger


def init_chat_logger():
    """Initialize chat logger for .chat file like AgentScope"""
    global temp_dir_path
    chat_log_path = temp_dir_path / "{}_{}_langchain.chat".format(
        model_name, 'AP' if autoplanning else "IF"
    )
    return chat_log_path


def save_chat_message(chat_log_path, message_data):
    """Save a single chat message to .chat file in AgentScope format"""
    import time
    from datetime import datetime
    import uuid
    
    # Format message in AgentScope style
    chat_record = {
        "__module__": "langchain.schema.messages",
        "__name__": "ChatMessage", 
        "id": str(uuid.uuid4()).replace('-', ''),
        "name": message_data.get('name', 'langchain_agent'),
        "role": message_data.get('role', 'assistant'),
        "content": message_data.get('content', []),
        "metadata": message_data.get('metadata', None),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Append to chat file (one JSON per line, like AgentScope)
    with open(chat_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chat_record, ensure_ascii=False) + '\n')


def load_langchain_config(config_path='./agent/config_interns1.json'):
    """Load configuration and initialize LangChain components"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize OpenAI model with stricter parameters
    model_config = config['models'][0]
    llm_kwargs = {
        'model': model_config['model_name'],
        'api_key': model_config['api_key'],
        'base_url': model_config['client_args']['base_url'],
        'temperature': 0.1,  # Lower temperature for more focused responses
        'request_timeout': 120  # 2 minute timeout per request
    }
    
    # Add generate_args via extra_body if present in config
    if 'generate_args' in model_config:
        llm_kwargs['extra_body'] = model_config['generate_args']
    
    llm = ChatOpenAI(**llm_kwargs)
    
    # Prepare MCP servers configuration
    mcp_servers = {}
    for server_name, server_config in config['mcpServers'].items():
        # Update paths to use current temp directory
        updated_args = []
        for arg in server_config['args']:
            if 'tmp/tmp/out' in arg:
                updated_args.append(str(temp_dir_path / 'out'))
            elif arg.startswith('tools/'):
                updated_args.append('agent/' + arg)
            else:
                updated_args.append(arg)
        
        mcp_servers[server_name] = {
            "command": server_config['command'],
            "args": updated_args,
            "transport": "stdio"
        }
    
    return llm, mcp_servers


async def create_langchain_agent(llm, mcp_servers):
    """Create LangChain ReAct agent with MCP tools"""
    # Create MCP client
    client = MultiServerMCPClient(mcp_servers)
    
    try:
        # Get tools from all MCP servers
        tools = await client.get_tools()
        print(f"Successfully loaded {len(tools)} tools from MCP servers")
        
        # Create ReAct agent
        agent = create_react_agent(llm, tools)
        
        return agent, client
    except Exception as e:
        print(f"Error creating agent: {e}")
        if hasattr(client, 'close'):
            await client.close()
        raise


def load_questions(test_json_path: str = 'benchmark/question.json'):
    """Load evaluation questions"""
    with open(test_json_path, 'r') as f:
        test_json = json.load(f)

    out = []
    for _, (question_idx, question_info) in enumerate(test_json.items()):
        AP_INDEX = 0 if question_info['evaluation'][0]['type'] == 'autonomous planning' else 1
        data = question_info['evaluation'][AP_INDEX].get('data', None)
        data = question_info['evaluation'][1 - AP_INDEX].get('data', None) if data is None else data

        if data is None:
            continue
        out.append({
            "question_id": question_idx,
            "auto": question_info['evaluation'][AP_INDEX]['question'],
            "instruct": question_info['evaluation'][1 - AP_INDEX]['question'],
            "data": data,
            "choices": question_info.get('choices', None)
        })

    return out


def extract_answer_from_response(response):
    """Extract final answer from agent response"""
    messages = response.get("messages", [])
    
    # Look for the final answer in the last assistant message
    for message in reversed(messages):
        if hasattr(message, 'type') and message.type == 'ai':
            content = message.content
            if '<Answer>' in content and '</Answer>' in content:
                # Extract answer between tags
                start = content.find('<Answer>') + len('<Answer>')
                end = content.find('</Answer>')
                return content[start:end].strip()
            return content
    
    return "No answer found"


async def handle_question(agent, question, chat_log_path):
    """Handle a single question with the LangChain agent"""
    try:
        # Prepare query
        query = question['auto'] + question['data'] if autoplanning else \
            question['instruct'] + question['data']

        if question['choices']:
            query += '\n'.join([''] + [
                '{}.{}'.format(chr(ord('A') + i), choice) 
                for i, choice in enumerate(question['choices'])
            ])

        # Add system prompt
        full_query = f"{sys_prompt}\n\nQuestion: {query}"
        
        print(f"\n--- Processing Question {question['question_id']} ---")
        print(f"Query: {query[:200]}...")
        
        # Save user message to chat log
        user_message = {
            "name": "user",
            "role": "user", 
            "content": full_query,
            "metadata": {"question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, user_message)
        
        # Invoke agent with configuration to prevent infinite loops
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=full_query)]},
            config={
                "recursion_limit": 50,  # Increase recursion limit
                "max_execution_time": 300,  # 5 minutes timeout
            }
        )
        
        # Extract final answer
        final_answer = extract_answer_from_response(response)
        
        # Convert response to AgentScope-compatible format for logging
        conversation_log = []
        
        for message in response.get("messages", []):
            if hasattr(message, 'type'):
                if message.type == 'human':
                    # User message
                    conversation_log.append({
                        "role": "user",
                        "content": message.content
                    })
                
                elif message.type == 'ai':
                    # Assistant message - handle both thinking and tool calls
                    assistant_content = []
                    
                    # First check if there's thinking content (text before tool calls)
                    if message.content and message.content.strip():
                        assistant_content.append({
                            "type": "text",
                            "content": message.content
                        })
                    
                    # Then check for tool calls
                    if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                        # Format tool calls in AgentScope style
                        for tool_call in message.additional_kwargs['tool_calls']:
                            try:
                                arguments = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                            except:
                                arguments = tool_call['function']['arguments']
                            
                            assistant_content.append({
                                "name": tool_call['function']['name'],
                                "input": arguments
                            })
                    
                    # Only add to log if there's actual content
                    if assistant_content:
                        conversation_log.append({
                            "role": "assistant", 
                            "content": assistant_content
                        })
                
                elif message.type == 'tool':
                    # Tool result message in AgentScope format
                    conversation_log.append({
                        "role": "tool",
                        "name": message.name,
                        "content": [{
                            "output": [{
                                "text": str(message.content),
                            }],
                        }]
                    })
        
        # Save detailed messages to .chat file (AgentScope format)
        for message in response.get("messages", []):
            if hasattr(message, 'type'):
                if message.type == 'human':
                    # Skip user message for .chat as it's already saved
                    continue
        
                elif message.type == 'ai':
                    # Assistant message - handle both thinking and tool calls for .chat file
                    assistant_chat_content = []
                    
                    # First check if there's thinking content (text before tool calls)
                    if message.content and message.content.strip():
                        assistant_chat_content.append({
                            "type": "text",
                            "text": message.content
                        })
                    
                    # Then check for tool calls
                    if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                        # Format tool calls like AgentScope
                        for tool_call in message.additional_kwargs['tool_calls']:
                            try:
                                arguments = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                            except:
                                arguments = tool_call['function']['arguments']
                            
                            assistant_chat_content.append({
                                "type": "tool_use",
                                "id": tool_call['id'],
                                "name": tool_call['function']['name'],
                                "input": arguments
                            })
                    
                    # Save assistant message with both thinking and tool calls
                    if assistant_chat_content:
                        assistant_message = {
                            "name": question['question_id'],
                            "role": "assistant",
                            "content": assistant_chat_content,
                            "metadata": None
                        }
                        save_chat_message(chat_log_path, assistant_message)
                
                elif message.type == 'tool':
                    # Tool result message
                    tool_result_message = {
                        "name": "system",
                        "role": "system",
                        "content": [{
                            "type": "tool_result",
                            "id": getattr(message, 'tool_call_id', 'unknown'),
                            "output": [{"type": "text", "text": str(message.content), "annotations": None, "meta": None}],
                            "name": message.name
                        }],
                        "metadata": None
                    }
                    save_chat_message(chat_log_path, tool_result_message)
        
        # Log the conversation in the same format as original code
        logger.info("Chat Content", question['question_id'], conversation_log, final_answer)
        
        print(f"Final Answer: {final_answer}")
        return final_answer
        
    except Exception as e:
        error_msg = f"Error processing question {question['question_id']}: {e}"
        print(error_msg)
        
        # Save error to chat log
        error_message = {
            "name": "system",
            "role": "system",
            "content": [{"type": "text", "content": error_msg}],
            "metadata": {"error": True, "question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, error_message)
        
        logger.info(question['question_id'], [], error_msg)
        return f"Error: {e}"


async def main():
    """Main evaluation function"""
    print("Initializing LangChain-based Earth Science Agent...")
    
    # Initialize global parameters
    init_global_params()
    
    # Initialize chat logger
    chat_log_path = init_chat_logger()
    print(f"Chat log will be saved to: {chat_log_path}")
    
    # Load configuration and create agent
    llm, mcp_servers = load_langchain_config()
    agent, client = await create_langchain_agent(llm, mcp_servers)
    
    try:
        # Load questions
        questions = load_questions()[0:188]  # First 100 questions for testing
        print(f"Loaded {len(questions)} questions for evaluation")
        
        # Process questions
        results = []
        for question in tqdm(questions, desc="Processing questions"):
            answer = await handle_question(agent, question, chat_log_path)
            results.append({
                "question_id": question['question_id'],
                "answer": answer
            })
            
            # Optional: Add delay between questions to avoid rate limiting
            await asyncio.sleep(1)
        
        # Save results summary
        results_path = temp_dir_path / "results_summary.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"\nEvaluation completed! Results saved to {results_path}")
        print(f"Detailed logs available at: {temp_dir_path}")
        print(f"Chat history saved to: {chat_log_path}")
        
    except Exception as e:
        print(f"Error in main evaluation: {e}")
        raise
    
    finally:
        # Clean up
        if hasattr(client, 'close'):
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())