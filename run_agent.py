import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import asyncio

from langchain.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient


SYS_PROMPT = '''
You are a helpful assistant.
'''

MODEL_NAME = 'gpt-4o'

async def process(agent, question):
    try:
        resp = await agent.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        messages = resp.get("messages", [])

        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content
                if '<Answer>' in content and '</Answer>' in content:
                    start = content.find('<Answer>') + len('<Answer>')
                    end = content.find('</Answer>')
                    return content[start:end].strip()
                return content
    except:
        return 'Error'


async def main():
    with open('agent_config.json', 'r') as f:
        config = json.load(f)

    model_config = next(
        (m for m in config["models"] if m["model_name"] == MODEL_NAME),
        None
    )

    if model_config is None:
        raise

    mcp_config = config.get('mcpServers', {})
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()

    model = init_chat_model(
        model_config["model_name"],
        model_provider="openai",
        openai_api_base=model_config.get('base_url', None),
        openai_api_key=model_config.get('api_key', 'dummy'),
        temperature=0.1
    )

    agent = create_react_agent(model, tools, prompt=SYS_PROMPT)

    query = 'What tools do you have?'
    response = await process(agent, query)
    print(response)



if __name__ == "__main__":
    asyncio.run(main())


        