import json
from collections import OrderedDict

def merge_adjacent_same_tools(input_file_path, output_file_path):
    """
    Merge consecutive tool calls with the same name
    
    Args:
        input_file_path: Path to the input JSON file
        output_file_path: Path to the output JSON file
    """
    # Read the original data
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process tool calls for each question
    merged_data = []

    for question_data in data:
        question_index = question_data.get('question_index', '')
        tool_calls = question_data.get('tool_calls', [])

        # Merge consecutive tool calls with the same name
        merged_tool_calls = merge_consecutive_same_tools(tool_calls)

        merged_question_data = {
            'question_index': question_index,
            'tool_calls': merged_tool_calls
        }
        merged_data.append(merged_question_data)

    # Save the merged data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Merging completed! A total of {len(merged_data)} questions processed.")
    print(f"Results saved to: {output_file_path}")

def merge_consecutive_same_tools(tool_calls):
    """
    Merge consecutive tool calls with the same name
    """
    if not tool_calls:
        return []

    merged = []
    i = 0

    while i < len(tool_calls):
        current_tool = tool_calls[i]

        # Find consecutive tool calls with the same name
        j = i + 1
        while j < len(tool_calls) and tool_calls[j]['name'] == current_tool['name']:
            j += 1

        # If only one tool, add it directly
        if j - i == 1:
            merged.append(current_tool)
        else:
            # Merge multiple tool calls with the same name
            group = tool_calls[i:j]
            merged_tool = merge_tool_group(group)
            merged.append(merged_tool)

        i = j

    return merged

def merge_tool_group(group):
    """
    Merge a group of tool calls with the same name, maintaining the original order of parameters
    """
    if not group:
        return None

    merged_tool = {
        'name': group[0]['name'],
        'arguments': OrderedDict(),
        'output': []
    }

    # Get the parameter order from the first tool call
    if isinstance(group[0].get('input'), dict):
        param_keys = list(group[0].get('input').keys())

        # Create lists for each parameter and collect values in the original order
        for key in param_keys:
            merged_tool['arguments'][key] = []
            for tool in group:
                if isinstance(tool.get('input'), dict) and key in tool['input']:
                    merged_tool['arguments'][key].append(tool['input'][key])

    # Merge the output into an array
    for tool in group:
        merged_tool['output'].append(tool.get('output'))

    return merged_tool

if __name__ == "__main__":
    # Set file paths
    input_file = "" #TODO: add input file path
    # For example: "/mnt/petrelfs/fengpeilin/EarthLMM/EO_Langchain/evaluate_langchain/qwen3max_IF_25-09-06_02-18/extracted_tool_calls.json"
    output_file = "" #TODO: add output file path
    # For example: "/mnt/petrelfs/fengpeilin/EarthLMM/EO_Langchain/evaluate_langchain/qwen3max_IF_25-09-06_02-18/merged_tool_calls_Model.json"

    merge_adjacent_same_tools(input_file, output_file)
