#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import requests
import os
import sys
import re
from pathlib import Path

def gpt4o_polish_answer(question, answer):
    """
    Use GPT-4o to extract the final answer from model's response
    """
    # API configuration
    BASE_URL = ""##TODO: add the base url
    API_KEY = ''##TODO: add the api key
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Construct prompt for GPT-4o
    prompt = f"""You are tasked with extracting the final answer from a model's response to a multiple-choice question.

Question:
{question}

Model's Response:
{answer}

Please extract ONLY the final answer from the model's response. The answer should be in the format <Answer>X<Answer> where X is the choice (A, B, C, or D).
If the model provided multiple answers or explanations, extract only the final choice.
If the answer contains something like "D.8.5 K higher", just extract "D".

Return ONLY the answer in this exact format: <Answer>X<Answer>
Do not include any explanation or additional text.
Attention:
The most important thing is:
You just need to extract the normal answer, if the "answer" has some errors, you just need to keep it and don't give me a answer determined by yourself! For example, the model_answer is Sorry, need more steps to process this request. you just need to keep this, don't give me an answer you guess!!!"""
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts final answers from model responses."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # Low temperature for consistent extraction
        "max_tokens": 50
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content'].strip()
        
        # Extract answer from GPT response if it's not already in the correct format
        match = re.search(r'<Answer>([A-D])[^<]*<Answer>', result)
        if match:
            return f"<Answer>{match.group(1)}<Answer>"
        else:
            # Try to find just a letter if GPT didn't format it correctly
            match = re.search(r'\b([A-D])\b', result)
            if match:
                return f"<Answer>{match.group(1)}<Answer>"
            else:
                print(f"Warning: Could not extract answer from GPT response: {result}")
                return result
                
    except Exception as e:
        print(f"Error calling GPT-4o API: {e}")
        # Fallback: try to extract answer directly from original response
        match = re.search(r'<Answer>([^<]+)<Answer>', answer)
        if match:
            # Extract just the letter if present
            letter_match = re.search(r'([A-D])', match.group(1))
            if letter_match:
                return f"<Answer>{letter_match.group(1)}<Answer>"
        return answer


def load_log_file(log_path):
    """
    Load and parse the log file to extract questions
    """
    questions = {}

    with open(log_path, 'r', encoding='utf-8') as f:
        buffer = ""
        for line in f:
            buffer += line
            try:
                data = json.loads(buffer) 
            except json.JSONDecodeError:
                continue  
            else:
                buffer = ""

                question_id = data.get('question_index', '')
                
                if isinstance(question_id, list):
                    if len(question_id) > 0:
                        question_id = str(question_id[0])
                    else:
                        question_id = ''
                elif not isinstance(question_id, str):
                    question_id = str(question_id)
                
                if question_id.startswith('question'):
                    question_id = question_id[8:]

                # 提取 user 的内容
                conversations = data.get('conversations', [])
                for conv in conversations:
                    if isinstance(conv, dict) and conv.get('role') == 'user':
                        user_content = conv.get('content', '')
                        questions[question_id] = user_content
                        break

    return questions


def polish_results(base_dir):
    """
    Polish the results_summary.json file using GPT-4o
    
    Args:
        base_dir: Directory containing the log file and results_summary.json
    """
    base_path = Path(base_dir)
    
    # Find the log file
    log_files = list(base_path.glob("*.log"))
    if not log_files:
        print(f"Error: No log file found in {base_dir}")
        return
    
    log_file = log_files[0]
    results_file = base_path / "results_summary.json"
    output_file = base_path / "results_summary_polished.json"
    
    if not results_file.exists():
        print(f"Error: results_summary.json not found in {base_dir}")
        return
    
    print(f"Processing files in {base_dir}")
    print(f"Log file: {log_file}")
    print(f"Results file: {results_file}")
    
    # Load questions from log file
    print("Loading questions from log file...")
    questions = load_log_file(log_file)
    print(f"Loaded {len(questions)} questions")
    
    # Load results
    print("Loading results...")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Process each result
    polished_results = []
    for item in results:
        question_id = item.get('question_id', '')
        answer = item.get('answer', '')
        
        # Get the corresponding question
        question = questions.get(question_id, '')
        
        if not question:
            print(f"Warning: No question found for ID {question_id}")
            polished_answer = answer
        else:
            print(f"Processing question {question_id}...")
            # Use GPT-4o to polish the answer
            polished_answer = gpt4o_polish_answer(question, answer)
        
        # Create polished result
        polished_item = {
            "question_id": question_id,
            "original_answer": answer,
            "final_answer": polished_answer
        }
        polished_results.append(polished_item)
    
    # Save polished results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(polished_results, f, ensure_ascii=False, indent=2)
    
    print(f"Polished results saved to: {output_file}")
    
    # Print summary
    print(f"\nProcessed {len(polished_results)} questions")
    success_count = sum(1 for r in polished_results if '<Answer>' in r.get('polished_answer', ''))
    print(f"Successfully extracted answers: {success_count}/{len(polished_results)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process specified directory
        base_dir = sys.argv[1]
    else:
        # Default directory
        base_dir = "" ##TODO: add the base path
        ## For example: "/mnt/petrelfs/fengpeilin/EarthLMM/EO_Langchain/evaluate_langchain/qwen3max_IF_25-09-06_02-18"
    
    polish_results(base_dir)