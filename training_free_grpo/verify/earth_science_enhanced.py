"""
Earth Science Verification Function - 选择题专用（增强版）
支持：1. 字符串匹配（快速）2. LLM Judge（兜底）
"""
import re
import asyncio
from typing import Dict, Optional


async def verify_earth_science_answer(sample, judge_llm=None, **kwargs) -> Dict:
    """
    验证选择题答案（两阶段）

    阶段1: 尝试提取选项字母并精确匹配（快速、免费）
    阶段2: 如果提取失败，使用 LLM Judge（慢、需要 API）

    Args:
        sample: 包含 response, correct_answer, choices 的样本对象
        judge_llm: 可选的 LLM Judge（用于语义判断）

    Returns:
        {"reward": 0.0 或 1.0, "reasoning": "原因"}
    """
    # 异常处理
    if sample.response == "TIMEOUT_ERROR":
        return {"reward": 0.0, "reasoning": "Timeout"}

    if sample.response and "ERROR" in sample.response:
        return {"reward": 0.0, "reasoning": "Error"}

    if not sample.response:
        return {"reward": 0.0, "reasoning": "Empty response"}

    if not sample.correct_answer:
        return {"reward": 0.0, "reasoning": "No ground truth"}

    # 阶段1: 尝试提取选项字母（快速路径）
    extracted = _extract_choice(sample.response)

    if extracted:
        # 提取成功，直接匹配
        if extracted.upper() == sample.correct_answer.upper().strip():
            return {"reward": 1.0, "reasoning": f"Correct: {extracted}"}
        else:
            return {"reward": 0.0, "reasoning": f"Wrong: {extracted} != {sample.correct_answer}"}

    # 阶段2: 提取失败，使用 LLM Judge（如果提供）
    if judge_llm:
        try:
            result = await _llm_judge(sample, judge_llm)
            return result
        except Exception as e:
            return {"reward": 0.0, "reasoning": f"LLM judge error: {e}"}

    # 都失败了
    return {"reward": 0.0, "reasoning": f"Cannot extract choice from: {sample.response[:50]}"}


def _extract_choice(response: str) -> Optional[str]:
    """从回答中提取选项字母 A/B/C/D"""
    if not response:
        return None

    # Pattern 1: <Answer>C</Answer>
    match = re.search(r'<Answer>\s*([A-D])\s*</Answer>', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: Answer: C or ANSWER: C
    match = re.search(r'(?:answer|ANSWER)[\s:]+([A-D])(?:\s|$|\.)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Option C or option C
    match = re.search(r'(?:option|Option|OPTION)[\s:]+([A-D])(?:\s|$|\.)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 4: (C) or [C]
    match = re.search(r'[\(\[]([A-D])[\)\]]', response)
    if match:
        return match.group(1).upper()

    # Pattern 5: Standalone A/B/C/D at the end or beginning
    match = re.search(r'(?:^|\s)([A-D])(?:\s|$|\.)', response[-50:])  # Check last 50 chars
    if match:
        return match.group(1).upper()

    # Pattern 6: "is C" or "choose C"
    match = re.search(r'(?:is|choose|select)[\s:]+([A-D])(?:\s|$|\.)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


async def _llm_judge(sample, judge_llm) -> Dict:
    """使用 LLM 判断答案是否正确（兜底方案）"""
    # 构建选项文本
    choices_text = "\n".join([
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(sample.choices)
    ]) if sample.choices else ""

    prompt = f"""You are a teacher grading a multiple-choice quiz.

QUESTION: {sample.question}

CHOICES:
{choices_text}

CORRECT ANSWER: {sample.correct_answer}

STUDENT ANSWER: {sample.response}

Task: Determine if the student's answer is CORRECT or INCORRECT.
- If the student clearly chose the correct option (even without mentioning the letter), grade as CORRECT.
- If the student mentioned the correct percentage/value from the correct option, grade as CORRECT.
- Otherwise, grade as INCORRECT.

Format:
EXPLANATION: [your reasoning]
GRADE: CORRECT or INCORRECT

Begin:"""

    try:
        response = await judge_llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # 解析判断结果
        response_text = response_text.replace("**", "")
        if "GRADE:" in response_text:
            grade_line = response_text.split("GRADE:")[-1].strip().upper()
            is_correct = "CORRECT" in grade_line and "INCORRECT" not in grade_line
        else:
            is_correct = "CORRECT" in response_text.upper() and "INCORRECT" not in response_text.upper()

        return {
            "reward": 1.0 if is_correct else 0.0,
            "reasoning": f"LLM judge: {'CORRECT' if is_correct else 'INCORRECT'}"
        }
    except Exception as e:
        return {"reward": 0.0, "reasoning": f"LLM judge error: {e}"}
