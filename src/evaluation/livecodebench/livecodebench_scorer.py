import re
import json
import subprocess
import tempfile
import os


# Regex to extract code from markdown code blocks
CODE_EXTRACTION_REGEX = r"(?<=```python\n)((?:\n|.)+?)(?=\n```)"


def extract_code(response: str) -> str | None:
    """Extract Python code from the agent's response.

    Args:
        response: The full response text from the agent

    Returns:
        Extracted Python code or None if no code found
    """
    match = re.search(CODE_EXTRACTION_REGEX, response, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def run_code_with_input(
    code: str, test_input: str, timeout: int = 10
) -> tuple[bool, str]:
    """Execute Python code with given input via subprocess.

    Args:
        code: Python code to execute
        test_input: Input to provide via stdin
        timeout: Execution timeout in seconds

    Returns:
        Tuple of (success, output/error)
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
        finally:
            os.unlink(tmp_path)

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


def score_livecodebench(question: str, ground_truth: str, predicted: str) -> float:
    """Score a LiveCodeBench solution using pass@1 evaluation.

    Args:
        question: The problem statement (text only, not JSON)
        ground_truth: JSON string containing test cases
        predicted: The agent's response containing code

    Returns:
        1.0 if all test cases pass, 0.0 otherwise
    """
    # Extract code from response
    code = extract_code(predicted)
    if not code:
        return 0.0

    # Parse test cases from ground_truth (which may be double-encoded JSON)
    try:
        test_cases = json.loads(ground_truth)
        # Handle double-encoded JSON (string -> string -> list)
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)
        if not isinstance(test_cases, list):
            return 0.0
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not test_cases:
        return 0.0

    # Run all test cases
    passed = 0
    for test_case in test_cases:
        test_input = test_case.get("input", "")
        expected_output = test_case.get("output", "").strip()

        success, actual_output = run_code_with_input(code, test_input, timeout=5)

        if success and actual_output == expected_output:
            passed += 1

    # Pass@1: all tests must pass
    return 1.0 if passed == len(test_cases) else 0.0
