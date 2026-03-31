"""LiveCodeBench question formatting utilities."""


def format_livecodebench_question(
    question_content: str, starter_code: str | None = None
) -> str:
    """Format a LiveCodeBench question according to Artificial Analysis specifications.

    Args:
        question_content: The problem statement
        starter_code: Optional starter code (if available)

    Returns:
        Formatted question string
    """
    # Check if starter code is provided and not empty
    has_starter = (
        starter_code
        and isinstance(starter_code, str)
        and starter_code.strip()
        and starter_code.lower() != "nan"
    )

    if has_starter:
        # Question with starter code format
        return f"""### Question:
{question_content}

### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters. Your code must be a complete runnable program. After the class definition, add an `if __name__ == "__main__":` block that reads each method argument from stdin (one per line, parse each with `json.loads(input())`), creates a Solution instance, calls the method, and prints the result with `print(result)`.
```python
{starter_code}
```

### Answer: (use the provided format with backticks)"""
    else:
        # Question without starter code format
        return f"""### Question:
{question_content}

### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
# YOUR CODE HERE
```

### Answer: (use the provided format with backticks)"""
