"""
Code grader for the CodeDebugger RL Environment.

Safely executes submitted Python code in a subprocess with timeout
and compares output against expected output.
"""

import subprocess
import sys
import textwrap
import tempfile
import os
from typing import Tuple


# Dangerous patterns that should never be executed
BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "__import__",
    "exec(",
    "eval(",
    "open(",
    "os.system",
    "os.popen",
    "os.remove",
    "os.rmdir",
    "os.unlink",
    "shutil.rmtree",
    "socket",
    "urllib",
    "requests",
    "http",
    "ftplib",
    "smtplib",
]

# Safe imports that are allowed even if they look like blocked patterns
SAFE_IMPORTS = [
    "import copy",
    "import threading",
    "import collections",
    "import math",
    "import re",
    "import json",
    "import itertools",
    "import functools",
    "import typing",
    "from collections",
    "from typing",
    "from itertools",
    "from functools",
]

EXECUTION_TIMEOUT = 10  # seconds
MAX_OUTPUT_LENGTH = 2000  # chars


def is_trivial_exploit(code: str, expected_output: str) -> bool:
    """
    Detect if submitted code is a trivial exploit (just printing the expected output
    as a literal string, without actually solving the logic problem).

    An exploit looks like:
      print("x is greater than 5")   ← when expected_output is "x is greater than 5"
      print(55)                       ← when expected_output is "55"

    Returns True if the code is a trivial exploit.
    """
    if not expected_output:
        return False

    stripped = code.strip()
    expected_stripped = expected_output.strip()

    # Must be short (exploits are simple one-liners)
    if len(stripped) > 200:
        return False

    # Check for single-line that just prints the expected output literally
    single_line_patterns = [
        f'print("{expected_stripped}")',
        f"print('{expected_stripped}')",
        f"print({expected_stripped})",       # for numeric outputs like 55
        f'print("""{expected_stripped}""")',
    ]
    for p in single_line_patterns:
        if stripped == p or stripped == p + "\n":
            return True

    # Multi-line exploit: code is just a series of print statements matching expected
    expected_lines = expected_stripped.split("\n")
    if len(expected_lines) > 1:
        code_lines = [l.strip() for l in stripped.split("\n") if l.strip()]
        print_lines = [l for l in code_lines if l.startswith("print(")]
        if len(print_lines) == len(expected_lines) == len(code_lines):
            # All lines are prints and match expected outputs
            return True

    return False


def is_safe_code(code: str) -> Tuple[bool, str]:

    """
    Check if the submitted code is safe to execute.
    
    Args:
        code: Python code string
    
    Returns:
        (is_safe, reason) tuple
    """
    lines = code.split("\n")
    
    for line in lines:
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        
        # Check if it's a known safe import
        is_safe_line = any(stripped.startswith(safe) for safe in SAFE_IMPORTS)
        if is_safe_line:
            continue
        
        # Check for blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if pattern in line:
                return False, f"Blocked pattern detected: '{pattern}'"
    
    return True, "OK"


def run_code(code: str, timeout: int = EXECUTION_TIMEOUT) -> Tuple[bool, str, str]:
    """
    Execute Python code in an isolated subprocess.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
    
    Returns:
        (success, stdout, stderr) tuple
    """
    # Security check
    safe, reason = is_safe_code(code)
    if not safe:
        return False, "", f"Security violation: {reason}"
    
    # Write code to a temp file
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name
        
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Restrict environment variables
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        
        stdout = result.stdout[:MAX_OUTPUT_LENGTH]
        stderr = result.stderr[:MAX_OUTPUT_LENGTH]
        success = result.returncode == 0
        
        return success, stdout, stderr
    
    except subprocess.TimeoutExpired:
        return False, "", f"Execution timed out after {timeout} seconds"
    
    except Exception as e:
        return False, "", f"Execution error: {str(e)}"
    
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def normalize_output(output: str) -> str:
    """
    Normalize output for comparison — strips trailing whitespace and normalizes newlines.
    """
    lines = output.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.rstrip() for line in lines]
    # Remove trailing empty lines
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def grade_submission(submitted_code: str, expected_output: str) -> dict:
    """
    Grade a submitted code fix.

    Checks for exploits (hardcoded print statements), security issues, then
    executes the code and compares output to expected_output.

    Args:
        submitted_code: The code submitted by the agent
        expected_output: The expected output string

    Returns:
        Grade result dict with keys:
            - passed (bool): Whether the submission is correct
            - reward (float): Reward value (-1.0 to 1.0)
            - actual_output (str): What the code actually printed
            - error (str | None): Error message if execution failed
            - message (str): Human-readable result message
    """
    # Exploit detection — reject trivial hardcoded-print submissions
    if is_trivial_exploit(submitted_code, expected_output):
        return {
            "passed": False,
            "reward": -0.5,
            "actual_output": "",
            "error": "Exploit detected: submission appears to just print the expected output without solving the problem.",
            "message": "❌ Exploit detected. You must fix the actual bug, not just print the expected answer.",
        }

    # Run the submitted code
    success, stdout, stderr = run_code(submitted_code)
    
    if not success:
        error_msg = stderr.strip() if stderr else "Unknown execution error"
        return {
            "passed": False,
            "reward": -0.5,
            "actual_output": "",
            "error": error_msg,
            "message": f"Code execution failed: {error_msg[:200]}",
        }
    
    # Compare output
    actual = normalize_output(stdout)
    expected = normalize_output(expected_output)
    
    if actual == expected:
        return {
            "passed": True,
            "reward": 1.0,
            "actual_output": actual,
            "error": None,
            "message": "✅ Correct! Output matches expected.",
        }
    else:
        return {
            "passed": False,
            "reward": -0.2,
            "actual_output": actual,
            "error": None,
            "message": (
                f"❌ Wrong output.\n"
                f"Expected: {repr(expected[:100])}\n"
                f"Got:      {repr(actual[:100])}"
            ),
        }
