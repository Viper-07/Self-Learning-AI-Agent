"""
tools/python_executor.py — Sandboxed Python code execution tool.

Runs Python code in a subprocess with a timeout for safety.
stdout/stderr are captured and returned to the agent.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from typing import Any, Dict

from models.schemas import ToolDefinition

PYTHON_EXECUTOR_DEF = ToolDefinition(
    name="python_executor",
    description=(
        "Executes Python code and returns stdout/stderr. "
        "Use for data processing, calculations, file parsing, or any task "
        "that benefits from running actual Python. "
        "Runs in an isolated subprocess with a 30-second timeout."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 30, max: 60).",
                "default": 30,
            },
        },
        "required": ["code"],
    },
    is_dangerous=True,
)

# Characters that indicate clearly malicious intent
_HARD_BLOCKED = [
    "os.system",
    "subprocess",
    "__import__",
    "eval(",
    "exec(",
    "getattr(",
    "open(",
    "shutil",
]


async def run_python(params: Dict[str, Any]) -> str:
    """Execute Python code in a subprocess and return output.

    Args:
        params: Must contain ``code``. Optionally ``timeout`` (int, seconds).

    Returns:
        String combining stdout and stderr.

    Raises:
        ValueError: For blocked code patterns.
        TimeoutError: If execution exceeds the timeout.
    """
    code: str = params.get("code", "").strip()
    timeout: int = min(int(params.get("timeout", 30)), 60)

    if not code:
        raise ValueError("No code provided.")

    for blocked in _HARD_BLOCKED:
        if blocked in code:
            raise ValueError(f"Code contains a blocked pattern: '{blocked}'")

    # Wrap the code to capture output and restrict environment
    wrapped = textwrap.dedent(f"""\
        import sys, io, traceback
        _out = io.StringIO()
        _err = io.StringIO()
        sys.stdout = _out
        sys.stderr = _err

        # Restrict built-ins to a safe subset
        safe_builtins = {{
            'abs': __builtins__.get('abs'),
            'all': __builtins__.get('all'),
            'any': __builtins__.get('any'),
            'bin': __builtins__.get('bin'),
            'bool': __builtins__.get('bool'),
            'chr': __builtins__.get('chr'),
            'complex': __builtins__.get('complex'),
            'dict': __builtins__.get('dict'),
            'divmod': __builtins__.get('divmod'),
            'enumerate': __builtins__.get('enumerate'),
            'filter': __builtins__.get('filter'),
            'float': __builtins__.get('float'),
            'format': __builtins__.get('format'),
            'frozenset': __builtins__.get('frozenset'),
            'hash': __builtins__.get('hash'),
            'hex': __builtins__.get('hex'),
            'int': __builtins__.get('int'),
            'isinstance': __builtins__.get('isinstance'),
            'issubclass': __builtins__.get('issubclass'),
            'iter': __builtins__.get('iter'),
            'len': __builtins__.get('len'),
            'list': __builtins__.get('list'),
            'map': __builtins__.get('map'),
            'max': __builtins__.get('max'),
            'min': __builtins__.get('min'),
            'next': __builtins__.get('next'),
            'oct': __builtins__.get('oct'),
            'ord': __builtins__.get('ord'),
            'pow': __builtins__.get('pow'),
            'print': __builtins__.get('print'),
            'range': __builtins__.get('range'),
            'repr': __builtins__.get('repr'),
            'reversed': __builtins__.get('reversed'),
            'round': __builtins__.get('round'),
            'set': __builtins__.get('set'),
            'slice': __builtins__.get('slice'),
            'sorted': __builtins__.get('sorted'),
            'str': __builtins__.get('str'),
            'sum': __builtins__.get('sum'),
            'tuple': __builtins__.get('tuple'),
            'type': __builtins__.get('type'),
            'zip': __builtins__.get('zip'),
        }}

        try:
            # Execute with restricted globals and builtins
            exec({repr(code)}, {{'__builtins__': safe_builtins}}, {{}})
        except Exception:
            _err.write(traceback.format_exc())
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(_out.getvalue(), end="")
            if _err.getvalue():
                print("STDERR:", _err.getvalue(), end="")
    """)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", wrapped,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError(f"Python execution timed out after {timeout}s.")

    output = stdout.decode("utf-8", errors="replace")
    err_output = stderr.decode("utf-8", errors="replace")

    result_parts = []
    if output:
        result_parts.append(output)
    if err_output:
        result_parts.append(f"[stderr]: {err_output}")
    if proc.returncode != 0 and not result_parts:
        result_parts.append(f"Process exited with code {proc.returncode}")

    return "\n".join(result_parts) if result_parts else "(no output)"
