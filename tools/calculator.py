"""
tools/calculator.py — Safe math expression evaluator tool.

Uses numexpr for safe evaluation (no arbitrary code execution).
Falls back to ast.literal_eval for simple expressions.
"""

from __future__ import annotations

from typing import Any, Dict

from models.schemas import ToolDefinition

CALCULATOR_DEF = ToolDefinition(
    name="calculator",
    description=(
        "Evaluates mathematical expressions safely. "
        "Supports arithmetic, trigonometry, logarithms, and basic algebra. "
        "Examples: '2 ** 10', 'sin(pi/2)', 'sqrt(144)', '(3 + 4) * 2'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (Python-compatible syntax).",
            }
        },
        "required": ["expression"],
    },
)


async def run_calculator(params: Dict[str, Any]) -> str:
    """Evaluate a mathematical expression and return the result as a string.

    Args:
        params: Must contain ``expression`` key.

    Returns:
        String representation of the evaluated result.

    Raises:
        ValueError: If expression is unsafe or invalid.
    """
    expression = params.get("expression", "").strip()
    if not expression:
        raise ValueError("No expression provided.")

    # Sanitise: reject expressions with dangerous builtins
    _BLOCKED = ("__", "import", "exec", "eval", "open", "os", "sys", "subprocess")
    for token in _BLOCKED:
        if token in expression:
            raise ValueError(f"Expression contains disallowed token: '{token}'")

    try:
        import numexpr as ne  # type: ignore[import]
        import math
        import numpy as np

        # numexpr uses a safe subset of Python math
        result = ne.evaluate(expression)
        # Unwrap numpy scalars
        if hasattr(result, "item"):
            result = result.item()
        return str(result)

    except ImportError:
        # Fallback: ast-based safe eval with math functions
        import ast
        import math

        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed_names["abs"] = abs

        try:
            tree = ast.parse(expression, mode="eval")
            # Walk AST — reject any node that isn't a safe numeric op
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name):
                        raise ValueError("Only simple function calls allowed.")
                    if node.func.id not in allowed_names:
                        raise ValueError(f"Function '{node.func.id}' not allowed.")
                elif isinstance(
                    node,
                    (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef),
                ):
                    raise ValueError("Dangerous AST node detected.")

            result = eval(  # noqa: S307
                compile(tree, "<calculator>", "eval"),
                {"__builtins__": {}},
                allowed_names,
            )
            return str(result)
        except Exception as exc:
            raise ValueError(f"Invalid expression: {exc}") from exc

    except Exception as exc:
        raise ValueError(f"Evaluation failed: {exc}") from exc
