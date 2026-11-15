"""Built-in tools. Each one is safe to hand to a model that may be confused or misled:
the calculator only does arithmetic, and the file tools can't leave their folder."""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field

from .registry import ToolRegistry

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
MAX_FILE_BYTES = 64_000
MAX_EXPRESSION_CHARS = 500
MAX_RESULT_BITS = 4096  # about 1,200 digits: plenty for arithmetic, too small to stall the server


def _bits(value: float | int) -> float:
    return abs(value).bit_length() if isinstance(value, int) else math.log2(abs(value) or 1)


def evaluate(expression: str) -> float | int:
    """Arithmetic only: numbers, + - * / // % ** and brackets. Anything else raises ValueError.

    Results are size-checked *before* computing them: big-integer maths holds the GIL, so one
    `(9**99)**99**...` would freeze every request, and a thread can't be cancelled.
    """
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")

    def walk(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            if _bits(node.value) > MAX_RESULT_BITS:
                raise ValueError("number too large")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 1000 or (abs(left) > 1 and _bits(left) * abs(right) > MAX_RESULT_BITS):
                    raise ValueError("result too large")
            elif isinstance(node.op, ast.Mult) and _bits(left) + _bits(right) > MAX_RESULT_BITS:
                raise ValueError("result too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"not arithmetic: {ast.dump(node)[:60]}")

    return walk(ast.parse(expression, mode="eval").body)


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("that path is outside the workspace")
    return path


def register_builtins(registry: ToolRegistry, workspace: Path) -> ToolRegistry:
    root = workspace.resolve()

    @registry.tool
    def calculate(expression: Annotated[str, Field(description="Arithmetic, e.g. (1250.50 - 300) * 1.2")]) -> str:
        """Evaluate an arithmetic expression exactly. Use it instead of doing sums in your head."""
        result = evaluate(expression)
        return str(round(result, 10) if isinstance(result, float) else result)

    @registry.tool
    def current_time() -> str:
        """The current local date and time, with the time zone."""
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @registry.tool
    def list_files(
        folder: Annotated[str, Field(description="Folder inside the workspace; '.' for the top")] = ".",
    ) -> str:
        """List the files in the workspace folder you are allowed to read."""
        base = _inside(root, folder)
        if not base.is_dir():
            raise ValueError(f"no folder {folder!r} in the workspace")
        files = sorted(str(p.relative_to(root)) for p in base.rglob("*") if p.is_file())
        return "\n".join(files[:500]) or "(empty)"

    @registry.tool
    def read_file(path: Annotated[str, Field(description="File path relative to the workspace")]) -> str:
        """Read a text file from the workspace."""
        file = _inside(root, path)
        if not file.is_file():
            raise ValueError(f"no file {path!r} in the workspace")
        data = file.read_bytes()[:MAX_FILE_BYTES]
        return data.decode("utf-8", errors="replace")

    return registry
