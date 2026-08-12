"""Repository guard for the verifier's most important security invariant."""

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "src" / "duraseed"


def test_source_never_calls_general_purpose_code_evaluators() -> None:
    forbidden_names = {"eval", "exec", "compile"}
    violations: list[str] = []

    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                violations.append(f"{path}:{node.lineno}: {node.func.id}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "literal_eval"
            ):
                violations.append(f"{path}:{node.lineno}: literal_eval")

    assert violations == []
