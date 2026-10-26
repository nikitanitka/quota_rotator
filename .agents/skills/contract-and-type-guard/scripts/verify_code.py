#!/usr/bin/env python3
import ast
import json
import sys
from pathlib import Path

FORBIDDEN_COMMENTS = ["type: ignore", "ts-ignore", "TODO:", "FIXME:", "placeholder"]
diagnostics = []


def check_python_file(path: Path):
    rel = str(path)
    source = path.read_text(encoding="utf-8", errors="ignore")

    # 1. Проверка на заглушки и подавление типизации
    for idx, line in enumerate(source.splitlines(), start=1):
        for marker in FORBIDDEN_COMMENTS:
            if marker.lower() in line.lower():
                diagnostics.append(
                    {
                        "file": rel,
                        "line": idx,
                        "rule": "NO_SLOP_MARKERS",
                        "message": f"Запрещенный маркер или заглушка '{marker}'. Реализуй логику полностью.",
                    }
                )

    # 2. AST-анализ: синтаксис, запрет пустых except и функций без Return Type
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as e:
        diagnostics.append(
            {
                "file": rel,
                "line": e.lineno or 1,
                "rule": "SYNTAX_ERROR",
                "message": f"Синтаксическая ошибка: {e.msg}",
            }
        )
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                diagnostics.append(
                    {
                        "file": rel,
                        "line": node.lineno,
                        "rule": "NO_SILENT_EXCEPT",
                        "message": "Пустой блок 'except: pass' запрещен. Логируй или пробрасывай типизированную ошибку.",
                    }
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_") and node.returns is None:
                diagnostics.append(
                    {
                        "file": rel,
                        "line": node.lineno,
                        "rule": "EXPLICIT_RETURN_TYPE",
                        "message": f"Публичная функция '{node.name}' обязана иметь явную аннотацию возвращаемого типа (-> ...).",
                    }
                )


for py_file in Path(".").rglob("*.py"):
    if any(
        p in py_file.parts for p in (".venv", "venv", ".agents", "__pycache__", ".git")
    ):
        continue
    check_python_file(py_file)

ok = len(diagnostics) == 0
print(
    json.dumps(
        {"ok": ok, "error_count": len(diagnostics), "diagnostics": diagnostics[:20]},
        ensure_ascii=False,
        indent=2,
    )
)
sys.exit(0 if ok else 1)
