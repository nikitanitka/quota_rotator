---
name: contract-and-type-guard
description: Выполняет детерминированную проверку синтаксиса, AST-политик (запрет Any, пустых except, console.log, TODO-заглушек) и статическую проверку типов. Вызывается автоматически хуком PostToolUse или вручную субагентами.
---

# Инструкция скилла Contract & Type Guard

Запрещено проверять код угадыванием. Запусти детерминированный валидатор через `run_command`:

```bash
python3 .agents/skills/contract-and-type-guard/scripts/verify_code.py --scope=.
```

Контракт ответа (Structured JSON)

    Если "ok": true (Exit Code 0) — код чист.

    Если "ok": false (Exit Code 1) — изучи массив diagnostics (поля file, line, rule, message), исправь каждое нарушение и перезапусти скрипт.
