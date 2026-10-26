---
name: targeted-test-runner
description: Запускает юнит и интеграционные тесты без засорения контекста, возвращая строгий JSON-отчет о прошедших и упавших тестах.
---

# Инструкция скилла Targeted Test Runner

Для прогона тестов используй только обертку:

```bash
python3 .agents/skills/targeted-test-runner/scripts/run_tests.py
```

Не сдавай задачу оркестратору, пока "ok": true не будет достигнут.
