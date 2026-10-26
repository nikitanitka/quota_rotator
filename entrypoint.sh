#!/bin/sh
set -e

# Автоматическая настройка прав на монтируемую директорию /app/data для appuser
if [ -d "/app/data" ]; then
    chown -R appuser:appuser /app/data 2>/dev/null || true
    chmod -R u+rwX,g+rwX,o+rwX /app/data 2>/dev/null || true
fi

# Запуск uvicorn от непривилегированного пользователя appuser
exec gosu appuser "$@"

