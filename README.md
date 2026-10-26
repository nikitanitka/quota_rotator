# Gemini Key Pool Router & Quota Control

Production-ready прокси-роутер и система балансировки пула бесплатных API-ключей Google Gemini с контролем квот (RPM/TPM/RPD), автосбросом в 10:00 МСК, прозрачным фейловером, очередью ожидания, поддержкой SSE-стриминга и клиентскими Ingress-ключами.

---

## Быстрый старт

```bash
# Запуск в фоновом режиме через Docker Compose
docker compose up --build -d

# Просмотр логов
docker compose logs -f gemini-router
```

Дашборд доступен по адресу: **`http://localhost:8000`**

---

## Документация

Подробное руководство по работе с роутером, настройке клиентских ключей и примеры интеграции доступны в файле:
📖 **[docs/ROUTER_USAGE_GUIDE.md](docs/ROUTER_USAGE_GUIDE.md)**

### Краткое оглавление руководства:
1. **Архитектура и возможности**: Least-Loaded балансировка, счетчики in-flight, сброс в 10:00 МСК, SSE стриминг.
2. **Развертывание**: Docker Compose, персистентность базы SQLite WAL, внутренняя сеть `ai-infra-net`.
3. **Веб-дашборд**: мониторинг пула, добавление исходящих ключей Google, интерактивный плейграунд.
4. **Аутентификация микросервисов**: создание клиентских ключей (`gkr-live-...`) с персональными лимитами RPM и RPD.
5. **Очередь ожидания (`wait_timeout`)**: обработка пиковых нагрузок без ошибок 429 (`X-Wait-Timeout: 60`, `-1` для 24 часов).
6. **Примеры кода**: Python (`google-genai`, `httpx`), Node.js (`@google/genai`), cURL.
7. **REST API Справочник**: эндпоинты `/v1beta/*` и административные `/api/*`.
8. **Таблица кодов ошибок**: расшифровка 401, 403, 429 (`CLIENT_RATE_LIMITED`, `QUEUE_TIMEOUT`, `ALL_KEYS_RATE_LIMITED`) и 502/503.

---

## Тестирование и верификация

```bash
# Запуск всех тестов (38 passed)
pytest tests/test_router.py

# Проверка типизации и целостности контрактов
python .agents/skills/contract-and-type-guard/scripts/verify_code.py
```
