# Руководство по работе с Gemini Key Pool Router

Полнофункциональный прокси-роутер и система балансировки пула бесплатных API-ключей Google Gemini с контролем квот (RPM/TPM/RPD), прозрачным фейловером, поддержкой стриминга (SSE), очередью ожидания и собственной системой клиентских Ingress-ключей.

---

## Содержание
1. [Архитектура и возможности](#1-архитектура-и-возможности)
2. [Быстрый старт и развертывание](#2-быстрый-старт-и-развертывание)
   - [Запуск через Docker Compose (Production 24/7)](#запуск-через-docker-compose-production-247)
   - [Локальный запуск без Docker](#локальный-запуск-без-docker)
3. [Веб-дашборд и мониторинг](#3-веб-дашборд-и-мониторинг)
4. [Аутентификация клиентских микросервисов](#4-аутентификация-клиентских-микросервисов)
5. [Управление очередью ожидания (Wait Timeout)](#5-управление-очередью-ожидания-wait-timeout)
6. [Примеры подключения микросервисов](#6-примеры-подключения-микросервисов)
   - [Python: Официальный SDK `google-genai`](#python-официальный-sdk-google-genai)
   - [Python: Библиотека `httpx` (Async / Streaming)](#python-библиотека-httpx-async--streaming)
   - [Node.js / TypeScript: `@google/genai`](#nodejs--typescript-googlegenai)
   - [cURL: Прямые HTTP-запросы](#curl-прямые-http-запросы)
7. [REST API Справочник](#7-rest-api-справочник)
8. [Коды ошибок и решение проблем](#8-коды-ошибок-и-решение-проблем)

---

## 1. Архитектура и возможности

```
                                      ┌─────────────────────────────────────────┐
                                      │       Внешние микросервисы / боты       │
                                      └────────────────────┬────────────────────┘
                                                           │
                                        Client API Key     │ Authorization: Bearer /
                                        (gkr-live-...)     │ x-goog-api-key / ?key=
                                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       Gemini Key Pool Router                                           │
│                                                                                                        │
│  ┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────────────┐  │
│  │ Ingress Auth & Quota   │ ───► │ Least-Loaded Balancer   │ ───► │ Stream Proxy / SSE & Fallback   │  │
│  │ (Client RPM / RPD)     │      │ (In-Flight Concurrency) │      │ (Transparent Retry 429/503)     │  │
│  └────────────────────────┘      └─────────────────────────┘      └─────────────────────────────────┘  │
│               │                               │                                     │                  │
│               ▼                               ▼                                     ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                    SQLite Database in WAL mode (./data/router.db)                                │  │
│  │           - client_keys, client_usage_events | keys, usage_events | 10:00 MSK Reset              │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                   │
                  Outgoing Google Pool Keys        │ x-goog-api-key: AQ.Ab8... / AIzaSy...
                  (Round-Robin / Least RPD)        ▼
                                      ┌─────────────────────────────────────────┐
                                      │        Google Gemini API Upstream       │
                                      │   (gemini-3.1-flash-lite, gemma, ...)   │
                                      └─────────────────────────────────────────┘
```

### Основные фичи:
- **Балансировка Least-Loaded RPD**: Роутер выбирает наименее нагруженный ключ по суточному лимиту, учитывая активные конкурентные соединения (`in-flight`).
- **Сброс суточных квот в 10:00 МСК**: Официальное время сброса бесплатных лимитов Google (07:00 UTC). Роутер автоматически сбрасывает счетчики RPD ровно в это время.
- **Прозрачный фейловер на 429 и 503**: При перегрузке или ошибке ключ отправляется в мягкий кулдаун (15 сек -> 60 сек), а запрос прозрачно перенаправляется на следующий доступный ключ.
- **Очередь ожидания (`wait_timeout`)**: Если все ключи заняты или на кулдауне, запрос не отбивается сразу, а ожидает в очереди освобождения ключа (значение `-1` позволяет фоновым задачам ждать до 24 часов).
- **Полноценный SSE-стриминг**: Бесшовное проксирование `streamGenerateContent` с подсчетом токенов из финального чанка `usageMetadata`.
- **Client Ingress API Keys**: Выпуск собственных клиентских токенов (`gkr-live-...`) для микросервисов с персональными лимитами RPM и RPD.
- **Интерактивный Web UI**: Дашборд реального времени на Tailwind CSS со счетчиками, прогресс-барами, плейграундом и управлением ключами.

---

## 2. Быстрый старт и развертывание

### Запуск через Docker Compose (Production 24/7)

Файл `docker-compose.yml` настроен для круглосуточной работы на VPS в общей сети микросервисов `ai-infra-net`:

```bash
# 1. Запустить сборку и фоновый запуск контейнера
docker compose up --build -d

# 2. Проверить статус контейнера и healthcheck
docker compose ps

# 3. Просмотр логов в реальном времени
docker compose logs -f gemini-router
```

#### Параметры контейнера:
- **Порт дашборда**: `http://localhost:8000` (проброшен наружу).
- **Внутреннее DNS-имя**: `http://gemini-router:8000` (доступно для всех микросервисов, подключенных к Docker-сети `ai-infra-net`).
- **Персистентность**: Директория `./data` на хосте монтируется в `/app/data`. База SQLite `router.db` полностью сохраняет ключи, историю и лимиты между пересборками.
- **Права доступа**: Контейнер использует паттерн `gosu` в `entrypoint.sh` — стартует инициализацию прав доступа под root и затем безопасно сбрасывает привилегии на непривилегированного пользователя `appuser (UID 10001)`, что исключает ошибку SQLite `attempt to write a readonly database`.

### Локальный запуск без Docker

```bash
# 1. Активировать виртуальное окружение
source .venv/bin/activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить Uvicorn сервер
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3. Веб-дашборд и мониторинг

Откройте в браузере: **`http://localhost:8000`**

### Основные блоки дашборда:
1. **Hero Stats Bar (Верхняя панель)**:
   - Общее число ключей в пуле, число активных (READY), находящихся на кулдауне (COOLDOWN) и исчерпанных (EXHAUSTED).
   - Обратный отсчет до следующего суточного сброса квот (в формате `ЧЧ:ММ:СС до сброса в 10:00 МСК`).
   - Суммарная суточная емкость пула (например, `41 / 2,000 RPD`).
2. **Key Management Pool (Исходящие ключи Google)**:
   - Карточки исходящих ключей с индикатором 60-секундного скользящего окна RPM, TPM и RPD.
   - Кнопка **+ Add New Key**: добавление нового ключа (поддерживаются ключи `AQ.*` и `AIza*`).
   - Кнопка **Pause / Resume**: временное исключение ключа из роутинга.
   - Кнопка **Reset CD**: ручной сброс кулдауна.
   - Кнопка **Delete**: безопасное удаление ключа из пула с подтверждением.
3. **Client API Keys (Входящие клиентские ключи)**:
   - Управление токенами доступа для ваших собственных микросервисов.
   - Кнопка **+ Create Client Key**: создание нового клиента с персональными лимитами RPM и RPD.
   - Однократный защищенный просмотр токена с кнопкой быстрого копирования.
   - Мониторинг расхода скорости и суточной квоты по каждому микросервису.
4. **Interactive Pool Router Playground**:
   - Тестирование промптов прямо из интерфейса.
   - Переключение моделей (Gemini 2.5/3/3.1 Flash, Gemma 4, TTS, Transcribe).
   - Поддержка действий `generateContent` и `streamGenerateContent`.
   - Режимы `Dry-Run Simulation` (бесплатная симуляция) и `Live Upstream` (реальный запрос в Google).
   - Поле **Queue Wait (sec)**: настройка допустимого ожидания в очереди.

---

## 4. Аутентификация клиентских микросервисов

Роутер защищает эндпоинты `/v1beta/*` с помощью собственной системы клиентских ключей.

> [!NOTE]
> Если в системе еще не создано ни одного клиентского ключа, роутер работает в **Open Access Mode** (обратная совместимость). Как только вы создаете первый клиентский ключ через дашборд или API, роутер автоматически включает строгую авторизацию.

### Передача токена клиентом (4 поддерживаемых формата):
Вы можете передать токен любым удобным способом:
1. **HTTP-заголовок `x-goog-api-key`** (дефолт в Google SDK):
   ```http
   x-goog-api-key: gkr-live-7f8a9b1c2d3e4f5a...
   ```
2. **HTTP-заголовок `Authorization: Bearer`** (стандарт OpenAI/FastAPI):
   ```http
   Authorization: Bearer gkr-live-7f8a9b1c2d3e4f5a...
   ```
3. **HTTP-заголовок `X-Router-Key`**:
   ```http
   X-Router-Key: gkr-live-7f8a9b1c2d3e4f5a...
   ```
4. **Query-параметр в URL**:
   ```http
   POST /v1beta/models/gemini-3.1-flash-lite:generateContent?key=gkr-live-7f8a9b1c2d3e4f5a...
   ```

---

## 5. Управление очередью ожидания (Wait Timeout)

Если все исходящие ключи Google одновременно исчерпали скользящее окно RPM (например, 15 запросов в минуту) или временно находятся в кулдауне 429, роутер **не возвращает ошибку немедленно**, а может поставить запрос в очередь до освобождения ключа.

### Как передать таймаут ожидания:
- **Через HTTP-заголовок**: `X-Wait-Timeout: <секунды>`
- **Через Query-параметр**: `?wait_timeout=<секунды>`
- **В Playground**: поле ввода `Queue Wait (sec)`

### Значения таймаута:
- `0` (по умолчанию) — не ждать; если ни один ключ не готов прямо сейчас, сразу вернуть ошибку.
- `> 0` (например, `60` или `3600`) — ожидать в очереди не более указанного количества секунд. Каждые 0.5 сек роутер проверяет освобождение ключей. Как только любой ключ готов — запрос немедленно выполняется.
- `-1` — **ожидание до 24 часов (86,400 сек)**. Специальный режим для асинхронных воркеров и очередей: задача будет ожидать суточного сброса квот в 10:00 МСК.

> [!TIP]
> Роутер оптимизирован против бессмысленного ожидания: если все ключи в пуле выключены (`is_disabled=True`) или суточный сброс произойдет позже указанного дедлайна, роутер вернет ошибку немедленно, не задерживая процесс.

---

## 6. Примеры подключения микросервисов

### Python: Официальный SDK `google-genai`

Современный официальный клиент Google Gemini SDK для Python (`pip install google-genai`).

#### Обычная генерация (generate_content):
```python
from google import genai
from google.genai import types

# Инициализируем клиент, направив его на наш роутер
client = genai.Client(
    api_key="gkr-live-YOUR_CLIENT_KEY",
    http_options={
        "base_url": "http://gemini-router:8000",  # или http://localhost:8000
        "headers": {
            "X-Wait-Timeout": "120"  # Ждать до 2 минут при пиковых нагрузках
        }
    }
)

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Объясни квантовую суперпозицию в двух предложениях."
)

print(response.text)
```

#### Потоковая генерация (Streaming):
```python
response = client.models.generate_content_stream(
    model="gemini-3.1-flash-lite",
    contents="Напиши стихотворение об искусственном интеллекте."
)

for chunk in response:
    print(chunk.text, end="", flush=True)
print()
```

---

### Python: Библиотека `httpx` (Async / Streaming)

Прямое обращение через асинхронный HTTP-клиент:

```python
import asyncio
import httpx

ROUTER_URL = "http://localhost:8000"
CLIENT_KEY = "gkr-live-YOUR_CLIENT_KEY"

async def generate():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. Обычный запрос с ожиданием до 60 секунд
        response = await client.post(
            f"{ROUTER_URL}/v1beta/models/gemini-3.1-flash-lite:generateContent",
            headers={
                "x-goog-api-key": CLIENT_KEY,
                "X-Wait-Timeout": "60",
                "Content-Type": "application/json"
            },
            json={
                "contents": [{"parts": [{"text": "Привет! Ответь одним словом."}]}]
            }
        )
        print("Ответ:", response.json()["candidates"][0]["content"]["parts"][0]["text"])

        # 2. SSE Стриминг
        async with client.stream(
            "POST",
            f"{ROUTER_URL}/v1beta/models/gemini-3.1-flash-lite:streamGenerateContent?alt=sse",
            headers={"x-goog-api-key": CLIENT_KEY},
            json={"contents": [{"parts": [{"text": "Сосчитай от 1 до 5."}]}]}
        ) as stream_resp:
            async for line in stream_resp.aiter_lines():
                if line.startswith("data: "):
                    print(line)

asyncio.run(generate())
```

---

### Node.js / TypeScript: `@google/genai`

```typescript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: "gkr-live-YOUR_CLIENT_KEY",
  serverUrl: "http://gemini-router:8000"
});

async function main() {
  const response = await ai.models.generateContent({
    model: "gemini-3.1-flash-lite",
    contents: ["Explain event loops in Node.js briefly."]
  });

  console.log(response.text);
}

main();
```

---

### cURL: Прямые HTTP-запросы

#### Запрос генерации:
```bash
curl -X POST "http://localhost:8000/v1beta/models/gemini-3.1-flash-lite:generateContent" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: gkr-live-YOUR_CLIENT_KEY" \
  -H "X-Wait-Timeout: 30" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "Назови 3 факта о планете Марс."}]
      }
    ]
  }'
```

#### Потоковый запрос (Server-Sent Events):
```bash
curl -N -X POST "http://localhost:8000/v1beta/models/gemini-3.1-flash-lite:streamGenerateContent?alt=sse" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: gkr-live-YOUR_CLIENT_KEY" \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "Расскажи короткую историю."}]}
    ]
  }'
```

---

## 7. REST API Справочник

### Прокси-эндпоинты Gemini (Ingress)
- **`POST /v1beta/models/{model}:generateContent`**  
  Основной эндпоинт генерации текста. Совместим со спецификацией Gemini REST API.  
  *Заголовки*: `x-goog-api-key` или `Authorization: Bearer`, опционально `X-Wait-Timeout`.
- **`POST /v1beta/models/{model}:streamGenerateContent`**  
  Потоковый эндпоинт (`text/event-stream`). Принимает `?alt=sse`.
- **`GET /v1beta/models`**  
  Список моделей каталога квот с актуальными лимитами.

### Эндпоинты управления роутером и дашбордом
- **`GET /api/status?model_id={model}`**  
  Сводный статус пула, количество свободных/заблокированных ключей, емкость RPD и обратный отсчет до сброса в 10:00 МСК.
- **`GET /api/keys`**  
  Список всех исходящих ключей Google с их метриками (RPM за 60с, TPM, RPD, оставшийся кулдаун, статус).
- **`POST /api/keys`**  
  Добавление нового исходящего ключа в пул (`label`, `key_value`).
- **`PATCH /api/keys/{key_id}`**  
  Обновление параметров исходящего ключа (`is_disabled`, `reset_cooldown`, `label`).
- **`DELETE /api/keys/{key_id}`**  
  Удаление исходящего ключа из пула.
- **`POST /api/playground`**  
  Вызов тестового запроса из дашборда (поддерживает `dry_run: true/false`, `action: generateContent/streamGenerateContent`, `wait_timeout`).

### Эндпоинты управления клиентскими ключами
- **`GET /api/client-keys`**  
  Список всех клиентских ключей с текущим расходом RPM/RPD и счетчиками вызовов.
- **`POST /api/client-keys`**  
  Создание клиентского ключа (`name`, `rpm_limit`, `rpd_limit`, опционально `custom_key`).  
  *Возвращает токен `raw_key` один раз!*
- **`PATCH /api/client-keys/{id}`**  
  Обновление имени, лимитов или смена статуса (`ACTIVE`, `PAUSED`, `REVOKED`).
- **`DELETE /api/client-keys/{id}`**  
  Удаление клиентского ключа.

---

## 8. Коды ошибок и решение проблем

| HTTP Код | error_code | Причина | Что делать |
|:---|:---|:---|:---|
| **401** | `UNAUTHORIZED` | Клиентский ключ не передан или не существует в базе. | Передайте валидный ключ через `x-goog-api-key` или `Authorization: Bearer`. |
| **403** | `UNAUTHORIZED` | Клиентский ключ находится в статусе `PAUSED` или `REVOKED`. | Возобновите ключ в дашборде (кнопка Resume) или создайте новый. |
| **429** | `CLIENT_RATE_LIMITED` | Превышен персональный лимит RPM или суточный лимит RPD клиента. | Снизьте частоту запросов или увеличьте лимит в настройках клиентского ключа. |
| **429** | `QUEUE_TIMEOUT` | Истекло время ожидания в очереди (`wait_timeout`), а свободный ключ Google так и не появился. | Увеличьте `X-Wait-Timeout`, добавьте больше ключей Google в пул или подождите сброса кулдауна. |
| **429** | `ALL_KEYS_RATE_LIMITED` | Все ключи Google временно находятся на кулдауне (429 от Google) или исчерпали скользящее окно RPM. | Включите очередь ожидания `X-Wait-Timeout: 30`, чтобы роутер дождался выхода ключей из кулдауна. |
| **429** | `ALL_KEYS_EXHAUSTED` | Все ключи Google исчерпали суточный лимит RPD для данной модели. | Лимиты сбросятся в **10:00 МСК**. Добавьте новые ключи в пул или используйте `X-Wait-Timeout: -1`. |
| **502 / 503** | `UPSTREAM_ERROR` | Google вернул сетевую ошибку или 503 Overloaded, и после 3 автоматических повторов здоровых ключей не осталось. | Google временно перегружен. Запрос будет повторен при следующем вызове. |

---

## 9. Полезные команды для обслуживания

```bash
# Проверить размер и статус базы SQLite
ls -lh data/router.db*

# Создать резервную копию базы данных
sqlite3 data/router.db ".backup data/router_backup_$(date +%Y%m%d).db"

# Запуск тестов проверки стабильности
pytest tests/test_router.py

# Проверка типизации и целостности контрактов
python .agents/skills/contract-and-type-guard/scripts/verify_code.py
```
