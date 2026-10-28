"""Router Engine for Gemini Key Pool.

Implements MSK reset window calculation, dynamic key status determination,
least-loaded RPD key selection, transparent 429 retry with stepped cooldown,
and request proxying / dry-run simulation.
"""

import asyncio
from datetime import datetime, timedelta
from collections.abc import AsyncGenerator
import json
import logging
import random
import time
from zoneinfo import ZoneInfo
import httpx
from src.contracts import (
    COOLDOWN_INITIAL_SECONDS,
    COOLDOWN_SECONDARY_SECONDS,
    MAX_QUEUE_WAIT_SECONDS,
    MODEL_QUOTA_CATALOG,
    RESET_HOUR_MSK,
    RESET_TIMEZONE,
    SLIDING_WINDOW_SECONDS,
    ClientKeyRecord,
    ClientKeyStatus,
    KeyModelUsage,
    KeyRecord,
    KeySelectionResult,
    KeyStatus,
    ModelQuotaLimit,
    PlaygroundRequest,
    PlaygroundResponse,
    PlaygroundStatus,
    PoolStatusResponse,
    RouterErrorCode,
    RouterErrorResponse,
    calculate_next_reset_time,
    calculate_seconds_until_reset,
    extract_text_from_contents,
    extract_tool_calls,
    mask_api_key,
    mask_client_api_key,
)
from src.database import DatabaseManager, get_database

logger: logging.Logger = logging.getLogger(__name__)

COOLDOWN_SOFT_ERROR_SECONDS: float = 5.0

UPSTREAM_MODEL_ALIASES: dict[str, str] = {
    "gemma-4-26b": "gemma-4-26b-a4b-it",
    "gemma-4-31b": "gemma-4-31b-it",
}

_client_lock: asyncio.Lock = asyncio.Lock()
_shared_client: httpx.AsyncClient | None = None

_in_flight_lock: asyncio.Lock = asyncio.Lock()
_in_flight_requests: dict[int, int] = {}

_thought_signature_lock: asyncio.Lock = asyncio.Lock()
_thought_signature_cache: dict[tuple[str, str], str] = {}
_thought_signature_by_name: dict[str, str] = {}
THOUGHT_SIGNATURE_CACHE_LIMIT: int = 500


async def clear_thought_signature_cache() -> None:
    """Clear thought signature caches."""
    async with _thought_signature_lock:
        _thought_signature_cache.clear()
        _thought_signature_by_name.clear()


async def cache_thought_signatures(response_payload: object) -> None:
    """Extract all functionCall parts from Gemini response and cache their thought signatures.

    Caches both by exact key (func_name, json.dumps(args, sort_keys=True)) and by func_name.
    Keeps size limited to 500 entries per LRU policy.
    """
    if response_payload is None:
        return

    payload_obj = response_payload
    if isinstance(payload_obj, str):
        cleaned = payload_obj.strip()
        if cleaned.startswith("data:"):
            cleaned = cleaned[5:].strip()
        try:
            payload_obj = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return

    calls = extract_tool_calls(payload_obj)
    if not calls:
        return

    async with _thought_signature_lock:
        for call in calls:
            sig = str(call.get("thought_signature") or "").strip()
            if not sig:
                continue

            func_name = str(call.get("name") or "").strip()
            if not func_name:
                continue

            raw_args = call.get("args")
            args_dict: dict[str, object] = {}
            if isinstance(raw_args, dict):
                args_dict = dict(raw_args)
            elif isinstance(raw_args, str):
                try:
                    p = json.loads(raw_args)
                    if isinstance(p, dict):
                        args_dict = p
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.debug("Failed to decode functionCall raw args in cache: %s", exc)

            exact_key = (func_name, json.dumps(args_dict, sort_keys=True))

            # Maintain LRU for _thought_signature_cache
            _thought_signature_cache.pop(exact_key, None)
            _thought_signature_cache[exact_key] = sig
            while len(_thought_signature_cache) > THOUGHT_SIGNATURE_CACHE_LIMIT:
                _thought_signature_cache.pop(next(iter(_thought_signature_cache)))

            # Maintain LRU for _thought_signature_by_name
            _thought_signature_by_name.pop(func_name, None)
            _thought_signature_by_name[func_name] = sig
            while len(_thought_signature_by_name) > THOUGHT_SIGNATURE_CACHE_LIMIT:
                _thought_signature_by_name.pop(next(iter(_thought_signature_by_name)))


async def restore_missing_thought_signatures(body: dict[str, object] | None) -> bool:
    """Traverse body['contents'] and restore missing thought_signature on functionCall parts.

    Searches cache first by exact (func_name, json.dumps(args, sort_keys=True)), then by func_name.
    If found, injects part['thought_signature'] = sig and returns True if at least one was restored.
    """
    if not body or not isinstance(body, dict):
        return False

    contents = body.get("contents")
    if not contents:
        return False

    def _collect_parts(item: object) -> list[dict[str, object]]:
        parts_list: list[dict[str, object]] = []
        if isinstance(item, list):
            for elem in item:
                parts_list.extend(_collect_parts(elem))
        elif isinstance(item, dict):
            if "parts" in item and isinstance(item["parts"], list):
                for p in item["parts"]:
                    if isinstance(p, dict):
                        parts_list.append(p)
            for k, v in item.items():
                if k != "parts" and isinstance(v, (dict, list)):
                    parts_list.extend(_collect_parts(v))
        return parts_list

    parts = _collect_parts(contents)
    if not parts:
        return False

    restored = False
    async with _thought_signature_lock:
        for part in parts:
            fc_obj: object | None = None
            if "functionCall" in part:
                fc_obj = part["functionCall"]
            elif "function_call" in part:
                fc_obj = part["function_call"]

            if not isinstance(fc_obj, dict):
                continue

            has_sig = False
            for k in ("thought_signature", "thoughtSignature"):
                v_part = part.get(k)
                if v_part is not None and str(v_part).strip():
                    has_sig = True
                    break
                v_fc = fc_obj.get(k)
                if v_fc is not None and str(v_fc).strip():
                    has_sig = True
                    break

            if has_sig:
                continue

            func_name = str(fc_obj.get("name") or "").strip()
            if not func_name:
                continue

            raw_args = fc_obj.get("args") if "args" in fc_obj else fc_obj.get("arguments")
            args_dict: dict[str, object] = {}
            if isinstance(raw_args, dict):
                args_dict = dict(raw_args)
            elif isinstance(raw_args, str):
                try:
                    p = json.loads(raw_args)
                    if isinstance(p, dict):
                        args_dict = p
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.debug("Failed to decode functionCall raw args in restore: %s", exc)

            exact_key = (func_name, json.dumps(args_dict, sort_keys=True))
            sig = _thought_signature_cache.get(exact_key) or _thought_signature_by_name.get(func_name)
            if sig:
                part["thought_signature"] = sig
                restored = True

    return restored


async def get_http_client() -> httpx.AsyncClient:
    """Return shared singleton httpx.AsyncClient with connection pooling."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        async with _client_lock:
            if _shared_client is None or _shared_client.is_closed:
                limits = httpx.Limits(
                    max_keepalive_connections=50,
                    max_connections=100,
                )
                timeout = httpx.Timeout(
                    connect=5.0,
                    read=45.0,
                    write=10.0,
                    pool=10.0,
                )
                _shared_client = httpx.AsyncClient(
                    limits=limits,
                    timeout=timeout,
                )
    return _shared_client


async def close_http_client() -> None:
    """Close shared singleton httpx.AsyncClient."""
    global _shared_client
    async with _client_lock:
        if _shared_client is not None and not _shared_client.is_closed:
            await _shared_client.aclose()
            _shared_client = None


async def increment_in_flight(key_id: int) -> int:
    """Increment the in-flight request count for an API key."""
    async with _in_flight_lock:
        count = _in_flight_requests.get(key_id, 0) + 1
        _in_flight_requests[key_id] = count
        return count


async def decrement_in_flight(key_id: int) -> int:
    """Decrement the in-flight request count for an API key."""
    async with _in_flight_lock:
        count = _in_flight_requests.get(key_id, 0) - 1
        if count <= 0:
            _in_flight_requests.pop(key_id, None)
            return 0
        _in_flight_requests[key_id] = count
        return count


def get_in_flight_requests() -> dict[int, int]:
    """Get snapshot of current in-flight requests count per key."""
    return dict(_in_flight_requests)


def reset_in_flight_requests() -> None:
    """Reset in-flight request counters (used in testing and maintenance)."""
    _in_flight_requests.clear()


class RouterException(Exception):
    """Exception raised for router errors with HTTP status code and response payload."""

    def __init__(
        self,
        status_code: int,
        error_code: RouterErrorCode,
        message: str,
        details: dict[str, str] | None = None,
        retry_after_seconds: float | None = None,
        selected_key_label: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: int = status_code
        self.error_code: RouterErrorCode = error_code
        self.message: str = message
        self.details: dict[str, str] = details or {}
        self.retry_after_seconds: float | None = retry_after_seconds
        self.selected_key_label: str | None = selected_key_label

    def to_error_response(self) -> RouterErrorResponse:
        return RouterErrorResponse(
            error_code=self.error_code.value,
            message=self.message,
            selected_key_label=self.selected_key_label,
            details=self.details,
            retry_after_seconds=self.retry_after_seconds,
        )


def get_current_msk_window_start(from_time: datetime | None = None) -> float:
    """Calculate the unix timestamp of the start of the current MSK 10:00:00 window."""
    msk_tz = ZoneInfo(RESET_TIMEZONE)
    current = from_time if from_time is not None else datetime.now(msk_tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=msk_tz)
    else:
        current = current.astimezone(msk_tz)

    reset_today = current.replace(hour=RESET_HOUR_MSK, minute=0, second=0, microsecond=0)
    if current >= reset_today:
        window_start = reset_today
    else:
        window_start = reset_today - timedelta(days=1)
    return window_start.timestamp()


def get_seconds_until_next_msk_reset(from_time: datetime | None = None) -> float:
    """Calculate seconds until next 10:00:00 MSK reset."""
    return calculate_seconds_until_reset(from_time)


def normalize_model_id(model_id: str) -> str:
    """Strip 'models/' prefix if present."""
    cleaned = model_id.strip()
    if cleaned.startswith("models/"):
        return cleaned[len("models/"):]
    return cleaned


def get_model_limit(model_id: str) -> ModelQuotaLimit:
    """Look up model in catalog or raise RouterException."""
    normalized = normalize_model_id(model_id)
    limit = MODEL_QUOTA_CATALOG.get(normalized)
    if limit is None:
        raise RouterException(
            status_code=404,
            error_code=RouterErrorCode.MODEL_NOT_FOUND,
            message=f"Model '{model_id}' was not found in catalog.",
        )
    return limit


async def determine_key_status(
    key: dict[str, object],
    model_id: str,
    db: DatabaseManager,
    now: float | None = None,
) -> tuple[KeyStatus, float]:
    """Calculate dynamic status and remaining cooldown seconds for a key."""
    if bool(key["is_disabled"]):
        return KeyStatus.DISABLED, 0.0

    current_time = time.time() if now is None else now
    cooldown_until = float(key["cooldown_until"])
    if cooldown_until > current_time:
        return KeyStatus.COOLDOWN, max(0.0, cooldown_until - current_time)

    normalized = normalize_model_id(model_id)
    limit = MODEL_QUOTA_CATALOG.get(normalized)
    if limit is not None:
        if limit.rpm > 0:
            rpm_used, _ = await db.get_sliding_window_usage(
                int(key["id"]), normalized, SLIDING_WINDOW_SECONDS
            )
            if rpm_used >= limit.rpm:
                return KeyStatus.RATE_LIMITED_RPM, 0.0

        if limit.rpd > 0:
            dt = datetime.fromtimestamp(current_time, tz=ZoneInfo(RESET_TIMEZONE))
            window_start = get_current_msk_window_start(from_time=dt)
            rpd_used = await db.get_rpd_usage(int(key["id"]), normalized, window_start)
            if rpd_used >= limit.rpd:
                return KeyStatus.EXHAUSTED_RPD, 0.0

    return KeyStatus.READY, 0.0


async def select_best_key(
    model_id: str,
    db: DatabaseManager,
    exclude_ids: set[int] | None = None,
    now: float | None = None,
    acquire_in_flight: bool = False,
) -> KeySelectionResult:
    """Select the least-loaded READY key based on in-flight count, total RPD, and round-robin index."""
    limit = get_model_limit(model_id)
    if limit.rpm == 0 and limit.tpm == 0 and limit.rpd == 0:
        raise RouterException(
            status_code=429,
            error_code=RouterErrorCode.ALL_KEYS_EXHAUSTED,
            message=f"Model '{model_id}' has zero quota allocation in Free Tier. Requests are blocked.",
        )

    keys = await db.get_all_keys()
    if not keys:
        raise RouterException(
            status_code=503,
            error_code=RouterErrorCode.NO_KEYS_AVAILABLE,
            message="No API keys configured in pool.",
        )

    all_disabled = all(bool(k["is_disabled"]) for k in keys)
    if all_disabled:
        raise RouterException(
            status_code=503,
            error_code=RouterErrorCode.NO_KEYS_AVAILABLE,
            message="All API keys in pool are disabled.",
        )

    excluded = exclude_ids or set()
    current_time = time.time() if now is None else now
    dt = datetime.fromtimestamp(current_time, tz=ZoneInfo(RESET_TIMEZONE))
    window_start = get_current_msk_window_start(from_time=dt)

    ready_candidates: list[dict[str, object]] = []
    has_exhausted = False
    has_cooldown = False

    for k in keys:
        k_id = int(k["id"])
        if k_id in excluded:
            continue
        status, _ = await determine_key_status(k, limit.model_id, db, now=current_time)
        if status == KeyStatus.READY:
            ready_candidates.append(k)
        elif status == KeyStatus.EXHAUSTED_RPD:
            has_exhausted = True
        elif status in (KeyStatus.COOLDOWN, KeyStatus.RATE_LIMITED_RPM):
            has_cooldown = True

    if not ready_candidates:
        if has_exhausted:
            sec_to_reset = get_seconds_until_next_msk_reset(from_time=dt)
            raise RouterException(
                status_code=429,
                error_code=RouterErrorCode.ALL_KEYS_EXHAUSTED,
                message=f"All available keys have reached daily RPD limit for model '{limit.model_id}'.",
                retry_after_seconds=sec_to_reset,
            )
        if has_cooldown:
            cooldown_values = [
                float(k["cooldown_until"]) - current_time
                for k in keys
                if float(k["cooldown_until"]) > current_time
            ]
            retry_sec = min(cooldown_values) if cooldown_values else 15.0
            raise RouterException(
                status_code=429,
                error_code=RouterErrorCode.ALL_KEYS_RATE_LIMITED,
                message="All available keys are currently rate-limited or in cooldown.",
                retry_after_seconds=max(1.0, retry_sec),
            )
        raise RouterException(
            status_code=503,
            error_code=RouterErrorCode.NO_KEYS_AVAILABLE,
            message="No keys are available for selection.",
        )

    async with _in_flight_lock:
        best_key: dict[str, object] | None = None
        min_sort_key: tuple[int, int, int] | None = None

        for round_robin_index, candidate in enumerate(ready_candidates):
            cid = int(candidate["id"])
            in_flight_count = _in_flight_requests.get(cid, 0)
            total_rpd = await db.get_total_rpd_usage(cid, window_start)
            sort_key = (in_flight_count, total_rpd, round_robin_index)
            if min_sort_key is None or sort_key < min_sort_key:
                min_sort_key = sort_key
                best_key = candidate

        if best_key is None:
            raise RouterException(
                status_code=503,
                error_code=RouterErrorCode.NO_KEYS_AVAILABLE,
                message="Key selection failure.",
            )

        best_id = int(best_key["id"])
        if acquire_in_flight:
            _in_flight_requests[best_id] = _in_flight_requests.get(best_id, 0) + 1

    return KeySelectionResult(
        key_id=best_id,
        label=str(best_key["label"]),
        key_value=str(best_key["key_value"]),
        model_id=limit.model_id,
    )


async def select_best_key_with_wait(
    model_id: str,
    db: DatabaseManager,
    wait_timeout: float = 0.0,
    exclude_ids: set[int] | None = None,
    acquire_in_flight: bool = False,
) -> KeySelectionResult:
    """Select the best available key, optionally waiting in queue if all keys are temporarily busy."""
    if wait_timeout == 0.0:
        return await select_best_key(
            model_id,
            db,
            exclude_ids=exclude_ids,
            acquire_in_flight=acquire_in_flight,
        )

    effective_timeout = (
        MAX_QUEUE_WAIT_SECONDS if wait_timeout < 0 else wait_timeout
    )
    deadline = time.time() + effective_timeout

    while True:
        try:
            return await select_best_key(
                model_id,
                db,
                exclude_ids=exclude_ids,
                acquire_in_flight=acquire_in_flight,
            )
        except RouterException as exc:
            if exc.error_code in (
                RouterErrorCode.MODEL_NOT_FOUND,
                RouterErrorCode.NO_KEYS_AVAILABLE,
            ):
                raise

            now = time.time()
            rem = deadline - now

            if exc.error_code == RouterErrorCode.ALL_KEYS_EXHAUSTED:
                limit = get_model_limit(model_id)
                if limit.rpm == 0 and limit.tpm == 0 and limit.rpd == 0:
                    raise
                seconds_until_reset = calculate_seconds_until_reset()
                if seconds_until_reset > rem:
                    raise

            if rem <= 0:
                raise RouterException(
                    status_code=429,
                    error_code=RouterErrorCode.QUEUE_TIMEOUT,
                    message=(
                        f"Queue wait timeout ({wait_timeout}s) exceeded while waiting "
                        f"for an available key for model '{model_id}'."
                    ),
                ) from exc

            sleep_time = min(0.5, max(0.05, rem))
            await asyncio.sleep(sleep_time)


async def execute_request(
    model_id: str,
    action: str,
    body: dict[str, object] | None,
    is_dry_run: bool,
    headers: dict[str, str],
    db: DatabaseManager,
    out_selected_key: list[KeySelectionResult] | None = None,
    wait_timeout: float = 0.0,
    client_key_id: int | None = None,
    client_key_name: str | None = None,
) -> dict[str, object]:
    """Execute API request either in dry-run mode or proxied to Google Upstream API with retries."""
    start_time = time.perf_counter()
    limit = get_model_limit(model_id)

    if body and isinstance(body, dict):
        await restore_missing_thought_signatures(body)

    prompt_text = ""
    if body:
        prompt_text = extract_text_from_contents(body.get("contents") if isinstance(body, dict) else None)
        if not prompt_text and isinstance(body, dict):
            prompt_text = extract_text_from_contents(body)
        if not prompt_text and body:
            prompt_text = str(body)

    prompt_full_json = json.dumps(body) if body is not None else None

    if is_dry_run:
        key_res = await select_best_key_with_wait(
            limit.model_id, db, wait_timeout=wait_timeout, acquire_in_flight=True
        )
        try:
            if out_selected_key is not None:
                out_selected_key.append(key_res)
            await asyncio.sleep(0.02)
            tokens_in = 15
            tokens_out = 25
            await db.record_usage(
                key_id=key_res.key_id,
                model_id=key_res.model_id,
                action=action,
                timestamp=time.time(),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                status_code=200,
                is_dry_run=True,
            )
            sim_text = f"[Dry-Run Simulation via {key_res.label}] Successful generation with model {key_res.model_id}"
            resp_data = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": sim_text
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": tokens_in,
                    "candidatesTokenCount": tokens_out,
                    "totalTokenCount": tokens_in + tokens_out,
                },
            }
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            now_ts = time.time()
            now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
            await db.record_request_log(
                timestamp=now_ts,
                timestamp_iso=now_iso,
                client_key_id=client_key_id,
                client_key_name=client_key_name,
                google_key_id=key_res.key_id,
                google_key_label=key_res.label,
                model_id=limit.model_id,
                action=action,
                prompt_text=prompt_text,
                prompt_full_json=prompt_full_json,
                response_text=sim_text,
                response_full_json=json.dumps(resp_data),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                status_code=200,
                error_message=None,
            )
            return resp_data
        finally:
            await decrement_in_flight(key_res.key_id)

    excluded_key_ids: set[int] = set()
    upstream_model = UPSTREAM_MODEL_ALIASES.get(limit.model_id, limit.model_id)
    upstream_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{upstream_model}:{action}"
    )
    client = await get_http_client()

    last_error_status: int = 502
    last_error_message: str = "Request routing could not be completed."
    last_key_id: int | None = None
    last_key_label: str | None = None

    try:
        for attempt in range(3):
            try:
                key_res = await select_best_key_with_wait(
                    limit.model_id,
                    db,
                    wait_timeout=wait_timeout,
                    exclude_ids=excluded_key_ids,
                    acquire_in_flight=True,
                )
                last_key_id = key_res.key_id
                last_key_label = key_res.label
            except RouterException as exc:
                if attempt > 0 and last_error_status in (500, 502, 503, 504):
                    raise RouterException(
                        status_code=last_error_status,
                        error_code=RouterErrorCode.UPSTREAM_ERROR,
                        message=last_error_message,
                        selected_key_label=last_key_label,
                    ) from exc
                raise

            try:
                if out_selected_key is not None:
                    out_selected_key.append(key_res)

                req_headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": key_res.key_value,
                }

                try:
                    await restore_missing_thought_signatures(body)
                    upstream_body: dict[str, object] = dict(body) if isinstance(body, dict) else {}
                    upstream_body.pop("dry_run", None)
                    upstream_body.pop("wait_timeout", None)
                    resp = await client.post(
                        upstream_url,
                        json=upstream_body,
                        headers=req_headers,
                    )
                except httpx.RequestError as exc:
                    logger.warning(
                        f"Upstream network error on key {key_res.key_id} ({key_res.label}): {exc}. "
                        f"Applying soft cooldown {COOLDOWN_SOFT_ERROR_SECONDS}s and retrying..."
                    )
                    await db.set_cooldown(key_res.key_id, COOLDOWN_SOFT_ERROR_SECONDS)
                    await db.record_usage(
                        key_id=key_res.key_id,
                        model_id=key_res.model_id,
                        action=action,
                        timestamp=time.time(),
                        tokens_in=0,
                        tokens_out=0,
                        status_code=502,
                        is_dry_run=False,
                    )
                    excluded_key_ids.add(key_res.key_id)
                    last_error_status = 502
                    last_error_message = f"Network error communicating with upstream: {exc}"
                    last_key_label = key_res.label
                    if attempt < 2:
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                        continue
                    raise RouterException(
                        status_code=502,
                        error_code=RouterErrorCode.UPSTREAM_ERROR,
                        message=last_error_message,
                        selected_key_label=key_res.label,
                    ) from exc

                # 429 Rate Limited
                if resp.status_code == 429:
                    current_key = await db.get_key_by_id(key_res.key_id)
                    consecutive = int(current_key["consecutive_429"]) if current_key else 0
                    cooldown = (
                        COOLDOWN_INITIAL_SECONDS
                        if consecutive == 0
                        else COOLDOWN_SECONDARY_SECONDS
                    )
                    new_consecutive = 1 if consecutive == 0 else 2

                    await db.set_cooldown(key_res.key_id, cooldown, new_consecutive)
                    await db.record_usage(
                        key_id=key_res.key_id,
                        model_id=key_res.model_id,
                        action=action,
                        timestamp=time.time(),
                        tokens_in=0,
                        tokens_out=0,
                        status_code=429,
                        is_dry_run=False,
                    )
                    excluded_key_ids.add(key_res.key_id)
                    last_error_status = 429
                    last_error_message = (
                        "Upstream returned 429 and all retry attempts were exhausted."
                    )
                    last_key_label = key_res.label
                    if attempt < 2:
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                        continue
                    raise RouterException(
                        status_code=429,
                        error_code=RouterErrorCode.ALL_KEYS_RATE_LIMITED,
                        message=last_error_message,
                        retry_after_seconds=cooldown,
                        selected_key_label=key_res.label,
                    )

                # 200 OK
                if resp.status_code == 200:
                    await db.reset_429_counter(key_res.key_id)
                    data = resp.json()
                    await cache_thought_signatures(data)
                    tool_calls = extract_tool_calls(data)
                    usage_meta = (
                        data.get("usageMetadata", {}) if isinstance(data, dict) else {}
                    )
                    tokens_in = (
                        int(usage_meta.get("promptTokenCount", 0))
                        if isinstance(usage_meta, dict)
                        else 0
                    )
                    tokens_out = (
                        int(usage_meta.get("candidatesTokenCount", 0))
                        if isinstance(usage_meta, dict)
                        else 0
                    )

                    await db.record_usage(
                        key_id=key_res.key_id,
                        model_id=key_res.model_id,
                        action=action,
                        timestamp=time.time(),
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        status_code=200,
                        is_dry_run=False,
                    )

                    response_text = ""
                    if isinstance(data, dict):
                        candidates = data.get("candidates")
                        if isinstance(candidates, list) and candidates:
                            first_cand = candidates[0]
                            if isinstance(first_cand, dict):
                                content = first_cand.get("content")
                                if isinstance(content, dict):
                                    parts = content.get("parts")
                                    if isinstance(parts, list):
                                        texts: list[str] = []
                                        for p in parts:
                                            if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                                                texts.append(p["text"])
                                            elif isinstance(p, str):
                                                texts.append(p)
                                        response_text = "".join(texts)
                    if not response_text and isinstance(data, dict) and "candidates" in data:
                        response_text = extract_text_from_contents(data.get("candidates"))

                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    now_ts = time.time()
                    now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
                    await db.record_request_log(
                        timestamp=now_ts,
                        timestamp_iso=now_iso,
                        client_key_id=client_key_id,
                        client_key_name=client_key_name,
                        google_key_id=key_res.key_id,
                        google_key_label=key_res.label,
                        model_id=limit.model_id,
                        action=action,
                        prompt_text=prompt_text,
                        prompt_full_json=prompt_full_json,
                        response_text=response_text,
                        response_full_json=json.dumps(data) if data is not None else None,
                        tool_calls=tool_calls,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        latency_ms=latency_ms,
                        status_code=200,
                        error_message=None,
                    )
                    return data

                # 500, 502, 503, 504 Upstream Server Errors / Overloaded
                if resp.status_code in (500, 502, 503, 504):
                    logger.warning(
                        f"Upstream returned HTTP {resp.status_code} on key {key_res.key_id} ({key_res.label}): {resp.text[:100]}. "
                        f"Applying soft cooldown {COOLDOWN_SOFT_ERROR_SECONDS}s and retrying..."
                    )
                    await db.set_cooldown(key_res.key_id, COOLDOWN_SOFT_ERROR_SECONDS)
                    await db.record_usage(
                        key_id=key_res.key_id,
                        model_id=key_res.model_id,
                        action=action,
                        timestamp=time.time(),
                        tokens_in=0,
                        tokens_out=0,
                        status_code=resp.status_code,
                        is_dry_run=False,
                    )
                    excluded_key_ids.add(key_res.key_id)
                    last_error_status = resp.status_code
                    last_error_message = (
                        f"Upstream returned HTTP {resp.status_code}: {resp.text}"
                    )
                    last_key_label = key_res.label
                    if attempt < 2:
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                        continue
                    raise RouterException(
                        status_code=resp.status_code,
                        error_code=RouterErrorCode.UPSTREAM_ERROR,
                        message=last_error_message,
                        selected_key_label=key_res.label,
                    )

                # Permanent client errors (400, 401, 403, 404, etc.)
                await db.record_usage(
                    key_id=key_res.key_id,
                    model_id=key_res.model_id,
                    action=action,
                    timestamp=time.time(),
                    tokens_in=0,
                    tokens_out=0,
                    status_code=resp.status_code,
                    is_dry_run=False,
                )
                raise RouterException(
                    status_code=resp.status_code,
                    error_code=RouterErrorCode.UPSTREAM_ERROR,
                    message=f"Upstream returned HTTP {resp.status_code}: {resp.text}",
                    selected_key_label=key_res.label,
                )
            finally:
                await decrement_in_flight(key_res.key_id)

        raise RouterException(
            status_code=last_error_status,
            error_code=RouterErrorCode.UPSTREAM_ERROR,
            message=last_error_message,
            selected_key_label=last_key_label,
        )
    except RouterException as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        now_ts = time.time()
        now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
        await db.record_request_log(
            timestamp=now_ts,
            timestamp_iso=now_iso,
            client_key_id=client_key_id,
            client_key_name=client_key_name,
            google_key_id=last_key_id,
            google_key_label=exc.selected_key_label or last_key_label,
            model_id=limit.model_id,
            action=action,
            prompt_text=prompt_text,
            prompt_full_json=prompt_full_json,
            response_text="",
            response_full_json=None,
            tokens_in=0,
            tokens_out=0,
            latency_ms=latency_ms,
            status_code=exc.status_code,
            error_message=exc.message,
        )
        raise


async def execute_streaming_request(
    model_id: str,
    action: str,
    body: dict[str, object] | None,
    headers: dict[str, str],
    db: DatabaseManager,
    wait_timeout: float = 0.0,
    client_key_id: int | None = None,
    client_key_name: str | None = None,
) -> AsyncGenerator[str, None]:
    """Proxy streaming request (streamGenerateContent) to Google Upstream API with retries."""
    start_time = time.perf_counter()
    limit = get_model_limit(model_id)

    if body and isinstance(body, dict):
        await restore_missing_thought_signatures(body)

    prompt_text = ""
    if body:
        prompt_text = extract_text_from_contents(body.get("contents") if isinstance(body, dict) else None)
        if not prompt_text and isinstance(body, dict):
            prompt_text = extract_text_from_contents(body)
        if not prompt_text and body:
            prompt_text = str(body)

    prompt_full_json = json.dumps(body) if body is not None else None

    excluded_key_ids: set[int] = set()
    upstream_model = UPSTREAM_MODEL_ALIASES.get(limit.model_id, limit.model_id)
    clean_action = action.split("?")[0]
    upstream_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{upstream_model}:{clean_action}?alt=sse"
    )
    client = await get_http_client()

    last_error_status: int = 502
    last_error_message: str = "Streaming request routing could not be completed."
    last_key_id: int | None = None
    last_key_label: str | None = None

    try:
        for attempt in range(3):
            try:
                key_res = await select_best_key_with_wait(
                    limit.model_id,
                    db,
                    wait_timeout=wait_timeout,
                    exclude_ids=excluded_key_ids,
                    acquire_in_flight=True,
                )
                last_key_id = key_res.key_id
                last_key_label = key_res.label
            except RouterException as exc:
                if attempt > 0 and last_error_status in (500, 502, 503, 504):
                    raise RouterException(
                        status_code=last_error_status,
                        error_code=RouterErrorCode.UPSTREAM_ERROR,
                        message=last_error_message,
                        selected_key_label=last_key_label,
                    ) from exc
                raise

            req_headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": key_res.key_value,
            }

            should_retry = False
            try:
                try:
                    await restore_missing_thought_signatures(body)
                    upstream_body: dict[str, object] = dict(body) if isinstance(body, dict) else {}
                    upstream_body.pop("dry_run", None)
                    upstream_body.pop("wait_timeout", None)
                    stream_ctx = client.stream(
                        "POST",
                        upstream_url,
                        json=upstream_body,
                        headers=req_headers,
                    )
                    async with stream_ctx as resp:
                        if resp.status_code == 429:
                            error_text = (await resp.aread()).decode("utf-8", errors="replace")
                            current_key = await db.get_key_by_id(key_res.key_id)
                            consecutive = int(current_key["consecutive_429"]) if current_key else 0
                            cooldown = (
                                COOLDOWN_INITIAL_SECONDS
                                if consecutive == 0
                                else COOLDOWN_SECONDARY_SECONDS
                            )
                            new_consecutive = 1 if consecutive == 0 else 2
                            await db.set_cooldown(key_res.key_id, cooldown, new_consecutive)
                            await db.record_usage(
                                key_id=key_res.key_id,
                                model_id=key_res.model_id,
                                action=action,
                                timestamp=time.time(),
                                tokens_in=0,
                                tokens_out=0,
                                status_code=429,
                                is_dry_run=False,
                            )
                            excluded_key_ids.add(key_res.key_id)
                            last_error_status = 429
                            last_error_message = (
                                "Upstream returned 429 and all retry attempts were exhausted."
                            )
                            last_key_label = key_res.label
                            if attempt < 2:
                                should_retry = True
                            else:
                                raise RouterException(
                                    status_code=429,
                                    error_code=RouterErrorCode.ALL_KEYS_RATE_LIMITED,
                                    message=last_error_message,
                                    retry_after_seconds=cooldown,
                                    selected_key_label=key_res.label,
                                )

                        elif resp.status_code in (500, 502, 503, 504):
                            error_text = (await resp.aread()).decode("utf-8", errors="replace")
                            logger.warning(
                                f"Upstream returned HTTP {resp.status_code} on key {key_res.key_id} ({key_res.label}): {error_text[:100]}. "
                                f"Applying soft cooldown {COOLDOWN_SOFT_ERROR_SECONDS}s and retrying..."
                            )
                            await db.set_cooldown(key_res.key_id, COOLDOWN_SOFT_ERROR_SECONDS)
                            await db.record_usage(
                                key_id=key_res.key_id,
                                model_id=key_res.model_id,
                                action=action,
                                timestamp=time.time(),
                                tokens_in=0,
                                tokens_out=0,
                                status_code=resp.status_code,
                                is_dry_run=False,
                            )
                            excluded_key_ids.add(key_res.key_id)
                            last_error_status = resp.status_code
                            last_error_message = (
                                f"Upstream returned HTTP {resp.status_code}: {error_text}"
                            )
                            last_key_label = key_res.label
                            if attempt < 2:
                                should_retry = True
                            else:
                                raise RouterException(
                                    status_code=resp.status_code,
                                    error_code=RouterErrorCode.UPSTREAM_ERROR,
                                    message=last_error_message,
                                    selected_key_label=key_res.label,
                                )

                        elif resp.status_code == 200:
                            await db.reset_429_counter(key_res.key_id)
                            tokens_in = 0
                            tokens_out = 0
                            stream_success = False
                            accumulated_parts: list[str] = []
                            stream_tool_calls: list[dict[str, object]] = []
                            try:
                                async for line in resp.aiter_lines():
                                    yield f"{line}\n"
                                    if line.startswith("data:"):
                                        json_str = line[len("data:"):].strip()
                                        if json_str:
                                            try:
                                                chunk = json.loads(json_str)
                                                if isinstance(chunk, dict):
                                                    await cache_thought_signatures(chunk)
                                                    chunk_tools = extract_tool_calls(chunk)
                                                    if chunk_tools:
                                                        stream_tool_calls.extend(chunk_tools)
                                                    usage = chunk.get("usageMetadata")
                                                    if isinstance(usage, dict):
                                                        tokens_in = int(
                                                            usage.get("promptTokenCount", tokens_in)
                                                        )
                                                        tokens_out = int(
                                                            usage.get("candidatesTokenCount", tokens_out)
                                                        )
                                                    candidates = chunk.get("candidates")
                                                    if isinstance(candidates, list):
                                                        for cand in candidates:
                                                            if isinstance(cand, dict):
                                                                content = cand.get("content")
                                                                if isinstance(content, dict):
                                                                    parts = content.get("parts")
                                                                    if isinstance(parts, list):
                                                                        for p in parts:
                                                                            if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                                                                                accumulated_parts.append(p["text"])
                                                                            elif isinstance(p, str):
                                                                                accumulated_parts.append(p)
                                            except (json.JSONDecodeError, ValueError) as exc:
                                                logger.debug(
                                                    "Failed to parse SSE line JSON: %s", exc
                                                )
                                stream_success = True
                            finally:
                                status_code = 200 if stream_success else 500
                                await db.record_usage(
                                    key_id=key_res.key_id,
                                    model_id=key_res.model_id,
                                    action=action,
                                    timestamp=time.time(),
                                    tokens_in=tokens_in,
                                    tokens_out=tokens_out,
                                    status_code=status_code,
                                    is_dry_run=False,
                                )
                                if client_key_id is not None:
                                    await db.record_client_usage(
                                        client_key_id=client_key_id,
                                        timestamp=time.time(),
                                        tokens=tokens_in + tokens_out,
                                        status_code=status_code,
                                    )
                                latency_ms = (time.perf_counter() - start_time) * 1000.0
                                now_ts = time.time()
                                now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
                                response_text = "".join(accumulated_parts)
                                await db.record_request_log(
                                    timestamp=now_ts,
                                    timestamp_iso=now_iso,
                                    client_key_id=client_key_id,
                                    client_key_name=client_key_name,
                                    google_key_id=key_res.key_id,
                                    google_key_label=key_res.label,
                                    model_id=limit.model_id,
                                    action=action,
                                    prompt_text=prompt_text,
                                    prompt_full_json=prompt_full_json,
                                    response_text=response_text,
                                    response_full_json=None,
                                    tool_calls=stream_tool_calls,
                                    tokens_in=tokens_in,
                                    tokens_out=tokens_out,
                                    latency_ms=latency_ms,
                                    status_code=status_code,
                                    error_message=None if stream_success else "Stream interrupted before completion",
                                )
                            return

                        else:
                            error_text = (await resp.aread()).decode("utf-8", errors="replace")
                            await db.record_usage(
                                key_id=key_res.key_id,
                                model_id=key_res.model_id,
                                action=action,
                                timestamp=time.time(),
                                tokens_in=0,
                                tokens_out=0,
                                status_code=resp.status_code,
                                is_dry_run=False,
                            )
                            raise RouterException(
                                status_code=resp.status_code,
                                error_code=RouterErrorCode.UPSTREAM_ERROR,
                                message=f"Upstream returned HTTP {resp.status_code}: {error_text}",
                                selected_key_label=key_res.label,
                            )

                except httpx.RequestError as exc:
                    logger.warning(
                        f"Upstream network error on key {key_res.key_id} ({key_res.label}): {exc}. "
                        f"Applying soft cooldown {COOLDOWN_SOFT_ERROR_SECONDS}s and retrying..."
                    )
                    await db.set_cooldown(key_res.key_id, COOLDOWN_SOFT_ERROR_SECONDS)
                    await db.record_usage(
                        key_id=key_res.key_id,
                        model_id=key_res.model_id,
                        action=action,
                        timestamp=time.time(),
                        tokens_in=0,
                        tokens_out=0,
                        status_code=502,
                        is_dry_run=False,
                    )
                    excluded_key_ids.add(key_res.key_id)
                    last_error_status = 502
                    last_error_message = f"Network error communicating with upstream: {exc}"
                    last_key_label = key_res.label
                    if attempt < 2:
                        should_retry = True
                    else:
                        raise RouterException(
                            status_code=502,
                            error_code=RouterErrorCode.UPSTREAM_ERROR,
                            message=last_error_message,
                            selected_key_label=key_res.label,
                        ) from exc
            finally:
                await decrement_in_flight(key_res.key_id)

            if should_retry:
                await asyncio.sleep(random.uniform(0.2, 0.5))

        raise RouterException(
            status_code=last_error_status,
            error_code=RouterErrorCode.UPSTREAM_ERROR,
            message=last_error_message,
            selected_key_label=last_key_label,
        )
    except RouterException as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        now_ts = time.time()
        now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
        await db.record_request_log(
            timestamp=now_ts,
            timestamp_iso=now_iso,
            client_key_id=client_key_id,
            client_key_name=client_key_name,
            google_key_id=last_key_id,
            google_key_label=exc.selected_key_label or last_key_label,
            model_id=limit.model_id,
            action=action,
            prompt_text=prompt_text,
            prompt_full_json=prompt_full_json,
            response_text="",
            response_full_json=None,
            tokens_in=0,
            tokens_out=0,
            latency_ms=latency_ms,
            status_code=exc.status_code,
            error_message=exc.message,
        )
        raise


async def execute_playground_test(
    request: PlaygroundRequest, db: DatabaseManager
) -> PlaygroundResponse:
    """Execute test prompt from dashboard playground."""
    start_time = time.perf_counter()
    limit = get_model_limit(request.model_id)

    if request.dry_run:
        key_res = await select_best_key_with_wait(
            limit.model_id, db, wait_timeout=request.wait_timeout, acquire_in_flight=True
        )
        try:
            await asyncio.sleep(0.02)
            tokens_in = 15
            tokens_out = 25
            await db.record_usage(
                key_id=key_res.key_id,
                model_id=key_res.model_id,
                action=request.action,
                timestamp=time.time(),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                status_code=200,
                is_dry_run=True,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return PlaygroundResponse(
                selected_key_id=key_res.key_id,
                selected_key_label=key_res.label,
                status=PlaygroundStatus.DRY_RUN_SUCCESS.value,
                latency_ms=round(latency_ms, 2),
                response_text=f"[Dry-Run Simulation via {key_res.label}] Successful generation with model {key_res.model_id}",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                dry_run=True,
            )
        finally:
            await decrement_in_flight(key_res.key_id)

    if request.action == "streamGenerateContent":
        key_res = await select_best_key_with_wait(
            limit.model_id, db, wait_timeout=request.wait_timeout, acquire_in_flight=True
        )
        try:
            upstream_model = UPSTREAM_MODEL_ALIASES.get(limit.model_id, limit.model_id)
            upstream_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{upstream_model}:streamGenerateContent?alt=sse"
            )
            client = await get_http_client()
            req_headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": key_res.key_value,
            }
            body = {
                "contents": [
                    {
                        "parts": [{"text": request.prompt}],
                        "role": "user",
                    }
                ]
            }
            async with client.stream(
                "POST", upstream_url, json=body, headers=req_headers
            ) as resp:
                if resp.status_code != 200:
                    err_text = (await resp.aread()).decode("utf-8", errors="replace")
                    if resp.status_code == 429:
                        current_key = await db.get_key_by_id(key_res.key_id)
                        consecutive = int(current_key["consecutive_429"]) if current_key else 0
                        cooldown = (
                            COOLDOWN_INITIAL_SECONDS
                            if consecutive == 0
                            else COOLDOWN_SECONDARY_SECONDS
                        )
                        await db.set_cooldown(key_res.key_id, cooldown, 1 if consecutive == 0 else 2)
                        await db.record_usage(
                            key_id=key_res.key_id,
                            model_id=key_res.model_id,
                            action=request.action,
                            timestamp=time.time(),
                            tokens_in=0,
                            tokens_out=0,
                            status_code=429,
                            is_dry_run=False,
                        )
                        raise RouterException(
                            status_code=429,
                            error_code=RouterErrorCode.ALL_KEYS_RATE_LIMITED,
                            message="Upstream returned 429 during playground stream test.",
                            retry_after_seconds=cooldown,
                            selected_key_label=key_res.label,
                        )
                    await db.record_usage(
                        key_id=key_res.key_id,
                        model_id=key_res.model_id,
                        action=request.action,
                        timestamp=time.time(),
                        tokens_in=0,
                        tokens_out=0,
                        status_code=resp.status_code,
                        is_dry_run=False,
                    )
                    raise RouterException(
                        status_code=resp.status_code,
                        error_code=RouterErrorCode.UPSTREAM_ERROR,
                        message=f"Upstream returned HTTP {resp.status_code}: {err_text}",
                        selected_key_label=key_res.label,
                    )

                await db.reset_429_counter(key_res.key_id)
                text_parts: list[str] = []
                tokens_in = 0
                tokens_out = 0
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        json_str = line[len("data:"):].strip()
                        if json_str:
                            try:
                                chunk = json.loads(json_str)
                                if isinstance(chunk, dict):
                                    cands = chunk.get("candidates")
                                    if isinstance(cands, list) and len(cands) > 0:
                                        c0 = cands[0]
                                        if isinstance(c0, dict):
                                            cnt = c0.get("content")
                                            if isinstance(cnt, dict):
                                                pts = cnt.get("parts")
                                                if isinstance(pts, list):
                                                    for pt in pts:
                                                        if isinstance(pt, dict) and "text" in pt:
                                                            text_parts.append(str(pt["text"]))
                                    usage = chunk.get("usageMetadata")
                                    if isinstance(usage, dict):
                                        tokens_in = int(usage.get("promptTokenCount", tokens_in))
                                        tokens_out = int(usage.get("candidatesTokenCount", tokens_out))
                            except (json.JSONDecodeError, ValueError) as exc:
                                logger.debug("Failed to parse SSE line in playground: %s", exc)

                await db.record_usage(
                    key_id=key_res.key_id,
                    model_id=key_res.model_id,
                    action=request.action,
                    timestamp=time.time(),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    status_code=200,
                    is_dry_run=False,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return PlaygroundResponse(
                    selected_key_id=key_res.key_id,
                    selected_key_label=key_res.label,
                    status=PlaygroundStatus.SUCCESS.value,
                    latency_ms=round(latency_ms, 2),
                    response_text="".join(text_parts),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    dry_run=False,
                )
        finally:
            await decrement_in_flight(key_res.key_id)

    body = {
        "contents": [
            {
                "parts": [{"text": request.prompt}],
                "role": "user",
            }
        ]
    }
    selected_keys: list[KeySelectionResult] = []
    resp_data = await execute_request(
        model_id=limit.model_id,
        action=request.action,
        body=body,
        is_dry_run=False,
        headers={},
        db=db,
        out_selected_key=selected_keys,
        wait_timeout=request.wait_timeout,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    selected_key_id = selected_keys[-1].key_id if selected_keys else 0
    selected_key_label = selected_keys[-1].label if selected_keys else "unknown"

    response_text = ""
    candidates = resp_data.get("candidates", [])
    if isinstance(candidates, list) and len(candidates) > 0:
        first = candidates[0]
        if isinstance(first, dict):
            content = first.get("content", {})
            if isinstance(content, dict):
                parts = content.get("parts", [])
                if isinstance(parts, list) and len(parts) > 0:
                    response_text = str(parts[0].get("text", ""))

    usage_meta = resp_data.get("usageMetadata", {})
    tokens_in = (
        int(usage_meta.get("promptTokenCount", 0))
        if isinstance(usage_meta, dict)
        else 0
    )
    tokens_out = (
        int(usage_meta.get("candidatesTokenCount", 0))
        if isinstance(usage_meta, dict)
        else 0
    )

    return PlaygroundResponse(
        selected_key_id=selected_key_id,
        selected_key_label=selected_key_label,
        status=PlaygroundStatus.SUCCESS.value,
        latency_ms=round(latency_ms, 2),
        response_text=response_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        dry_run=False,
    )


async def compute_key_record(
    key: dict[str, object],
    db: DatabaseManager,
    active_model: str = "antigravity",
    now: float | None = None,
) -> KeyRecord:
    """Construct full KeyRecord with model usage metrics for dashboard."""
    key_id = int(key["id"])
    current_time = time.time() if now is None else now
    dt = datetime.fromtimestamp(current_time, tz=ZoneInfo(RESET_TIMEZONE))
    window_start = get_current_msk_window_start(from_time=dt)

    status, cooldown_rem = await determine_key_status(
        key, active_model, db, now=current_time
    )
    rpd_total = await db.get_total_rpd_usage(key_id, window_start)

    models_usage: dict[str, KeyModelUsage] = {}
    for m_id, limit in MODEL_QUOTA_CATALOG.items():
        rpm_u, tpm_u = await db.get_sliding_window_usage(
            key_id, m_id, SLIDING_WINDOW_SECONDS
        )
        rpd_u = await db.get_rpd_usage(key_id, m_id, window_start)
        models_usage[m_id] = KeyModelUsage(
            model_id=m_id,
            rpm_used=rpm_u,
            rpm_limit=limit.rpm,
            tpm_used=tpm_u,
            tpm_limit=limit.tpm,
            rpd_used=rpd_u,
            rpd_limit=limit.rpd,
        )

    return KeyRecord(
        id=key_id,
        masked_key=mask_api_key(str(key["key_value"])),
        label=str(key["label"]),
        status=status,
        consecutive_429=int(key["consecutive_429"]),
        cooldown_remaining_sec=round(cooldown_rem, 1),
        rpd_used_total=rpd_total,
        is_disabled=bool(key["is_disabled"]),
        created_at=str(key["created_at"]),
        updated_at=str(key["updated_at"]),
        models_usage=models_usage,
    )


async def compute_pool_status(
    db: DatabaseManager,
    active_model: str = "antigravity",
    now: float | None = None,
) -> PoolStatusResponse:
    """Calculate aggregate pool status response."""
    keys = await db.get_all_keys()
    total_keys = len(keys)
    current_time = time.time() if now is None else now
    dt = datetime.fromtimestamp(current_time, tz=ZoneInfo(RESET_TIMEZONE))
    window_start = get_current_msk_window_start(from_time=dt)

    normalized = normalize_model_id(active_model)
    limit = MODEL_QUOTA_CATALOG.get(normalized)
    active_model_rpd_limit = limit.rpd if limit else 100

    ready_keys = 0
    cooldown_keys = 0
    exhausted_keys = 0
    disabled_keys = 0
    total_rpd_used = 0

    for k in keys:
        key_id = int(k["id"])
        total_rpd_used += await db.get_total_rpd_usage(key_id, window_start)
        status, _ = await determine_key_status(k, active_model, db, now=current_time)
        if status == KeyStatus.DISABLED:
            disabled_keys += 1
        elif status == KeyStatus.COOLDOWN:
            cooldown_keys += 1
        elif status in (KeyStatus.EXHAUSTED_RPD, KeyStatus.RATE_LIMITED_RPM):
            exhausted_keys += 1
        else:
            ready_keys += 1

    next_reset_dt = calculate_next_reset_time(dt)
    seconds_remaining = calculate_seconds_until_reset(dt)

    return PoolStatusResponse(
        total_keys=total_keys,
        ready_keys=ready_keys,
        cooldown_keys=cooldown_keys,
        exhausted_keys=exhausted_keys,
        disabled_keys=disabled_keys,
        next_reset_iso=next_reset_dt.isoformat(),
        next_reset_seconds_remaining=round(seconds_remaining, 1),
        total_rpd_capacity=total_keys * active_model_rpd_limit,
        total_rpd_used=total_rpd_used,
        active_model=active_model,
    )


async def authenticate_and_verify_client(
    request_headers: dict[str, str],
    query_params: dict[str, str],
    db: DatabaseManager,
    now: float | None = None,
) -> dict[str, object] | None:
    """Validate client API key and check RPM / RPD limits.

    If no client keys are configured in the database, returns None (open access).
    Otherwise authenticates via Bearer, x-goog-api-key, X-Router-Key, or ?key=.
    """
    count = await db.count_client_keys()
    if count == 0:
        return None

    lower_headers = {str(k).lower(): str(v) for k, v in request_headers.items()}
    token: str | None = None

    auth_header = lower_headers.get("authorization")
    if auth_header:
        parts = auth_header.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()

    if not token:
        val = lower_headers.get("x-goog-api-key")
        if val and val.strip():
            token = val.strip()

    if not token:
        val = lower_headers.get("x-router-key")
        if val and val.strip():
            token = val.strip()

    if not token:
        val = query_params.get("key")
        if val and val.strip():
            token = val.strip()

    if not token:
        raise RouterException(
            status_code=401,
            error_code=RouterErrorCode.UNAUTHORIZED,
            message="Missing Client API Key. Pass via Authorization: Bearer, x-goog-api-key, X-Router-Key or ?key=",
        )

    client_key = await db.get_client_key_by_token(token)
    if client_key is None:
        raise RouterException(
            status_code=401,
            error_code=RouterErrorCode.UNAUTHORIZED,
            message="Invalid Client API Key.",
        )

    status_str = str(client_key["status"]).upper()
    if status_str != "ACTIVE":
        raise RouterException(
            status_code=403,
            error_code=RouterErrorCode.UNAUTHORIZED,
            message=f"Client API Key is {client_key['status'].lower()}. Access denied.",
        )

    rpm_limit = int(client_key["rpm_limit"])
    rpd_limit = int(client_key["rpd_limit"])

    current_time = time.time() if now is None else now
    dt = datetime.fromtimestamp(current_time, tz=ZoneInfo(RESET_TIMEZONE))
    window_start_ts = get_current_msk_window_start(from_time=dt)
    window_start_dt = datetime.fromtimestamp(window_start_ts, tz=ZoneInfo(RESET_TIMEZONE))
    window_start_msk_iso = window_start_dt.isoformat()

    rpm_used, rpd_used = await db.get_client_usage(
        int(client_key["id"]), current_time, window_start_msk_iso
    )

    if rpm_limit > 0 and rpm_used >= rpm_limit:
        raise RouterException(
            status_code=429,
            error_code=RouterErrorCode.CLIENT_RATE_LIMITED,
            message=f"Client API Key RPM limit ({rpm_limit}) exceeded.",
        )

    if rpd_limit > 0 and rpd_used >= rpd_limit:
        raise RouterException(
            status_code=429,
            error_code=RouterErrorCode.CLIENT_RATE_LIMITED,
            message=f"Client API Key daily RPD limit ({rpd_limit}) exceeded.",
        )

    return client_key


async def compute_client_key_record(
    client_key: dict[str, object],
    db: DatabaseManager,
    now: float | None = None,
) -> ClientKeyRecord:
    """Enrich client key with real-time usage metrics into ClientKeyRecord contract."""
    kid = int(client_key["id"])
    current_time = time.time() if now is None else now
    dt = datetime.fromtimestamp(current_time, tz=ZoneInfo(RESET_TIMEZONE))
    window_start_ts = get_current_msk_window_start(from_time=dt)
    window_start_dt = datetime.fromtimestamp(window_start_ts, tz=ZoneInfo(RESET_TIMEZONE))
    window_start_msk_iso = window_start_dt.isoformat()

    rpm_u, rpd_u = await db.get_client_usage(kid, current_time, window_start_msk_iso)

    return ClientKeyRecord(
        id=kid,
        name=str(client_key["name"]),
        key_prefix=str(client_key["key_prefix"]),
        rpm_limit=int(client_key["rpm_limit"]),
        rpd_limit=int(client_key["rpd_limit"]),
        status=ClientKeyStatus(str(client_key["status"])),
        total_requests=int(client_key["total_requests"]),
        total_tokens=int(client_key["total_tokens"]),
        rpm_current=rpm_u,
        rpd_current=rpd_u,
        created_at=str(client_key["created_at"]),
        updated_at=str(client_key["updated_at"]),
        last_used_at=str(client_key["last_used_at"]) if client_key["last_used_at"] is not None else None,
    )

