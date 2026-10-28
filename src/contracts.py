"""Frozen Data Contracts & Schemas for Gemini Key Pool Router.

Single source of truth for backend router, quota manager, and UI dashboard.
Adheres to Pydantic v2, strict type safety, zero Any, and Europe/Moscow timezone rules.
"""

from datetime import datetime, timedelta
from enum import Enum
import json
import secrets
from typing import Literal
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field


class KeyStatus(str, Enum):
    READY = "READY"
    COOLDOWN = "COOLDOWN"
    RATE_LIMITED_RPM = "RATE_LIMITED_RPM"
    EXHAUSTED_RPD = "EXHAUSTED_RPD"
    DISABLED = "DISABLED"


class ClientKeyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    REVOKED = "REVOKED"


class PlaygroundStatus(str, Enum):
    SUCCESS = "SUCCESS"
    DRY_RUN_SUCCESS = "DRY_RUN_SUCCESS"
    RATE_LIMITED = "RATE_LIMITED"
    EXHAUSTED = "EXHAUSTED"
    ERROR = "ERROR"


class RouterErrorCode(str, Enum):
    NO_KEYS_AVAILABLE = "NO_KEYS_AVAILABLE"
    ALL_KEYS_EXHAUSTED = "ALL_KEYS_EXHAUSTED"
    ALL_KEYS_RATE_LIMITED = "ALL_KEYS_RATE_LIMITED"
    QUEUE_TIMEOUT = "QUEUE_TIMEOUT"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    CLIENT_RATE_LIMITED = "CLIENT_RATE_LIMITED"


RESET_HOUR_MSK: int = 10
RESET_TIMEZONE: str = "Europe/Moscow"
COOLDOWN_INITIAL_SECONDS: float = 15.0
COOLDOWN_SECONDARY_SECONDS: float = 60.0
SLIDING_WINDOW_SECONDS: float = 60.0
MAX_QUEUE_WAIT_SECONDS: float = 86400.0

INITIAL_PRELOADED_KEYS: list[dict[str, str]] = [
    {
        "label": "Key-Alpha-1",
        "key_value": "test-key-alpha-1-mock",
    },
    {
        "label": "Key-Beta-2",
        "key_value": "test-key-beta-2-mock",
    },
]

KNOWN_KEY_SUFFIX_LABELS: dict[str, str] = {
    "fQiVQ": "Mama",
    "HACVg": "Nikitaloskutov",
    "EDTLw": "My5",
    "SsIgw": "my6",
    "Td6lg": "artem1",
    "OYp3g": "artem2",
    "mVqdA": "artem4",
    "vitGQ": "artem3",
    "wgNWw": "artem5",
    "NWZgA": "Key-Alpha-1",
    "kySHQ": "Key-Beta-2",
    "VWP_k": "Key-Gamma-3",
    "C5Eus": "Key-Delta-4",
}


def resolve_key_label(raw_key: str, fallback_prefix: str = "Key", next_index: int = 1) -> str:
    """Determine label for an API key by matching known suffix or generating fallback."""
    cleaned = raw_key.strip()
    for suffix, label in KNOWN_KEY_SUFFIX_LABELS.items():
        if cleaned.endswith(suffix):
            return label
    return f"{fallback_prefix}-{next_index}"


class ModelQuotaLimit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(description="Exact model identifier for Gemini API")
    category: str = Field(description="Spec category")
    rpm: int = Field(ge=0, description="Requests per minute limit")
    tpm: int = Field(ge=0, description="Tokens per minute limit")
    rpd: int = Field(ge=0, description="Requests per day limit")


MODEL_QUOTA_CATALOG: dict[str, ModelQuotaLimit] = {
    "antigravity": ModelQuotaLimit(
        model_id="antigravity",
        category="Agents",
        rpm=60,
        tpm=100000,
        rpd=100,
    ),
    "gemini-2.5-flash": ModelQuotaLimit(
        model_id="gemini-2.5-flash",
        category="Text-out models",
        rpm=5,
        tpm=250000,
        rpd=20,
    ),
    "gemini-2.5-flash-lite": ModelQuotaLimit(
        model_id="gemini-2.5-flash-lite",
        category="Text-out models",
        rpm=10,
        tpm=250000,
        rpd=20,
    ),
    "gemini-3-flash": ModelQuotaLimit(
        model_id="gemini-3-flash",
        category="Text-out models",
        rpm=5,
        tpm=250000,
        rpd=20,
    ),
    "gemini-3.1-flash-lite": ModelQuotaLimit(
        model_id="gemini-3.1-flash-lite",
        category="Text-out models",
        rpm=15,
        tpm=250000,
        rpd=500,
    ),
    "gemini-3.5-flash-lite": ModelQuotaLimit(
        model_id="gemini-3.5-flash-lite",
        category="Text-out models",
        rpm=15,
        tpm=250000,
        rpd=500,
    ),
    "gemini-3.6-flash": ModelQuotaLimit(
        model_id="gemini-3.6-flash",
        category="Text-out models",
        rpm=5,
        tpm=250000,
        rpd=20,
    ),
    "gemini-3.7-flash": ModelQuotaLimit(
        model_id="gemini-3.7-flash",
        category="Text-out models",
        rpm=5,
        tpm=250000,
        rpd=20,
    ),
    "gemini-3.8-flash": ModelQuotaLimit(
        model_id="gemini-3.8-flash",
        category="Text-out models",
        rpm=5,
        tpm=250000,
        rpd=20,
    ),
    "gemini-2.5-flash-tts": ModelQuotaLimit(
        model_id="gemini-2.5-flash-tts",
        category="Multi-modal TTS",
        rpm=3,
        tpm=10000,
        rpd=10,
    ),
    "gemini-3.1-flash-tts": ModelQuotaLimit(
        model_id="gemini-3.1-flash-tts",
        category="Multi-modal TTS",
        rpm=3,
        tpm=10000,
        rpd=10,
    ),
    "gemini-3.8-flash-tts": ModelQuotaLimit(
        model_id="gemini-3.8-flash-tts",
        category="Multi-modal TTS",
        rpm=3,
        tpm=10000,
        rpd=10,
    ),
    "gemini-3.8-flash-lite-tts": ModelQuotaLimit(
        model_id="gemini-3.8-flash-lite-tts",
        category="Multi-modal TTS",
        rpm=3,
        tpm=10000,
        rpd=10,
    ),
    "gemini-3.5-transcribe": ModelQuotaLimit(
        model_id="gemini-3.5-transcribe",
        category="Live API",
        rpm=3,
        tpm=10000,
        rpd=25,
    ),
    "gemini-robotics-er-2-preview": ModelQuotaLimit(
        model_id="gemini-robotics-er-2-preview",
        category="Other models",
        rpm=5,
        tpm=250000,
        rpd=20,
    ),
    "gemma-4-26b": ModelQuotaLimit(
        model_id="gemma-4-26b",
        category="Other models",
        rpm=30,
        tpm=16000,
        rpd=14400,
    ),
    "gemma-4-26b-a4b-it": ModelQuotaLimit(
        model_id="gemma-4-26b-a4b-it",
        category="Other models",
        rpm=30,
        tpm=16000,
        rpd=14400,
    ),
    "gemma-4-31b": ModelQuotaLimit(
        model_id="gemma-4-31b",
        category="Other models",
        rpm=30,
        tpm=16000,
        rpd=14400,
    ),
    "gemma-4-31b-it": ModelQuotaLimit(
        model_id="gemma-4-31b-it",
        category="Other models",
        rpm=30,
        tpm=16000,
        rpd=14400,
    ),
    "gemini-embedding-1": ModelQuotaLimit(
        model_id="gemini-embedding-1",
        category="Other models",
        rpm=100,
        tpm=30000,
        rpd=1000,
    ),
    "gemini-embedding-2": ModelQuotaLimit(
        model_id="gemini-embedding-2",
        category="Other models",
        rpm=100,
        tpm=30000,
        rpd=1000,
    ),
    "gemini-2.5-pro": ModelQuotaLimit(
        model_id="gemini-2.5-pro",
        category="Zero Quota",
        rpm=0,
        tpm=0,
        rpd=0,
    ),
    "gemini-3.1-pro": ModelQuotaLimit(
        model_id="gemini-3.1-pro",
        category="Zero Quota",
        rpm=0,
        tpm=0,
        rpd=0,
    ),
    "gemini-4-argon": ModelQuotaLimit(
        model_id="gemini-4-argon",
        category="Zero Quota",
        rpm=0,
        tpm=0,
        rpd=0,
    ),
    "gemini-2-flash": ModelQuotaLimit(
        model_id="gemini-2-flash",
        category="Zero Quota",
        rpm=0,
        tpm=0,
        rpd=0,
    ),
}


def mask_api_key(key: str) -> str:
    """Mask an API key for safe logging, reporting, and dashboard displays."""
    cleaned = key.strip()
    if len(cleaned) <= 12:
        if len(cleaned) <= 4:
            return "***"
        return f"{cleaned[:2]}...{cleaned[-2:]}"
    return f"{cleaned[:6]}...{cleaned[-5:]}"


def generate_client_api_key() -> str:
    """Generate a secure client API key with 'gkr-live-' prefix."""
    return f"gkr-live-{secrets.token_hex(16)}"


def mask_client_api_key(key: str) -> str:
    """Mask a client API key for safe display and logging (e.g. first 12 chars + '...' + last 4 chars)."""
    cleaned = key.strip()
    if len(cleaned) <= 16:
        if len(cleaned) <= 4:
            return "***"
        return f"{cleaned[:2]}...{cleaned[-2:]}"
    return f"{cleaned[:12]}...{cleaned[-4:]}"


def calculate_next_reset_time(from_time: datetime | None = None) -> datetime:
    """Calculate the next 10:00:00 MSK reset timestamp."""
    msk_tz = ZoneInfo(RESET_TIMEZONE)
    current = from_time if from_time is not None else datetime.now(msk_tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=msk_tz)
    else:
        current = current.astimezone(msk_tz)

    candidate = current.replace(hour=RESET_HOUR_MSK, minute=0, second=0, microsecond=0)
    if current >= candidate:
        candidate = candidate + timedelta(days=1)
    return candidate


def calculate_seconds_until_reset(from_time: datetime | None = None) -> float:
    """Calculate remaining seconds until the next 10:00:00 MSK reset."""
    msk_tz = ZoneInfo(RESET_TIMEZONE)
    current = from_time if from_time is not None else datetime.now(msk_tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=msk_tz)
    else:
        current = current.astimezone(msk_tz)

    next_reset = calculate_next_reset_time(current)
    remaining = (next_reset - current).total_seconds()
    return max(0.0, remaining)


class KeyModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    rpm_used: int = 0
    rpm_limit: int
    tpm_used: int = 0
    tpm_limit: int
    rpd_used: int = 0
    rpd_limit: int


class KeyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    masked_key: str
    label: str
    status: KeyStatus
    consecutive_429: int = 0
    cooldown_remaining_sec: float = 0.0
    rpd_used_total: int = 0
    is_disabled: bool = False
    created_at: str
    updated_at: str
    models_usage: dict[str, KeyModelUsage] = Field(default_factory=dict)


class KeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_value: str = Field(..., min_length=8, description="Raw API key value")
    label: str = Field(..., min_length=1, description="Human-readable label")


class KeyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_disabled: bool | None = None
    reset_cooldown: bool | None = None
    label: str | None = None


KeyBulkItemStatus = Literal["added", "skipped_duplicate", "invalid"]


class KeyBulkItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    masked_key: str
    status: Literal["added", "skipped_duplicate", "invalid"]
    key_id: int | None = None


class KeyBulkCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_keys: str = Field(..., min_length=1, description="Multiline text of API keys or label:key pairs")
    default_prefix: str = Field(default="Key", description="Prefix for auto-naming if no suffix matches")


class KeyBulkCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    added_count: int
    skipped_count: int
    results: list[KeyBulkItemResult]
    keys: list[KeyRecord]


class ClientKeyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    key_prefix: str
    rpm_limit: int = Field(ge=0, description="0 means unlimited RPM")
    rpd_limit: int = Field(ge=0, description="0 means unlimited RPD")
    status: ClientKeyStatus
    total_requests: int = 0
    total_tokens: int = 0
    rpm_current: int = 0
    rpd_current: int = 0
    created_at: str
    updated_at: str
    last_used_at: str | None = None


class ClientKeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)
    rpm_limit: int = Field(default=60, ge=0)
    rpd_limit: int = Field(default=1000, ge=0)
    custom_key: str | None = Field(default=None, min_length=8)


class ClientKeyCreatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_record: ClientKeyRecord
    raw_key: str


class ClientKeyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1)
    rpm_limit: int | None = Field(default=None, ge=0)
    rpd_limit: int | None = Field(default=None, ge=0)
    status: ClientKeyStatus | None = None


class PoolStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_keys: int
    ready_keys: int
    cooldown_keys: int
    exhausted_keys: int
    disabled_keys: int
    next_reset_iso: str
    next_reset_seconds_remaining: float
    total_rpd_capacity: int
    total_rpd_used: int
    active_model: str


class PlaygroundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    prompt: str
    action: str = "generateContent"
    dry_run: bool = True
    wait_timeout: float = Field(default=0.0, ge=-1.0, description="Max seconds to wait in queue for key. -1 means 24h.")


class PlaygroundResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_key_id: int
    selected_key_label: str
    status: str
    latency_ms: float
    response_text: str
    tokens_in: int
    tokens_out: int
    dry_run: bool


class RouterErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_code: str
    message: str
    selected_key_label: str | None = None
    details: dict[str, str] = Field(default_factory=dict)
    retry_after_seconds: float | None = None


class KeySelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_id: int
    label: str
    key_value: str
    model_id: str


class RequestLogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    timestamp: float
    timestamp_iso: str
    client_key_id: int | None = None
    client_key_name: str | None = None
    google_key_id: int | None = None
    google_key_label: str | None = None
    model_id: str
    action: str
    prompt_text: str
    prompt_full: dict[str, object] | None = None
    response_text: str = ""
    response_full: dict[str, object] | None = None
    tool_calls: list[dict[str, object]] = Field(default_factory=list, description="Extracted tool calls if any")
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    status_code: int = 200
    error_message: str | None = None


class RequestLogListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logs: list[RequestLogRecord]
    total_count: int
    latest_timestamp: float = 0.0


def extract_tool_calls(payload: object) -> list[dict[str, object]]:
    """Safely extract function/tool calls from Gemini API payload or contents.

    Returns a list of dicts: `[{"name": "...", "args": {...}, "id": "...", "thought_signature": "..."}]`.
    Accepts candidate responses, request contents, part lists, or single part objects.
    """
    if payload is None:
        return []

    calls: list[dict[str, object]] = []

    def _process_part(part_obj: object) -> None:
        if not isinstance(part_obj, dict):
            return

        fc_obj: object | None = None
        if "functionCall" in part_obj:
            fc_obj = part_obj["functionCall"]
        elif "function_call" in part_obj:
            fc_obj = part_obj["function_call"]
        elif "name" in part_obj and ("args" in part_obj or "arguments" in part_obj):
            fc_obj = part_obj

        if not isinstance(fc_obj, dict):
            return

        name = str(fc_obj.get("name") or "")

        raw_args = fc_obj.get("args") if "args" in fc_obj else fc_obj.get("arguments")
        args_dict: dict[str, object] = {}
        if isinstance(raw_args, dict):
            args_dict = dict(raw_args)
        elif isinstance(raw_args, str):
            try:
                parsed: object = json.loads(raw_args)
                if isinstance(parsed, dict):
                    args_dict = dict(parsed)
                else:
                    args_dict = {"value": parsed}
            except (json.JSONDecodeError, TypeError, ValueError):
                args_dict = {"raw": raw_args}
        elif raw_args is not None:
            args_dict = {"value": raw_args}

        call_id = ""
        for key in ("id", "call_id"):
            val = fc_obj.get(key) or part_obj.get(key)
            if val is not None and str(val).strip():
                call_id = str(val).strip()
                break

        thought_sig = ""
        for key in ("thought_signature", "thoughtSignature"):
            val = fc_obj.get(key) or part_obj.get(key)
            if val is not None and str(val).strip():
                thought_sig = str(val).strip()
                break

        calls.append(
            {
                "name": name,
                "args": args_dict,
                "id": call_id,
                "thought_signature": thought_sig,
            }
        )

    def _traverse(obj: object) -> None:
        if obj is None:
            return
        if isinstance(obj, list):
            for item in obj:
                _traverse(item)
        elif isinstance(obj, dict):
            if "functionCall" in obj or "function_call" in obj:
                _process_part(obj)
            elif "candidates" in obj:
                _traverse(obj["candidates"])
            elif "content" in obj:
                _traverse(obj["content"])
            elif "contents" in obj:
                _traverse(obj["contents"])
            elif "parts" in obj:
                parts = obj["parts"]
                if isinstance(parts, list):
                    for p in parts:
                        _process_part(p)
                elif isinstance(parts, dict):
                    _process_part(parts)
            elif "name" in obj and ("args" in obj or "arguments" in obj):
                _process_part(obj)

    _traverse(payload)
    return calls


def extract_text_from_contents(contents: object) -> str:
    """Safely extract concatenated text, tool calls, and tool responses from Gemini API payload.

    Accepts:
    - List of content dicts (e.g. `[{"parts": [...], "role": "..."}]`)
    - Dict with `"contents"` or `"candidates"`
    - List of candidates (e.g. `[{"content": {"parts": [...]}, "finishReason": "STOP"}]`)
    - Single content or candidate dict
    - Single part dict or plain string

    Formats tool calls as: `🛠️ Tool Call: {name}({json_args})`
    Formats tool responses as: `📦 Tool Response [{name}]: {json_response}`
    Takes into account `finishMessage` when `finishReason == "STOP"`.
    Returns empty string if parsing fails or no content is present.
    """
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents.strip()

    items: list[object] = []
    if isinstance(contents, list):
        items = contents
    elif isinstance(contents, dict):
        if "contents" in contents:
            c = contents["contents"]
            if isinstance(c, list):
                items = c
            elif isinstance(c, (dict, str)):
                items = [c]
        elif "candidates" in contents:
            cands = contents["candidates"]
            if isinstance(cands, list):
                items = cands
            elif isinstance(cands, (dict, str)):
                items = [cands]
        elif (
            "parts" in contents
            or "content" in contents
            or "text" in contents
            or "role" in contents
            or "functionCall" in contents
            or "function_call" in contents
            or "functionResponse" in contents
            or "function_response" in contents
            or "finishReason" in contents
            or "finish_reason" in contents
        ):
            items = [contents]
        else:
            return ""
    else:
        return ""

    if not items:
        return ""

    def _extract_from_part(part_obj: object) -> list[str]:
        part_texts: list[str] = []
        if isinstance(part_obj, str) and part_obj.strip():
            part_texts.append(part_obj.strip())
            return part_texts
        if not isinstance(part_obj, dict):
            return part_texts

        # 1. Text part
        t = part_obj.get("text")
        if isinstance(t, str) and t.strip():
            part_texts.append(t.strip())

        # 2. Tool Call part
        fc: object | None = None
        if "functionCall" in part_obj:
            fc = part_obj["functionCall"]
        elif "function_call" in part_obj:
            fc = part_obj["function_call"]
        elif "name" in part_obj and ("args" in part_obj or "arguments" in part_obj):
            fc = part_obj

        if isinstance(fc, dict):
            name = str(fc.get("name") or "")
            raw_args = fc.get("args") if "args" in fc else fc.get("arguments")
            if raw_args is None:
                json_args = "{}"
            elif isinstance(raw_args, str):
                json_args = raw_args
            elif isinstance(raw_args, (dict, list)):
                json_args = json.dumps(raw_args, ensure_ascii=False)
            else:
                json_args = json.dumps(raw_args, ensure_ascii=False)
            part_texts.append(f"🛠️ Tool Call: {name}({json_args})")

        # 3. Tool Response part
        fr: object | None = None
        if "functionResponse" in part_obj:
            fr = part_obj["functionResponse"]
        elif "function_response" in part_obj:
            fr = part_obj["function_response"]
        elif "name" in part_obj and "response" in part_obj:
            fr = part_obj

        if isinstance(fr, dict):
            name = str(fr.get("name") or "")
            raw_response = fr.get("response")
            if raw_response is None:
                json_response = "{}"
            elif isinstance(raw_response, str):
                json_response = raw_response
            elif isinstance(raw_response, (dict, list)):
                json_response = json.dumps(raw_response, ensure_ascii=False)
            else:
                json_response = json.dumps(raw_response, ensure_ascii=False)
            part_texts.append(f"📦 Tool Response [{name}]: {json_response}")

        return part_texts

    def _extract_from_item(item_obj: object) -> list[str]:
        texts: list[str] = []
        if isinstance(item_obj, str) and item_obj.strip():
            texts.append(item_obj.strip())
            return texts
        if not isinstance(item_obj, dict):
            return texts

        cand_content = item_obj.get("content")
        if isinstance(cand_content, dict):
            texts.extend(_extract_from_item(cand_content))
        elif isinstance(cand_content, list):
            for sub in cand_content:
                texts.extend(_extract_from_item(sub))

        parts = item_obj.get("parts")
        if isinstance(parts, list):
            for p in parts:
                texts.extend(_extract_from_part(p))
        elif isinstance(parts, (dict, str)):
            texts.extend(_extract_from_part(parts))

        if (
            "text" in item_obj
            or "functionCall" in item_obj
            or "function_call" in item_obj
            or "functionResponse" in item_obj
            or "function_response" in item_obj
        ):
            if "parts" not in item_obj and "content" not in item_obj:
                texts.extend(_extract_from_part(item_obj))

        finish_reason = item_obj.get("finishReason") or item_obj.get("finish_reason")
        finish_message = item_obj.get("finishMessage") or item_obj.get("finish_message")
        if finish_reason == "STOP" and isinstance(finish_message, str) and finish_message.strip():
            fm = finish_message.strip()
            if fm not in texts:
                texts.append(fm)

        return texts

    user_items: list[object] = []
    other_items: list[object] = []
    for item in items:
        if isinstance(item, dict):
            role = item.get("role")
            if role is None and isinstance(item.get("content"), dict):
                role = item["content"].get("role")

            if role is None or str(role).lower() in ("user", "function", "tool", ""):
                user_items.append(item)
            else:
                other_items.append(item)
        else:
            user_items.append(item)

    target_items = user_items if user_items else other_items

    all_texts: list[str] = []
    for item in target_items:
        all_texts.extend(_extract_from_item(item))

    if not all_texts and other_items and target_items is not other_items:
        for item in other_items:
            all_texts.extend(_extract_from_item(item))

    return "\n".join(all_texts).strip()

