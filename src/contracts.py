"""Frozen Data Contracts & Schemas for Gemini Key Pool Router.

Single source of truth for backend router, quota manager, and UI dashboard.
Adheres to Pydantic v2, strict type safety, zero Any, and Europe/Moscow timezone rules.
"""

from datetime import datetime, timedelta
from enum import Enum
import secrets
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
        "key_value": "test",
    },
    {
        "label": "Key-Beta-2",
        "key_value": "test",
    },
]


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


def extract_text_from_contents(contents: object) -> str:
    """Safely extract concatenated user message text from Gemini API contents or payload dict.

    Accepts a list of content dicts (e.g. `[{"parts": [{"text": "..."}, ...], "role": "..."}]`)
    or a dict (e.g. `{"contents": ...}` or `{"parts": ...}`), and safely extracts combined text.
    Returns empty string if parsing fails or no text parts are present.
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
            elif isinstance(c, dict):
                items = [c]
            elif isinstance(c, str):
                return c.strip()
        elif "candidates" in contents:
            cands = contents["candidates"]
            if isinstance(cands, list):
                for cand in cands:
                    if isinstance(cand, dict) and "content" in cand:
                        items.append(cand["content"])
        elif "parts" in contents or "text" in contents or "role" in contents:
            items = [contents]
        else:
            return ""
    else:
        return ""

    if not items:
        return ""

    def _extract_parts(item_obj: object) -> list[str]:
        texts: list[str] = []
        if isinstance(item_obj, str) and item_obj.strip():
            texts.append(item_obj.strip())
        elif isinstance(item_obj, dict):
            parts = item_obj.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict):
                        t = p.get("text")
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())
                    elif isinstance(p, str) and p.strip():
                        texts.append(p.strip())
            elif isinstance(parts, str) and parts.strip():
                texts.append(parts.strip())
            elif "text" in item_obj:
                t = item_obj.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        return texts

    user_items: list[object] = []
    other_items: list[object] = []
    for item in items:
        if isinstance(item, dict):
            role = item.get("role")
            if role is None or str(role).lower() in ("user", ""):
                user_items.append(item)
            else:
                other_items.append(item)
        else:
            user_items.append(item)

    target_items = user_items if user_items else other_items

    all_texts: list[str] = []
    for item in target_items:
        all_texts.extend(_extract_parts(item))

    return "\n".join(all_texts).strip()

