"""FastAPI application for Gemini Key Pool Router and Dashboard API.

Exposes Gemini v1beta proxy endpoints and REST management APIs.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Header, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from src.contracts import (
    MAX_QUEUE_WAIT_SECONDS,
    MODEL_QUOTA_CATALOG,
    RESET_TIMEZONE,
    ClientKeyCreateRequest,
    ClientKeyCreatedResponse,
    ClientKeyRecord,
    ClientKeyUpdateRequest,
    KeyCreateRequest,
    KeyRecord,
    KeySelectionResult,
    KeyUpdateRequest,
    PlaygroundRequest,
    PlaygroundResponse,
    PoolStatusResponse,
    RequestLogListResponse,
    RequestLogRecord,
    RouterErrorCode,
    RouterErrorResponse,
    extract_text_from_contents,
    generate_client_api_key,
)
from src.database import DatabaseManager, get_database, init_db
from src.router_engine import (
    RouterException,
    authenticate_and_verify_client,
    close_http_client,
    compute_client_key_record,
    compute_key_record,
    compute_pool_status,
    decrement_in_flight,
    execute_playground_test,
    execute_request,
    execute_streaming_request,
    get_http_client,
    normalize_model_id,
    select_best_key,
    select_best_key_with_wait,
)

logger: logging.Logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to initialize SQLite database, HTTP client, and preloaded keys."""
    db = await init_db()
    await get_http_client()
    yield
    await close_http_client()
    await db.close()


app = FastAPI(
    title="Gemini Key Pool Router",
    version="1.0.0",
    description="High-performance proxy router with Least-Loaded RPD key rotation and stepped 429 cooldown.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path("static")
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.exception_handler(RouterException)
async def router_exception_handler(
    request: Request, exc: RouterException
) -> JSONResponse:
    """Handle RouterException and format response according to RouterErrorResponse contract."""
    headers: dict[str, str] = {}
    if exc.retry_after_seconds is not None:
        headers["Retry-After"] = str(int(exc.retry_after_seconds))
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_error_response().model_dump(),
        headers=headers,
    )


@app.get("/", response_class=HTMLResponse)
async def root_index() -> HTMLResponse:
    """Serve Dashboard HTML or fallback status message."""
    index_file = Path("static/index.html")
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gemini Key Pool Router</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #0b0f19; color: #f1f5f9; padding: 2rem; }
    h1 { color: #38bdf8; }
  </style>
</head>
<body>
  <h1>Gemini Key Pool Router</h1>
  <p>API Server is running in WAL mode.</p>
</body>
</html>"""
    )


@app.get("/api/status", response_model=PoolStatusResponse)
async def get_pool_status(
    model: str = Query("antigravity", description="Target model for quota status evaluation")
) -> PoolStatusResponse:
    """Return aggregate key pool status and MSK reset countdown."""
    db = get_database()
    return await compute_pool_status(db, active_model=model)


@app.get("/api/keys", response_model=list[KeyRecord])
async def list_keys(
    model: str = Query("antigravity", description="Target model for key status calculation")
) -> list[KeyRecord]:
    """Return all keys with real-time dynamic status and usage stats."""
    db = get_database()
    keys = await db.get_all_keys()
    records: list[KeyRecord] = []
    for k in keys:
        rec = await compute_key_record(k, db, active_model=model)
        records.append(rec)
    return records


@app.post("/api/keys", response_model=KeyRecord, status_code=status.HTTP_201_CREATED)
async def create_key(payload: KeyCreateRequest) -> KeyRecord:
    """Add a new API key to the pool."""
    db = get_database()
    try:
        inserted = await db.add_key(payload.key_value, payload.label)
    except Exception as exc:
        raise RouterException(
            status_code=400,
            error_code=RouterErrorCode.VALIDATION_ERROR,
            message=f"Failed to add key: {exc}",
        ) from exc
    return await compute_key_record(inserted, db)


@app.patch("/api/keys/{key_id}", response_model=KeyRecord)
async def modify_key(key_id: int, payload: KeyUpdateRequest) -> KeyRecord:
    """Update API key status, label, or reset cooldown."""
    db = get_database()
    updated = await db.update_key(
        key_id=key_id,
        is_disabled=payload.is_disabled,
        reset_cooldown=payload.reset_cooldown,
        label=payload.label,
    )
    if updated is None:
        raise RouterException(
            status_code=404,
            error_code=RouterErrorCode.VALIDATION_ERROR,
            message=f"API key with ID {key_id} was not found.",
        )
    return await compute_key_record(updated, db)


@app.delete("/api/keys/{key_id}")
async def remove_key(key_id: int) -> dict[str, bool]:
    """Delete an API key from the pool."""
    db = get_database()
    deleted = await db.delete_key(key_id)
    if not deleted:
        raise RouterException(
            status_code=404,
            error_code=RouterErrorCode.VALIDATION_ERROR,
            message=f"API key with ID {key_id} was not found.",
        )
    return {"ok": True}


@app.get("/api/client-keys", response_model=list[ClientKeyRecord])
async def list_client_keys() -> list[ClientKeyRecord]:
    """List all client API keys with real-time usage metrics."""
    db = get_database()
    keys = await db.get_all_client_keys()
    records: list[ClientKeyRecord] = []
    for k in keys:
        rec = await compute_client_key_record(k, db)
        records.append(rec)
    return records


@app.post(
    "/api/client-keys",
    response_model=ClientKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_client_key_endpoint(
    payload: ClientKeyCreateRequest,
) -> ClientKeyCreatedResponse:
    """Create a new client API key."""
    db = get_database()
    raw_key = (
        payload.custom_key.strip()
        if payload.custom_key
        else generate_client_api_key()
    )
    try:
        inserted = await db.create_client_key(
            name=payload.name,
            key_value=raw_key,
            rpm_limit=payload.rpm_limit,
            rpd_limit=payload.rpd_limit,
        )
    except Exception as exc:
        raise RouterException(
            status_code=400,
            error_code=RouterErrorCode.VALIDATION_ERROR,
            message=f"Failed to create client API key: {exc}",
        ) from exc

    key_record = await compute_client_key_record(inserted, db)
    return ClientKeyCreatedResponse(key_record=key_record, raw_key=raw_key)


@app.patch("/api/client-keys/{key_id}", response_model=ClientKeyRecord)
async def modify_client_key(
    key_id: int, payload: ClientKeyUpdateRequest
) -> ClientKeyRecord:
    """Update client API key properties, limits, or status."""
    db = get_database()
    status_str = payload.status.value if payload.status is not None else None
    updated = await db.update_client_key(
        client_key_id=key_id,
        name=payload.name,
        rpm_limit=payload.rpm_limit,
        rpd_limit=payload.rpd_limit,
        status=status_str,
    )
    if updated is None:
        raise RouterException(
            status_code=404,
            error_code=RouterErrorCode.VALIDATION_ERROR,
            message=f"Client API key with ID {key_id} was not found.",
        )
    return await compute_client_key_record(updated, db)


@app.delete("/api/client-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_client_key(key_id: int) -> Response:
    """Delete a client API key and its usage events."""
    db = get_database()
    deleted = await db.delete_client_key(key_id)
    if not deleted:
        raise RouterException(
            status_code=404,
            error_code=RouterErrorCode.VALIDATION_ERROR,
            message=f"Client API key with ID {key_id} was not found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/playground", response_model=PlaygroundResponse)
@app.post("/api/playground/test", response_model=PlaygroundResponse)
async def test_playground(payload: PlaygroundRequest) -> PlaygroundResponse:
    """Execute test prompt via dry-run simulation or live upstream call."""
    db = get_database()
    return await execute_playground_test(payload, db)


@app.get("/v1beta/models")
async def list_models() -> dict[str, object]:
    """List supported models matching the Google Gemini models API format."""
    models_list: list[dict[str, object]] = []
    for model_id, limit in MODEL_QUOTA_CATALOG.items():
        models_list.append(
            {
                "name": f"models/{model_id}",
                "version": "001",
                "displayName": model_id,
                "description": (
                    f"Quota: {limit.rpm} RPM, {limit.rpd} RPD, {limit.tpm} TPM ({limit.category})"
                ),
                "inputTokenLimit": 1048576,
                "outputTokenLimit": 8192,
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            }
        )
    return {"models": models_list}


@app.post("/v1beta/models/{model}:{action}", response_model=None)
async def proxy_model_action(
    model: str,
    action: str,
    request: Request,
    dry_run: bool = Query(False, alias="dry_run"),
    x_dry_run: str | None = Header(None, alias="X-Dry-Run"),
    wait_timeout: float = Query(0.0, alias="wait_timeout"),
    x_wait_timeout: str | None = Header(None, alias="X-Wait-Timeout"),
) -> Response:
    """Proxy inference requests to Google Gemini API or simulate in Dry-Run mode."""
    db = get_database()
    normalized = normalize_model_id(model)
    is_dry = dry_run or (x_dry_run is not None and x_dry_run.lower() in ("true", "1"))

    raw_body = await request.body()
    body_dict: dict[str, object] = {}
    if raw_body:
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
            if isinstance(parsed, dict):
                body_dict = parsed
                if "dry_run" in body_dict and bool(body_dict["dry_run"]):
                    is_dry = True
        except json.JSONDecodeError as exc:
            raise RouterException(
                status_code=400,
                error_code=RouterErrorCode.VALIDATION_ERROR,
                message=f"Invalid JSON request body: {exc}",
            ) from exc

    prompt_text = extract_text_from_contents(body_dict.get("contents") if body_dict else None)
    if not prompt_text and body_dict:
        prompt_text = extract_text_from_contents(body_dict)
    if not prompt_text and body_dict:
        prompt_text = str(body_dict)

    client_key: dict[str, object] | None = None
    try:
        client_key = await authenticate_and_verify_client(
            request_headers=dict(request.headers),
            query_params=dict(request.query_params),
            db=db,
        )
    except RouterException as auth_exc:
        cid: int | None = None
        cname: str | None = None
        token = None
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if not token:
            token = (
                request.headers.get("x-goog-api-key")
                or request.headers.get("x-router-key")
                or request.query_params.get("key")
            )
        if token:
            found_key = await db.get_client_key_by_token(token)
            if found_key:
                cid = int(found_key["id"])
                cname = str(found_key["name"])

        now_ts = time_now()
        now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
        await db.record_request_log(
            timestamp=now_ts,
            timestamp_iso=now_iso,
            client_key_id=cid,
            client_key_name=cname,
            google_key_id=None,
            google_key_label=None,
            model_id=normalized,
            action=action,
            prompt_text=prompt_text,
            prompt_full_json=json.dumps(body_dict) if body_dict else None,
            response_text="",
            response_full_json=None,
            tokens_in=0,
            tokens_out=0,
            latency_ms=0.0,
            status_code=auth_exc.status_code,
            error_message=auth_exc.message,
        )
        raise auth_exc

    client_key_id = int(client_key["id"]) if client_key is not None else None
    client_key_name = str(client_key["name"]) if client_key is not None else None

    timeout_val = wait_timeout
    if x_wait_timeout is not None:
        try:
            timeout_val = float(x_wait_timeout)
        except ValueError as exc:
            logger.debug("Failed to parse X-Wait-Timeout header: %s", exc)

    if "wait_timeout" in body_dict and timeout_val == 0.0 and x_wait_timeout is None:
        raw_bt = body_dict["wait_timeout"]
        if isinstance(raw_bt, (int, float)):
            timeout_val = float(raw_bt)
        elif isinstance(raw_bt, str):
            try:
                timeout_val = float(raw_bt)
            except ValueError as exc:
                logger.debug("Failed to parse wait_timeout from body: %s", exc)

    if timeout_val < 0:
        effective_wait_timeout = MAX_QUEUE_WAIT_SECONDS
    else:
        effective_wait_timeout = timeout_val

    if action == "streamGenerateContent":
        if is_dry:
            key_res = await select_best_key_with_wait(
                normalized, db, wait_timeout=effective_wait_timeout, acquire_in_flight=True
            )
            try:
                async def dry_run_sse_stream() -> AsyncGenerator[str, None]:
                    chunk1 = {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": (
                                                f"[Dry-Run Simulation via {key_res.label}] "
                                                f"Successful generation with model {key_res.model_id}"
                                            )
                                        }
                                    ],
                                    "role": "model",
                                },
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk1)}\n\n"
                    chunk2 = {
                        "candidates": [{"finishReason": "STOP", "index": 0}],
                        "usageMetadata": {
                            "promptTokenCount": 15,
                            "candidatesTokenCount": 25,
                            "totalTokenCount": 40,
                        },
                    }
                    yield f"data: {json.dumps(chunk2)}\n\n"

                await db.record_usage(
                    key_id=key_res.key_id,
                    model_id=key_res.model_id,
                    action=action,
                    timestamp=time_now(),
                    tokens_in=15,
                    tokens_out=25,
                    status_code=200,
                    is_dry_run=True,
                )
                if client_key is not None:
                    await db.record_client_usage(
                        client_key_id=int(client_key["id"]),
                        timestamp=time_now(),
                        tokens=40,
                        status_code=200,
                    )
                now_ts = time_now()
                now_iso = datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat()
                sim_text = f"[Dry-Run Simulation via {key_res.label}] Successful generation with model {key_res.model_id}"
                await db.record_request_log(
                    timestamp=now_ts,
                    timestamp_iso=now_iso,
                    client_key_id=client_key_id,
                    client_key_name=client_key_name,
                    google_key_id=key_res.key_id,
                    google_key_label=key_res.label,
                    model_id=normalized,
                    action=action,
                    prompt_text=prompt_text,
                    prompt_full_json=json.dumps(body_dict) if body_dict else None,
                    response_text=sim_text,
                    response_full_json=None,
                    tokens_in=15,
                    tokens_out=25,
                    latency_ms=20.0,
                    status_code=200,
                    error_message=None,
                )
                return StreamingResponse(
                    dry_run_sse_stream(), media_type="text/event-stream"
                )
            finally:
                await decrement_in_flight(key_res.key_id)
        else:
            return StreamingResponse(
                execute_streaming_request(
                    model_id=normalized,
                    action=action,
                    body=body_dict,
                    headers=dict(request.headers),
                    db=db,
                    wait_timeout=effective_wait_timeout,
                    client_key_id=client_key_id,
                    client_key_name=client_key_name,
                ),
                media_type="text/event-stream",
            )

    out_selected_key: list[KeySelectionResult] = []
    result = await execute_request(
        model_id=normalized,
        action=action,
        body=body_dict,
        is_dry_run=is_dry,
        headers=dict(request.headers),
        db=db,
        out_selected_key=out_selected_key,
        wait_timeout=effective_wait_timeout,
        client_key_id=client_key_id,
        client_key_name=client_key_name,
    )
    if client_key is not None:
        usage_meta = result.get("usageMetadata", {}) if isinstance(result, dict) else {}
        tokens = 0
        if isinstance(usage_meta, dict):
            tin = int(usage_meta.get("promptTokenCount", 0))
            tout = int(usage_meta.get("candidatesTokenCount", 0))
            tokens = tin + tout
        await db.record_client_usage(
            client_key_id=int(client_key["id"]),
            timestamp=time_now(),
            tokens=tokens,
            status_code=200,
        )
    resp_headers: dict[str, str] = {}
    if out_selected_key:
        resp_headers["X-Selected-Key-Label"] = out_selected_key[0].label
    return JSONResponse(status_code=200, content=result, headers=resp_headers)


@app.get("/api/logs", response_model=RequestLogListResponse)
async def list_request_logs(
    limit: int = Query(50, ge=1, le=500, description="Max logs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    client_key_id: int | None = Query(None, description="Filter by client key ID"),
    google_key_id: int | None = Query(None, description="Filter by Google key ID"),
    search: str | None = Query(None, description="Search keyword in prompt or response"),
    since_timestamp: float | None = Query(None, description="Fetch logs strictly after this timestamp"),
) -> RequestLogListResponse:
    """Retrieve request logs with filtering and JSON payload parsing."""
    db = get_database()
    raw_records, total_count, latest_timestamp = await db.get_request_logs(
        limit=limit,
        offset=offset,
        client_key_id=client_key_id,
        google_key_id=google_key_id,
        search=search,
        since_timestamp=since_timestamp,
    )

    records: list[RequestLogRecord] = []
    for r in raw_records:
        prompt_full: dict[str, object] | None = None
        raw_p = r.get("prompt_full_json")
        if raw_p and isinstance(raw_p, str):
            try:
                parsed = json.loads(raw_p)
                if isinstance(parsed, dict):
                    prompt_full = parsed
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug("Failed to decode prompt_full_json: %s", exc)

        response_full: dict[str, object] | None = None
        raw_resp = r.get("response_full_json")
        if raw_resp and isinstance(raw_resp, str):
            try:
                parsed = json.loads(raw_resp)
                if isinstance(parsed, dict):
                    response_full = parsed
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug("Failed to decode response_full_json: %s", exc)

        tool_calls_list: list[dict[str, object]] = []
        raw_tc = r.get("tool_calls")
        if isinstance(raw_tc, list):
            for tc in raw_tc:
                if isinstance(tc, dict):
                    tool_calls_list.append(tc)
        elif r.get("tool_calls_json") and isinstance(r.get("tool_calls_json"), str):
            try:
                parsed_tc = json.loads(str(r["tool_calls_json"]))
                if isinstance(parsed_tc, list):
                    for tc in parsed_tc:
                        if isinstance(tc, dict):
                            tool_calls_list.append(tc)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug("Failed to decode tool_calls_json in main: %s", exc)

        records.append(
            RequestLogRecord(
                id=int(r["id"]),
                timestamp=float(r["timestamp"]),
                timestamp_iso=str(r["timestamp_iso"]),
                client_key_id=int(r["client_key_id"]) if r.get("client_key_id") is not None else None,
                client_key_name=str(r["client_key_name"]) if r.get("client_key_name") is not None else None,
                google_key_id=int(r["google_key_id"]) if r.get("google_key_id") is not None else None,
                google_key_label=str(r["google_key_label"]) if r.get("google_key_label") is not None else None,
                model_id=str(r["model_id"]),
                action=str(r["action"]),
                prompt_text=str(r["prompt_text"]),
                prompt_full=prompt_full,
                response_text=str(r["response_text"]) if r.get("response_text") is not None else "",
                response_full=response_full,
                tool_calls=tool_calls_list,
                tokens_in=int(r["tokens_in"]),
                tokens_out=int(r["tokens_out"]),
                latency_ms=float(r["latency_ms"]),
                status_code=int(r["status_code"]),
                error_message=str(r["error_message"]) if r.get("error_message") is not None else None,
            )
        )

    return RequestLogListResponse(
        logs=records,
        total_count=total_count,
        latest_timestamp=latest_timestamp,
    )


@app.delete("/api/logs")
async def clear_request_logs_endpoint() -> dict[str, bool]:
    """Clear all request logs."""
    db = get_database()
    await db.clear_request_logs()
    return {"ok": True}


def time_now() -> float:
    """Helper returning current timestamp."""
    import time
    return time.time()

