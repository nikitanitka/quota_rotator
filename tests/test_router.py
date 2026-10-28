"""Comprehensive test suite for Gemini Key Pool Router.

Tests:
1. DB initialization and seeding with Key-Alpha-1 and Key-Beta-2.
2. Least-loaded RPD key rotation strategy.
3. Zero-quota model blocking.
4. MSK 10:00:00 reset window calculation.
5. Stepped cooldown on 429 errors (15s then 60s).
6. CRUD operations for API keys.
7. Playground dry-run execution.
8. Proxy endpoint routing and SSE streaming.
9. Upstream 429 transparent retry.
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
import time
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo
import httpx
import pytest
from src.contracts import (
    INITIAL_PRELOADED_KEYS,
    RESET_TIMEZONE,
    ClientKeyStatus,
    KeyStatus,
    PlaygroundRequest,
    PlaygroundStatus,
    RequestLogListResponse,
    RequestLogRecord,
    RouterErrorCode,
    generate_client_api_key,
    mask_client_api_key,
)
from src.database import DatabaseManager, set_database
from src.main import app
from src.router_engine import (
    UPSTREAM_MODEL_ALIASES,
    RouterException,
    cache_thought_signatures,
    clear_thought_signature_cache,
    close_http_client,
    decrement_in_flight,
    execute_playground_test,
    execute_request,
    execute_streaming_request,
    get_current_msk_window_start,
    get_http_client,
    get_in_flight_requests,
    get_seconds_until_next_msk_reset,
    increment_in_flight,
    reset_in_flight_requests,
    restore_missing_thought_signatures,
    select_best_key,
    select_best_key_with_wait,
)


@pytest.fixture
async def test_db(tmp_path: Path) -> AsyncGenerator[DatabaseManager, None]:
    """Create and initialize an isolated test database in a temporary directory."""
    reset_in_flight_requests()
    db_file = tmp_path / "test_router.db"
    db = DatabaseManager(db_path=str(db_file))
    await db.initialize()
    set_database(db)
    yield db
    await db.close()
    await close_http_client()
    reset_in_flight_requests()


@pytest.fixture
async def async_client(test_db: DatabaseManager) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client connected to the FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_db_init_and_seeding(test_db: DatabaseManager) -> None:
    """1. DB initialization seeds Key-Alpha-1 and Key-Beta-2 properly."""
    keys = await test_db.get_all_keys()
    assert len(keys) == 2

    key1 = keys[0]
    key2 = keys[1]

    assert key1["label"] == INITIAL_PRELOADED_KEYS[0]["label"]
    assert key1["key_value"] == INITIAL_PRELOADED_KEYS[0]["key_value"]
    assert key1["status"] == KeyStatus.READY.value
    assert key1["is_disabled"] is False
    assert key1["consecutive_429"] == 0

    assert key2["label"] == INITIAL_PRELOADED_KEYS[1]["label"]
    assert key2["key_value"] == INITIAL_PRELOADED_KEYS[1]["key_value"]
    assert key2["status"] == KeyStatus.READY.value
    assert key2["is_disabled"] is False
    assert key2["consecutive_429"] == 0


@pytest.mark.asyncio
async def test_least_loaded_rpd_rotation(test_db: DatabaseManager) -> None:
    """2. Key rotation strictly alternates based on least total RPD load."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    k2_id = int(keys[1]["id"])

    # Step 1: Both keys have 0 usage -> Pick k1 (tie-breaker lowest ID)
    res1 = await select_best_key("antigravity", test_db)
    assert res1.key_id == k1_id
    await test_db.record_usage(
        key_id=k1_id,
        model_id="antigravity",
        action="generateContent",
        timestamp=time.time(),
        tokens_in=10,
        tokens_out=20,
        status_code=200,
        is_dry_run=True,
    )

    # Step 2: k1 has 1 request, k2 has 0 -> Pick k2
    res2 = await select_best_key("antigravity", test_db)
    assert res2.key_id == k2_id
    await test_db.record_usage(
        key_id=k2_id,
        model_id="antigravity",
        action="generateContent",
        timestamp=time.time(),
        tokens_in=10,
        tokens_out=20,
        status_code=200,
        is_dry_run=True,
    )

    # Step 3: Both have 1 request -> Pick k1 (lowest ID)
    res3 = await select_best_key("antigravity", test_db)
    assert res3.key_id == k1_id
    await test_db.record_usage(
        key_id=k1_id,
        model_id="antigravity",
        action="generateContent",
        timestamp=time.time(),
        tokens_in=10,
        tokens_out=20,
        status_code=200,
        is_dry_run=True,
    )

    # Step 4: k1 has 2, k2 has 1 -> Pick k2
    res4 = await select_best_key("antigravity", test_db)
    assert res4.key_id == k2_id


@pytest.mark.asyncio
async def test_zero_quota_models_blocking(test_db: DatabaseManager) -> None:
    """3. Models with zero quota (gemini-3.1-pro, gemini-2.5-pro, etc.) are blocked immediately."""
    zero_quota_models = ["gemini-3.1-pro", "gemini-2.5-pro", "gemini-4-argon", "gemini-2-flash"]

    for model_id in zero_quota_models:
        with pytest.raises(RouterException) as exc_info:
            await select_best_key(model_id, test_db)
        assert exc_info.value.status_code == 429
        assert exc_info.value.error_code == RouterErrorCode.ALL_KEYS_EXHAUSTED

        # Test execute_request dry-run also rejects zero-quota models
        with pytest.raises(RouterException) as exc_info_exec:
            await execute_request(
                model_id=model_id,
                action="generateContent",
                body=None,
                is_dry_run=True,
                headers={},
                db=test_db,
            )
        assert exc_info_exec.value.status_code == 429
        assert exc_info_exec.value.error_code == RouterErrorCode.ALL_KEYS_EXHAUSTED


def test_msk_reset_window_calculation() -> None:
    """4. RPD reset window starts at 10:00:00 MSK and handles before/after 10:00 properly."""
    msk = ZoneInfo(RESET_TIMEZONE)

    # Scenario A: Before 10:00 MSK (e.g. 09:15:00 on 2026-10-19)
    before_time = datetime(2026, 10, 19, 9, 15, 0, tzinfo=msk)
    window_start_before = get_current_msk_window_start(from_time=before_time)
    expected_start_before = datetime(2026, 10, 18, 10, 0, 0, tzinfo=msk).timestamp()
    assert window_start_before == expected_start_before

    sec_remaining_before = get_seconds_until_next_msk_reset(from_time=before_time)
    assert sec_remaining_before == 45 * 60  # 45 minutes until 10:00 MSK today

    # Scenario B: Exactly at 10:00:00 MSK
    at_reset_time = datetime(2026, 10, 19, 10, 0, 0, tzinfo=msk)
    window_start_at = get_current_msk_window_start(from_time=at_reset_time)
    expected_start_at = datetime(2026, 10, 19, 10, 0, 0, tzinfo=msk).timestamp()
    assert window_start_at == expected_start_at

    sec_remaining_at = get_seconds_until_next_msk_reset(from_time=at_reset_time)
    assert sec_remaining_at == 24 * 3600  # 24 hours until 10:00 MSK tomorrow

    # Scenario C: After 10:00 MSK (e.g. 14:30:00 on 2026-10-19)
    after_time = datetime(2026, 10, 19, 14, 30, 0, tzinfo=msk)
    window_start_after = get_current_msk_window_start(from_time=after_time)
    expected_start_after = datetime(2026, 10, 19, 10, 0, 0, tzinfo=msk).timestamp()
    assert window_start_after == expected_start_after

    sec_remaining_after = get_seconds_until_next_msk_reset(from_time=after_time)
    assert sec_remaining_after == (19.5 * 3600)  # from 14:30 to 10:00 next day is 19.5 hours


@pytest.mark.asyncio
async def test_stepped_cooldown_on_429(test_db: DatabaseManager) -> None:
    """5. Stepped cooldown: 15s on first 429, 60s on second consecutive 429, reset on success."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])

    # First 429: 15 seconds cooldown, consecutive_429 = 1
    await test_db.set_cooldown(k1_id, cooldown_seconds=15.0, consecutive_429=1)
    k1 = await test_db.get_key_by_id(k1_id)
    assert k1 is not None
    assert k1["status"] == KeyStatus.COOLDOWN.value
    assert k1["consecutive_429"] == 1
    assert float(k1["cooldown_until"]) > time.time()

    # Second 429: 60 seconds cooldown, consecutive_429 = 2
    await test_db.set_cooldown(k1_id, cooldown_seconds=60.0, consecutive_429=2)
    k1 = await test_db.get_key_by_id(k1_id)
    assert k1 is not None
    assert k1["consecutive_429"] == 2
    assert float(k1["cooldown_until"]) > time.time() + 30.0

    # Reset on successful request (200 OK)
    await test_db.reset_429_counter(k1_id)
    k1 = await test_db.get_key_by_id(k1_id)
    assert k1 is not None
    assert k1["consecutive_429"] == 0
    assert k1["cooldown_until"] == 0.0
    assert k1["status"] == KeyStatus.READY.value


@pytest.mark.asyncio
async def test_key_crud_operations(test_db: DatabaseManager) -> None:
    """6. CRUD operations: add key, disable, reset cooldown, update label, delete key."""
    # Add new key
    new_key = await test_db.add_key(
        key_value="AQ.Ab8RN6NewKeyTest_1234567890abcdef",
        label="Key-Gamma-3",
    )
    new_id = int(new_key["id"])
    assert new_key["label"] == "Key-Gamma-3"

    all_keys = await test_db.get_all_keys()
    assert len(all_keys) == 3

    # Pause / disable key
    updated_disable = await test_db.update_key(new_id, is_disabled=True)
    assert updated_disable is not None
    assert updated_disable["is_disabled"] is True
    assert updated_disable["status"] == KeyStatus.DISABLED.value

    # Resume key
    updated_enable = await test_db.update_key(new_id, is_disabled=False)
    assert updated_enable is not None
    assert updated_enable["is_disabled"] is False
    assert updated_enable["status"] == KeyStatus.READY.value

    # Place in cooldown and reset cooldown
    await test_db.set_cooldown(new_id, cooldown_seconds=60.0, consecutive_429=2)
    updated_reset = await test_db.update_key(new_id, reset_cooldown=True)
    assert updated_reset is not None
    assert updated_reset["status"] == KeyStatus.READY.value
    assert updated_reset["consecutive_429"] == 0
    assert updated_reset["cooldown_until"] == 0.0

    # Delete key
    deleted = await test_db.delete_key(new_id)
    assert deleted is True

    remaining = await test_db.get_all_keys()
    assert len(remaining) == 2
    assert all(int(k["id"]) != new_id for k in remaining)


@pytest.mark.asyncio
async def test_playground_dry_run(test_db: DatabaseManager) -> None:
    """7. Playground dry-run returns synthetic generation without calling Google API."""
    req = PlaygroundRequest(
        model_id="antigravity",
        prompt="Tell me a joke",
        dry_run=True,
    )
    resp = await execute_playground_test(req, test_db)
    assert resp.status == PlaygroundStatus.DRY_RUN_SUCCESS.value
    assert resp.dry_run is True
    assert "Dry-Run Simulation via Key-Alpha-1" in resp.response_text
    assert resp.tokens_in == 15
    assert resp.tokens_out == 25
    assert resp.latency_ms > 0

    # Verify event logged in DB as is_dry_run == 1
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    sliding_rpm, _ = await test_db.get_sliding_window_usage(k1_id, "antigravity")
    assert sliding_rpm == 1


@pytest.mark.asyncio
async def test_api_rest_endpoints(async_client: httpx.AsyncClient) -> None:
    """8. Test FastAPI REST endpoints for dashboard."""
    # Test GET /api/status
    resp_status = await async_client.get("/api/status")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    assert status_data["total_keys"] == 2
    assert status_data["ready_keys"] == 2
    assert status_data["active_model"] == "antigravity"

    # Test GET /api/keys
    resp_keys = await async_client.get("/api/keys")
    assert resp_keys.status_code == 200
    keys_data = resp_keys.json()
    assert len(keys_data) == 2
    assert "masked_key" in keys_data[0]
    assert "antigravity" in keys_data[0]["models_usage"]

    # Test POST /api/keys (Create Key)
    resp_create = await async_client.post(
        "/api/keys",
        json={"key_value": "AQ.Ab8RN6NewKeyViaAPI_9876543210", "label": "Key-Delta-4"},
    )
    assert resp_create.status_code == 201
    created_key = resp_create.json()
    assert created_key["label"] == "Key-Delta-4"
    key_id = created_key["id"]

    # Test PATCH /api/keys/{id}
    resp_patch = await async_client.patch(
        f"/api/keys/{key_id}",
        json={"is_disabled": True, "label": "Key-Delta-Paused"},
    )
    assert resp_patch.status_code == 200
    assert resp_patch.json()["is_disabled"] is True
    assert resp_patch.json()["label"] == "Key-Delta-Paused"

    # Test DELETE /api/keys/{id}
    resp_delete = await async_client.delete(f"/api/keys/{key_id}")
    assert resp_delete.status_code == 200
    assert resp_delete.json() == {"ok": True}

    # Test GET /v1beta/models
    resp_models = await async_client.get("/v1beta/models")
    assert resp_models.status_code == 200
    models_data = resp_models.json()
    assert "models" in models_data
    model_names = [m["name"] for m in models_data["models"]]
    assert "models/antigravity" in model_names

    # Test POST /api/playground and /api/playground/test aliases
    for endpoint in ("/api/playground", "/api/playground/test"):
        resp_play = await async_client.post(
            endpoint,
            json={"model_id": "antigravity", "prompt": "test prompt", "dry_run": True},
        )
        assert resp_play.status_code == 200
        play_data = resp_play.json()
        assert play_data["status"] == "DRY_RUN_SUCCESS"
        assert play_data["dry_run"] is True


@pytest.mark.asyncio
async def test_proxy_generate_and_streaming(async_client: httpx.AsyncClient) -> None:
    """9. Test proxy endpoints in dry-run mode for regular and streaming requests."""
    # Regular generateContent dry-run via header
    resp = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true"},
        json={"contents": [{"parts": [{"text": "Hello world"}]}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data
    assert "usageMetadata" in data

    # Streaming generateContent dry-run via query param
    stream_resp = await async_client.post(
        "/v1beta/models/antigravity:streamGenerateContent?dry_run=true",
        json={"contents": [{"parts": [{"text": "Stream please"}]}]},
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    stream_content = stream_resp.text
    assert "data: " in stream_content
    assert "usageMetadata" in stream_content


@pytest.mark.asyncio
async def test_upstream_429_transparent_retry(test_db: DatabaseManager) -> None:
    """10. Upstream 429 transparent retry: first key gets 429, second key succeeds with 200."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    k2_id = int(keys[1]["id"])

    response_429 = httpx.Response(
        status_code=429,
        json={"error": {"message": "Resource has been exhausted"}},
    )
    response_200 = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello from upstream live response"}],
                        "role": "model",
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 100},
        },
    )

    call_count = 0

    async def mock_post(*args: object, **kwargs: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return response_429
        return response_200

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        res = await execute_request(
            model_id="antigravity",
            action="generateContent",
            body={"contents": [{"parts": [{"text": "Test live"}]}]},
            is_dry_run=False,
            headers={},
            db=test_db,
        )

    assert call_count == 2
    assert "candidates" in res

    # Check k1 received 15s cooldown and consecutive_429 == 1
    k1 = await test_db.get_key_by_id(k1_id)
    assert k1 is not None
    assert k1["consecutive_429"] == 1
    assert float(k1["cooldown_until"]) > time.time()

    # Check k2 succeeded with consecutive_429 == 0
    k2 = await test_db.get_key_by_id(k2_id)
    assert k2 is not None
    assert k2["consecutive_429"] == 0


@pytest.mark.asyncio
async def test_all_keys_exhausted_rpd(test_db: DatabaseManager) -> None:
    """11. When all keys reach their RPD limit for a model, router raises ALL_KEYS_EXHAUSTED."""
    # Model gemini-2.5-flash-tts has RPD limit = 10
    keys = await test_db.get_all_keys()
    past_time = time.time() - 120.0
    for k in keys:
        k_id = int(k["id"])
        for _ in range(10):
            await test_db.record_usage(
                key_id=k_id,
                model_id="gemini-2.5-flash-tts",
                action="generateContent",
                timestamp=past_time,
                tokens_in=10,
                tokens_out=10,
                status_code=200,
                is_dry_run=True,
            )

    with pytest.raises(RouterException) as exc_info:
        await select_best_key("gemini-2.5-flash-tts", test_db)
    assert exc_info.value.status_code == 429
    assert exc_info.value.error_code == RouterErrorCode.ALL_KEYS_EXHAUSTED
    assert exc_info.value.retry_after_seconds is not None


@pytest.mark.asyncio
async def test_all_keys_disabled(test_db: DatabaseManager) -> None:
    """12. When all keys are disabled, router raises NO_KEYS_AVAILABLE with 503."""
    keys = await test_db.get_all_keys()
    for k in keys:
        await test_db.update_key(int(k["id"]), is_disabled=True)

    with pytest.raises(RouterException) as exc_info:
        await select_best_key("antigravity", test_db)
    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == RouterErrorCode.NO_KEYS_AVAILABLE


@pytest.mark.asyncio
async def test_unknown_model_raises_404(test_db: DatabaseManager) -> None:
    """13. Unknown model ID raises MODEL_NOT_FOUND with 404."""
    with pytest.raises(RouterException) as exc_info:
        await select_best_key("non-existent-gemini-model-99", test_db)
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == RouterErrorCode.MODEL_NOT_FOUND


@pytest.mark.asyncio
async def test_root_index_fallback(async_client: httpx.AsyncClient) -> None:
    """14. GET / returns 200 HTML content."""
    resp = await async_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Gemini Key Pool Router" in resp.text


@pytest.mark.asyncio
async def test_connection_pool_singleton() -> None:
    """15. Verify shared httpx.AsyncClient singleton configuration and lifecycle."""
    client1 = await get_http_client()
    client2 = await get_http_client()
    assert client1 is client2
    assert not client1.is_closed

    # Verify connection limits on transport pool
    transport = client1._transport
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    assert transport._pool._max_keepalive_connections == 50
    assert transport._pool._max_connections == 100

    # Verify timeouts
    timeout = client1.timeout
    assert timeout.connect == 5.0
    assert timeout.read == 45.0
    assert timeout.write == 10.0
    assert timeout.pool == 10.0

    await close_http_client()
    assert client1.is_closed

    # Re-acquisition creates a fresh client
    client3 = await get_http_client()
    assert client3 is not client1
    assert not client3.is_closed
    await close_http_client()


@pytest.mark.asyncio
async def test_in_flight_concurrent_key_rotation(test_db: DatabaseManager) -> None:
    """16. Verify concurrent key selection balances across keys using in-flight tracking."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    k2_id = int(keys[1]["id"])

    # 1. First key acquired -> in_flight for k1 = 1
    sel1 = await select_best_key("antigravity", test_db, acquire_in_flight=True)
    assert sel1.key_id == k1_id
    counts1 = get_in_flight_requests()
    assert counts1.get(k1_id) == 1
    assert counts1.get(k2_id, 0) == 0

    # 2. Second concurrent request must pick k2 because k1 has in_flight=1
    sel2 = await select_best_key("antigravity", test_db, acquire_in_flight=True)
    assert sel2.key_id == k2_id
    counts2 = get_in_flight_requests()
    assert counts2.get(k1_id) == 1
    assert counts2.get(k2_id) == 1

    # 3. Third concurrent request: both have in_flight=1 and rpd=0 -> picks k1 (lowest ID)
    sel3 = await select_best_key("antigravity", test_db, acquire_in_flight=True)
    assert sel3.key_id == k1_id
    counts3 = get_in_flight_requests()
    assert counts3.get(k1_id) == 2
    assert counts3.get(k2_id) == 1

    # Decrement in_flight in finally-like behavior
    await decrement_in_flight(sel1.key_id)
    await decrement_in_flight(sel2.key_id)
    await decrement_in_flight(sel3.key_id)

    assert get_in_flight_requests() == {}


@pytest.mark.asyncio
async def test_upstream_503_transparent_retry(test_db: DatabaseManager) -> None:
    """17. Upstream 503 Overloaded error triggers soft cooldown and transparent retry on next key."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    k2_id = int(keys[1]["id"])

    response_503 = httpx.Response(
        status_code=503,
        json={"error": {"code": 503, "message": "The model is overloaded. Please try again later."}},
    )
    response_200 = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Success after 503 failover"}],
                        "role": "model",
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 30},
        },
    )

    calls: list[str] = []

    async def mock_post(*args: object, **kwargs: object) -> httpx.Response:
        req_headers = kwargs.get("headers", {})
        api_key = req_headers.get("x-goog-api-key", "") if isinstance(req_headers, dict) else ""
        calls.append(api_key)
        if len(calls) == 1:
            return response_503
        return response_200

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        res = await execute_request(
            model_id="antigravity",
            action="generateContent",
            body={"contents": [{"parts": [{"text": "Hello"}]}]},
            is_dry_run=False,
            headers={},
            db=test_db,
        )

    assert len(calls) == 2
    assert "candidates" in res

    # Key 1 was put in soft cooldown
    k1 = await test_db.get_key_by_id(k1_id)
    assert k1 is not None
    assert float(k1["cooldown_until"]) > time.time()
    # Consecutive 429 is not incremented for 503
    assert k1["consecutive_429"] == 0

    # Key 2 succeeded
    k2 = await test_db.get_key_by_id(k2_id)
    assert k2 is not None
    assert k2["consecutive_429"] == 0


@pytest.mark.asyncio
async def test_upstream_timeout_transparent_retry(test_db: DatabaseManager) -> None:
    """18. Upstream ReadTimeout triggers soft cooldown and transparent retry on next key."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    k2_id = int(keys[1]["id"])

    response_200 = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Success after timeout failover"}],
                        "role": "model",
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 25, "candidatesTokenCount": 35},
        },
    )

    calls_count = 0

    async def mock_post(*args: object, **kwargs: object) -> httpx.Response:
        nonlocal calls_count
        calls_count += 1
        if calls_count == 1:
            raise httpx.ReadTimeout("Timed out reading from server")
        return response_200

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        res = await execute_request(
            model_id="antigravity",
            action="generateContent",
            body={"contents": [{"parts": [{"text": "Hello timeout"}]}]},
            is_dry_run=False,
            headers={},
            db=test_db,
        )

    assert calls_count == 2
    assert "candidates" in res

    # Key 1 was put in soft cooldown
    k1 = await test_db.get_key_by_id(k1_id)
    assert k1 is not None
    assert float(k1["cooldown_until"]) > time.time()


@pytest.mark.asyncio
async def test_upstream_503_all_keys_exhausted(test_db: DatabaseManager) -> None:
    """19. When all keys return 503, router exhausts attempts and raises UPSTREAM_ERROR with 503."""
    response_503 = httpx.Response(
        status_code=503,
        json={"error": {"code": 503, "message": "The model is overloaded. Please try again later."}},
    )

    async def mock_post(*args: object, **kwargs: object) -> httpx.Response:
        return response_503

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        with pytest.raises(RouterException) as exc_info:
            await execute_request(
                model_id="antigravity",
                action="generateContent",
                body={"contents": [{"parts": [{"text": "Hello overload"}]}]},
                is_dry_run=False,
                headers={},
                db=test_db,
            )

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == RouterErrorCode.UPSTREAM_ERROR
    assert "503" in exc_info.value.message
    assert exc_info.value.selected_key_label in ("Key-Alpha-1", "Key-Beta-2")


@pytest.mark.asyncio
async def test_gemma_model_alias_mapping(test_db: DatabaseManager) -> None:
    """20. Gemma models transparently map to Google upstream endpoints."""
    assert UPSTREAM_MODEL_ALIASES["gemma-4-26b"] == "gemma-4-26b-a4b-it"
    assert UPSTREAM_MODEL_ALIASES["gemma-4-31b"] == "gemma-4-31b-it"

    captured_urls: list[str] = []

    async def mock_post(url: str, *args: object, **kwargs: object) -> httpx.Response:
        captured_urls.append(str(url))
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemma live response"}],
                            "role": "model",
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            },
        )

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        # 1. gemma-4-31b -> gemma-4-31b-it
        await execute_request(
            model_id="gemma-4-31b",
            action="generateContent",
            body={"contents": [{"parts": [{"text": "Prompt 31b"}]}]},
            is_dry_run=False,
            headers={},
            db=test_db,
        )
        assert captured_urls[-1] == "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent"

        # 2. gemma-4-26b -> gemma-4-26b-a4b-it
        await execute_request(
            model_id="gemma-4-26b",
            action="generateContent",
            body={"contents": [{"parts": [{"text": "Prompt 26b"}]}]},
            is_dry_run=False,
            headers={},
            db=test_db,
        )
        assert captured_urls[-1] == "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent"

        # 3. Direct gemma-4-31b-it stays gemma-4-31b-it
        await execute_request(
            model_id="gemma-4-31b-it",
            action="generateContent",
            body={"contents": [{"parts": [{"text": "Prompt 31b-it"}]}]},
            is_dry_run=False,
            headers={},
            db=test_db,
        )
        assert captured_urls[-1] == "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent"


@pytest.mark.asyncio
async def test_gemma_model_alias_mapping_fastapi_endpoint(
    async_client: httpx.AsyncClient,
) -> None:
    """21. FastAPI endpoint for gemma-4-31b routes to upstream gemma-4-31b-it."""
    internal_client = await get_http_client()
    captured_urls: list[str] = []

    async def mock_post(url: str, *args: object, **kwargs: object) -> httpx.Response:
        captured_urls.append(str(url))
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemma 31b API"}],
                            "role": "model",
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10},
            },
        )

    with patch.object(internal_client, "post", side_effect=mock_post):
        resp = await async_client.post(
            "/v1beta/models/gemma-4-31b:generateContent",
            json={"contents": [{"parts": [{"text": "Hello Gemma"}]}]},
        )
        assert resp.status_code == 200
        assert captured_urls[-1] == "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent"


@pytest.mark.asyncio
async def test_upstream_error_404_selected_key_label(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """22. Upstream 404 permanent client error returns selected_key_label."""
    internal_client = await get_http_client()

    async def mock_404(*args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            json={"error": {"code": 404, "message": "Model not found upstream"}},
        )

    with patch.object(internal_client, "post", side_effect=mock_404):
        resp_404 = await async_client.post(
            "/v1beta/models/gemma-4-31b:generateContent",
            json={"contents": [{"parts": [{"text": "Hello 404"}]}]},
        )
        assert resp_404.status_code == 404
        body_404 = resp_404.json()
        assert body_404["selected_key_label"] == "Key-Alpha-1"


@pytest.mark.asyncio
async def test_upstream_error_500_selected_key_label(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """23. Upstream 500 server error across retries returns selected_key_label."""
    internal_client = await get_http_client()
    await test_db.add_key("AQ.Ab8RN6ThirdKeyTesting_1234567890", "Key-Gamma-3")

    async def mock_500(*args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={"error": {"code": 500, "message": "Internal Server Error"}},
        )

    with patch.object(internal_client, "post", side_effect=mock_500):
        resp_500 = await async_client.post(
            "/v1beta/models/gemma-4-31b:generateContent",
            json={"contents": [{"parts": [{"text": "Hello 500"}]}]},
        )
        assert resp_500.status_code == 500
        body_500 = resp_500.json()
        assert body_500["selected_key_label"] in ("Key-Alpha-1", "Key-Beta-2", "Key-Gamma-3")


@pytest.mark.asyncio
async def test_upstream_error_network_selected_key_label(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """24. Upstream network error across retries returns selected_key_label."""
    internal_client = await get_http_client()
    await test_db.add_key("AQ.Ab8RN6ThirdKeyTesting_1234567890", "Key-Gamma-3")

    async def mock_net_err(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("Failed to connect to upstream server")

    with patch.object(internal_client, "post", side_effect=mock_net_err):
        resp_net = await async_client.post(
            "/v1beta/models/gemma-4-31b:generateContent",
            json={"contents": [{"parts": [{"text": "Hello network error"}]}]},
        )
        assert resp_net.status_code == 502
        body_net = resp_net.json()
        assert body_net["selected_key_label"] in ("Key-Alpha-1", "Key-Beta-2", "Key-Gamma-3")


@pytest.mark.asyncio
async def test_upstream_error_429_selected_key_label(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """25. Upstream 429 across retries returns selected_key_label."""
    internal_client = await get_http_client()
    await test_db.add_key("AQ.Ab8RN6ThirdKeyTesting_1234567890", "Key-Gamma-3")

    async def mock_429(*args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json={"error": {"code": 429, "message": "Resource has been exhausted"}},
        )

    with patch.object(internal_client, "post", side_effect=mock_429):
        resp_429 = await async_client.post(
            "/v1beta/models/gemma-4-31b:generateContent",
            json={"contents": [{"parts": [{"text": "Hello 429"}]}]},
        )
        assert resp_429.status_code == 429
        body_429 = resp_429.json()
        assert body_429["selected_key_label"] in ("Key-Alpha-1", "Key-Beta-2", "Key-Gamma-3")


@pytest.mark.asyncio
async def test_select_best_key_with_wait_success(test_db: DatabaseManager) -> None:
    """26. Key becomes ready after brief cooldown while waiting in queue."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    k2_id = int(keys[1]["id"])

    # Put both keys into a brief 0.1s cooldown
    await test_db.set_cooldown(k1_id, 0.1)
    await test_db.set_cooldown(k2_id, 0.1)

    # select_best_key_with_wait with wait_timeout=1.0 should wait and succeed
    start = time.time()
    res = await select_best_key_with_wait("antigravity", test_db, wait_timeout=1.0)
    duration = time.time() - start

    assert res.key_id in (k1_id, k2_id)
    assert duration >= 0.08


@pytest.mark.asyncio
async def test_select_best_key_with_wait_timeout_exceeded(test_db: DatabaseManager) -> None:
    """27. Queue wait timeout exceeded raises 429 QUEUE_TIMEOUT."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    k2_id = int(keys[1]["id"])

    # Put both keys into 5s cooldown
    await test_db.set_cooldown(k1_id, 5.0)
    await test_db.set_cooldown(k2_id, 5.0)

    with pytest.raises(RouterException) as exc_info:
        await select_best_key_with_wait("antigravity", test_db, wait_timeout=0.1)

    assert exc_info.value.status_code == 429
    assert exc_info.value.error_code == RouterErrorCode.QUEUE_TIMEOUT
    assert "Queue wait timeout (0.1s) exceeded" in exc_info.value.message


@pytest.mark.asyncio
async def test_select_best_key_with_wait_24h(test_db: DatabaseManager) -> None:
    """28. wait_timeout = -1 allows waiting with 24h effective timeout."""
    keys = await test_db.get_all_keys()
    k1_id = int(keys[0]["id"])
    k2_id = int(keys[1]["id"])

    # Key with brief cooldown works when wait_timeout=-1
    await test_db.set_cooldown(k1_id, 0.1)
    await test_db.set_cooldown(k2_id, 0.1)

    res = await select_best_key_with_wait("antigravity", test_db, wait_timeout=-1)
    assert res.key_id in (k1_id, k2_id)


@pytest.mark.asyncio
async def test_stream_generate_content_dry_run(async_client: httpx.AsyncClient) -> None:
    """29. Streaming generateContent dry-run returns text/event-stream chunks."""
    resp = await async_client.post(
        "/v1beta/models/antigravity:streamGenerateContent?dry_run=true",
        headers={"X-Wait-Timeout": "5.0"},
        json={"contents": [{"parts": [{"text": "Stream test prompt"}]}]},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    lines = resp.text.strip().split("\n\n")
    assert len(lines) >= 2
    assert "Dry-Run Simulation" in lines[0]
    assert "usageMetadata" in lines[1]


class MockStreamContext:
    """Mock context manager for httpx.AsyncClient.stream."""

    def __init__(self, status_code: int = 200, lines: list[str] | None = None) -> None:
        self.status_code: int = status_code
        self._lines: list[str] = lines or []

    async def __aenter__(self) -> "MockStreamContext":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""


@pytest.mark.asyncio
async def test_playground_stream_generate_content(test_db: DatabaseManager) -> None:
    """30. execute_playground_test supports streamGenerateContent in both dry-run and live modes."""
    # Test 1: Dry run
    req_dry = PlaygroundRequest(
        model_id="antigravity",
        prompt="Dry run test prompt",
        action="streamGenerateContent",
        dry_run=True,
        wait_timeout=1.0,
    )
    resp_dry = await execute_playground_test(req_dry, test_db)
    assert resp_dry.status == PlaygroundStatus.DRY_RUN_SUCCESS.value
    assert resp_dry.dry_run is True
    assert "Dry-Run Simulation" in resp_dry.response_text
    assert resp_dry.tokens_in == 15
    assert resp_dry.tokens_out == 25

    # Test 2: Live upstream mock
    req_live = PlaygroundRequest(
        model_id="antigravity",
        prompt="Live test stream",
        action="streamGenerateContent",
        dry_run=False,
        wait_timeout=2.0,
    )

    mock_lines = [
        'data: {"candidates": [{"content": {"parts": [{"text": "Chunk 1: Hello "}], "role": "model"}, "index": 0}]}',
        '',
        'data: {"candidates": [{"content": {"parts": [{"text": "Chunk 2: World!"}], "role": "model"}, "finishReason": "STOP", "index": 0}], "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 16}}',
        '',
    ]

    internal_client = await get_http_client()
    with patch.object(internal_client, "stream", return_value=MockStreamContext(200, mock_lines)):
        resp_live = await execute_playground_test(req_live, test_db)

    assert resp_live.status == PlaygroundStatus.SUCCESS.value
    assert resp_live.dry_run is False
    assert resp_live.response_text == "Chunk 1: Hello Chunk 2: World!"
    assert resp_live.tokens_in == 8
    assert resp_live.tokens_out == 16


@pytest.mark.asyncio
async def test_live_streaming_endpoint(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """31. Live proxy endpoint streamGenerateContent streams SSE chunks from upstream."""
    mock_lines = [
        'data: {"candidates": [{"content": {"parts": [{"text": "Streaming live chunk"}]}}]}',
        '',
    ]

    internal_client = await get_http_client()
    with patch.object(internal_client, "stream", return_value=MockStreamContext(200, mock_lines)):
        resp = await async_client.post(
            "/v1beta/models/antigravity:streamGenerateContent",
            headers={"X-Wait-Timeout": "1.0"},
            json={"contents": [{"parts": [{"text": "Hello stream"}]}]},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "Streaming live chunk" in resp.text


@pytest.mark.asyncio
async def test_client_key_crud_api(async_client: httpx.AsyncClient) -> None:
    """32. Test CRUD operations for client ingress API keys via REST API."""
    # 1. Initially empty
    resp_init = await async_client.get("/api/client-keys")
    assert resp_init.status_code == 200
    assert resp_init.json() == []

    # 2. Create client key with generated key
    create_payload = {
        "name": "Frontend App Key",
        "rpm_limit": 30,
        "rpd_limit": 500,
    }
    resp_create = await async_client.post("/api/client-keys", json=create_payload)
    assert resp_create.status_code == 201
    created_data = resp_create.json()
    assert "raw_key" in created_data
    raw_key = created_data["raw_key"]
    assert raw_key.startswith("gkr-live-")
    rec = created_data["key_record"]
    assert rec["name"] == "Frontend App Key"
    assert rec["rpm_limit"] == 30
    assert rec["rpd_limit"] == 500
    assert rec["status"] == "ACTIVE"
    assert rec["rpm_current"] == 0
    assert rec["rpd_current"] == 0
    key_id = rec["id"]

    # 3. Create client key with custom key
    custom_payload = {
        "name": "Custom Key Service",
        "custom_key": "custom-secret-key-12345",
        "rpm_limit": 60,
        "rpd_limit": 1000,
    }
    resp_custom = await async_client.post("/api/client-keys", json=custom_payload)
    assert resp_custom.status_code == 201
    custom_data = resp_custom.json()
    assert custom_data["raw_key"] == "custom-secret-key-12345"
    assert custom_data["key_record"]["key_prefix"] == mask_client_api_key("custom-secret-key-12345")

    # 4. List client keys
    resp_list = await async_client.get("/api/client-keys")
    assert resp_list.status_code == 200
    keys_list = resp_list.json()
    assert len(keys_list) == 2
    ids = [k["id"] for k in keys_list]
    assert key_id in ids

    # 5. Patch client key
    patch_payload = {
        "name": "Frontend App Renamed",
        "rpm_limit": 45,
        "status": "PAUSED",
    }
    resp_patch = await async_client.patch(f"/api/client-keys/{key_id}", json=patch_payload)
    assert resp_patch.status_code == 200
    patched = resp_patch.json()
    assert patched["name"] == "Frontend App Renamed"
    assert patched["rpm_limit"] == 45
    assert patched["rpd_limit"] == 500
    assert patched["status"] == "PAUSED"

    # 6. Delete client key
    resp_del = await async_client.delete(f"/api/client-keys/{key_id}")
    assert resp_del.status_code == 204

    # 7. Verify deletion
    resp_list_after = await async_client.get("/api/client-keys")
    assert resp_list_after.status_code == 200
    keys_after = resp_list_after.json()
    assert len(keys_after) == 1
    assert keys_after[0]["name"] == "Custom Key Service"

    # 8. Deleting non-existent key returns 404
    resp_del_404 = await async_client.delete(f"/api/client-keys/{key_id}")
    assert resp_del_404.status_code == 404


@pytest.mark.asyncio
async def test_client_open_access_when_pool_empty(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """33. Access to /v1beta/* when client keys pool is empty is open (backward compatible)."""
    keys_count = await test_db.count_client_keys()
    assert keys_count == 0

    resp = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true"},
        json={"contents": [{"parts": [{"text": "Hello world open access"}]}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data


@pytest.mark.asyncio
async def test_client_auth_401_errors(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """34. Error 401 UNAUTHORIZED when client keys exist and missing or invalid token."""
    await test_db.create_client_key(
        name="Test Auth Key",
        key_value="gkr-live-validtoken1234567890",
    )

    # A: Missing token
    resp_missing = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true"},
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert resp_missing.status_code == 401
    err_missing = resp_missing.json()
    assert err_missing["error_code"] == RouterErrorCode.UNAUTHORIZED.value
    assert "Missing Client API Key" in err_missing["message"]

    # B: Invalid token
    resp_invalid = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": "Bearer invalid_secret_key"},
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert resp_invalid.status_code == 401
    err_invalid = resp_invalid.json()
    assert err_invalid["error_code"] == RouterErrorCode.UNAUTHORIZED.value
    assert "Invalid Client API Key" in err_invalid["message"]


@pytest.mark.asyncio
async def test_client_auth_all_four_formats(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """35. Successful authorization via all 4 formats: Bearer, x-goog-api-key, X-Router-Key, ?key=."""
    created = await test_db.create_client_key(
        name="Multi-Auth Key",
        key_value="gkr-live-multiauth987654321",
        rpm_limit=60,
        rpd_limit=1000,
    )
    token = "gkr-live-multiauth987654321"
    kid = int(created["id"])

    # 1. Authorization: Bearer
    resp1 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": f"Bearer {token}"},
        json={"contents": [{"parts": [{"text": "Format 1"}]}]},
    )
    assert resp1.status_code == 200

    # 2. x-goog-api-key
    resp2 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "x-goog-api-key": token},
        json={"contents": [{"parts": [{"text": "Format 2"}]}]},
    )
    assert resp2.status_code == 200

    # 3. X-Router-Key
    resp3 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "X-Router-Key": token},
        json={"contents": [{"parts": [{"text": "Format 3"}]}]},
    )
    assert resp3.status_code == 200

    # 4. ?key= query parameter
    resp4 = await async_client.post(
        f"/v1beta/models/antigravity:generateContent?key={token}",
        headers={"X-Dry-Run": "true"},
        json={"contents": [{"parts": [{"text": "Format 4"}]}]},
    )
    assert resp4.status_code == 200

    # Verify total_requests incremented to 4
    updated_key = await test_db.get_client_key_by_id(kid)
    assert updated_key is not None
    assert updated_key["total_requests"] == 4
    assert updated_key["last_used_at"] is not None


@pytest.mark.asyncio
async def test_client_key_status_paused_and_revoked(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """36. Error 403 UNAUTHORIZED when client key status is PAUSED or REVOKED."""
    paused = await test_db.create_client_key(
        name="Paused Key",
        key_value="gkr-live-pausedkey12345678",
    )
    await test_db.update_client_key(int(paused["id"]), status="PAUSED")

    revoked = await test_db.create_client_key(
        name="Revoked Key",
        key_value="gkr-live-revokedkey12345678",
    )
    await test_db.update_client_key(int(revoked["id"]), status="REVOKED")

    # Test PAUSED key -> 403
    resp_paused = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": "Bearer gkr-live-pausedkey12345678"},
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert resp_paused.status_code == 403
    err_paused = resp_paused.json()
    assert err_paused["error_code"] == RouterErrorCode.UNAUTHORIZED.value
    assert "Client API Key is paused. Access denied." in err_paused["message"]

    # Test REVOKED key -> 403
    resp_revoked = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": "Bearer gkr-live-revokedkey12345678"},
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert resp_revoked.status_code == 403
    err_revoked = resp_revoked.json()
    assert err_revoked["error_code"] == RouterErrorCode.UNAUTHORIZED.value
    assert "Client API Key is revoked. Access denied." in err_revoked["message"]


@pytest.mark.asyncio
async def test_client_rate_limiting_rpm_and_rpd(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """37. Error 429 CLIENT_RATE_LIMITED when client RPM or daily RPD limit is exceeded."""
    # 1. Test RPM rate limit
    await test_db.create_client_key(
        name="RPM Limited Key",
        key_value="gkr-live-rpmlimitedkey123",
        rpm_limit=2,
        rpd_limit=1000,
    )
    rpm_token = "gkr-live-rpmlimitedkey123"

    # 1st request -> OK
    r1 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": f"Bearer {rpm_token}"},
        json={"contents": [{"parts": [{"text": "req 1"}]}]},
    )
    assert r1.status_code == 200

    # 2nd request -> OK
    r2 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": f"Bearer {rpm_token}"},
        json={"contents": [{"parts": [{"text": "req 2"}]}]},
    )
    assert r2.status_code == 200

    # 3rd request -> 429 RPM exceeded
    r3 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": f"Bearer {rpm_token}"},
        json={"contents": [{"parts": [{"text": "req 3"}]}]},
    )
    assert r3.status_code == 429
    err3 = r3.json()
    assert err3["error_code"] == RouterErrorCode.CLIENT_RATE_LIMITED.value
    assert "Client API Key RPM limit (2) exceeded." in err3["message"]

    # 2. Test RPD rate limit
    await test_db.create_client_key(
        name="RPD Limited Key",
        key_value="gkr-live-rpdlimitedkey123",
        rpm_limit=10,
        rpd_limit=2,
    )
    rpd_token = "gkr-live-rpdlimitedkey123"

    # 1st request -> OK
    rd1 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": f"Bearer {rpd_token}"},
        json={"contents": [{"parts": [{"text": "req 1"}]}]},
    )
    assert rd1.status_code == 200

    # 2nd request -> OK
    rd2 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": f"Bearer {rpd_token}"},
        json={"contents": [{"parts": [{"text": "req 2"}]}]},
    )
    assert rd2.status_code == 200

    # 3rd request -> 429 RPD exceeded
    rd3 = await async_client.post(
        "/v1beta/models/antigravity:generateContent",
        headers={"X-Dry-Run": "true", "Authorization": f"Bearer {rpd_token}"},
        json={"contents": [{"parts": [{"text": "req 3"}]}]},
    )
    assert rd3.status_code == 429
    err_rd3 = rd3.json()
    assert err_rd3["error_code"] == RouterErrorCode.CLIENT_RATE_LIMITED.value
    assert "Client API Key daily RPD limit (2) exceeded." in err_rd3["message"]


@pytest.mark.asyncio
async def test_client_streaming_usage_recorded(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """38. Verify client key usage is recorded properly on streamGenerateContent dry-run."""
    stream_key = await test_db.create_client_key(
        name="Stream Client Key",
        key_value="gkr-live-streamkey123456",
        rpm_limit=10,
        rpd_limit=100,
    )
    token = "gkr-live-streamkey123456"

    resp = await async_client.post(
        "/v1beta/models/antigravity:streamGenerateContent?dry_run=true",
        headers={"Authorization": f"Bearer {token}"},
        json={"contents": [{"parts": [{"text": "Stream test"}]}]},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    k = await test_db.get_client_key_by_id(int(stream_key["id"]))
    assert k is not None
    assert k["total_requests"] == 1
    assert k["total_tokens"] == 40
    assert k["last_used_at"] is not None


@pytest.mark.asyncio
async def test_record_and_get_request_logs(test_db: DatabaseManager) -> None:
    """39. Verify saving prompt, response, tokens, and latency to request_logs table."""
    ck = await test_db.create_client_key(name="Client-Alpha", key_value="gkr-live-clientalpha1")
    ck_id = int(ck["id"])

    inserted_id = await test_db.record_request_log(
        timestamp=1700000000.0,
        timestamp_iso="2023-11-14T22:13:20+03:00",
        model_id="gemini-2.5-flash",
        action="generateContent",
        prompt_text="Hello Gemini",
        client_key_id=ck_id,
        client_key_name="Client-Alpha",
        google_key_id=2,
        google_key_label="Key-Beta-2",
        prompt_full_json={"contents": [{"parts": [{"text": "Hello Gemini"}]}]},
        response_text="Hello Human",
        response_full_json={"candidates": [{"content": {"parts": [{"text": "Hello Human"}]}}]},
        tokens_in=10,
        tokens_out=20,
        latency_ms=125.5,
        status_code=200,
        error_message=None,
    )
    assert inserted_id > 0

    records, total_count, latest_ts = await test_db.get_request_logs()
    assert total_count == 1
    assert latest_ts == 1700000000.0
    assert len(records) == 1

    rec = records[0]
    assert rec["id"] == inserted_id
    assert rec["model_id"] == "gemini-2.5-flash"
    assert rec["action"] == "generateContent"
    assert rec["prompt_text"] == "Hello Gemini"
    assert rec["response_text"] == "Hello Human"
    assert rec["tokens_in"] == 10
    assert rec["tokens_out"] == 20
    assert rec["latency_ms"] == 125.5
    assert rec["status_code"] == 200
    assert rec["client_key_name"] == "Client-Alpha"
    assert rec["google_key_label"] == "Key-Beta-2"
    assert rec["error_message"] is None


@pytest.mark.asyncio
async def test_request_logs_client_key_filter(test_db: DatabaseManager) -> None:
    """40. Verify filtering request logs by client_key_id."""
    k1 = await test_db.create_client_key(name="Client 1", key_value="gkr-live-client1key1")
    k2 = await test_db.create_client_key(name="Client 2", key_value="gkr-live-client2key2")
    k1_id = int(k1["id"])
    k2_id = int(k2["id"])

    await test_db.record_request_log(
        timestamp=1700000001.0,
        timestamp_iso="2023-11-14T22:13:21+03:00",
        model_id="antigravity",
        action="generateContent",
        prompt_text="Prompt from client 1",
        client_key_id=k1_id,
        client_key_name="Client 1",
    )
    await test_db.record_request_log(
        timestamp=1700000002.0,
        timestamp_iso="2023-11-14T22:13:22+03:00",
        model_id="antigravity",
        action="generateContent",
        prompt_text="Prompt from client 2",
        client_key_id=k2_id,
        client_key_name="Client 2",
    )

    logs_k1, count_k1, _ = await test_db.get_request_logs(client_key_id=k1_id)
    assert count_k1 == 1
    assert len(logs_k1) == 1
    assert logs_k1[0]["client_key_id"] == k1_id
    assert logs_k1[0]["prompt_text"] == "Prompt from client 1"

    logs_k2, count_k2, _ = await test_db.get_request_logs(client_key_id=k2_id)
    assert count_k2 == 1
    assert len(logs_k2) == 1
    assert logs_k2[0]["client_key_id"] == k2_id
    assert logs_k2[0]["prompt_text"] == "Prompt from client 2"


@pytest.mark.asyncio
async def test_request_logs_search(test_db: DatabaseManager) -> None:
    """41. Verify searching request logs by keyword in prompt or response text."""
    await test_db.record_request_log(
        timestamp=1700000010.0,
        timestamp_iso="2023-11-14T22:13:30+03:00",
        model_id="antigravity",
        action="generateContent",
        prompt_text="Quantum physics and black holes",
        response_text="Hawking radiation detected",
    )
    await test_db.record_request_log(
        timestamp=1700000020.0,
        timestamp_iso="2023-11-14T22:13:40+03:00",
        model_id="antigravity",
        action="generateContent",
        prompt_text="Delicious chocolate cake recipe",
        response_text="Mix flour and cocoa powder",
    )

    # Search in prompt
    res_physics, count_p, _ = await test_db.get_request_logs(search="physics")
    assert count_p == 1
    assert len(res_physics) == 1
    assert "Quantum physics" in str(res_physics[0]["prompt_text"])

    # Search in response
    res_cocoa, count_c, _ = await test_db.get_request_logs(search="cocoa")
    assert count_c == 1
    assert len(res_cocoa) == 1
    assert "Mix flour" in str(res_cocoa[0]["response_text"])

    # No match
    res_none, count_n, _ = await test_db.get_request_logs(search="astronaut")
    assert count_n == 0
    assert len(res_none) == 0

    # Since timestamp
    res_since, count_s, _ = await test_db.get_request_logs(since_timestamp=1700000015.0)
    assert count_s == 1
    assert len(res_since) == 1
    assert str(res_since[0]["prompt_text"]) == "Delicious chocolate cake recipe"


@pytest.mark.asyncio
async def test_request_logs_api_endpoints(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """42. Verify GET /api/logs and DELETE /api/logs REST endpoints."""
    # Create client key
    key_resp = await async_client.post(
        "/api/client-keys",
        json={"name": "Jury Tester", "rpm_limit": 60, "rpd_limit": 100},
    )
    assert key_resp.status_code == 201
    key_data = key_resp.json()
    token = key_data["raw_key"]

    # Execute a request via proxy endpoint
    proxy_resp = await async_client.post(
        "/v1beta/models/antigravity:generateContent?dry_run=true",
        headers={"Authorization": f"Bearer {token}"},
        json={"contents": [{"parts": [{"text": "Hello jury inspection"}]}]},
    )
    assert proxy_resp.status_code == 200

    # GET /api/logs
    logs_resp = await async_client.get("/api/logs")
    assert logs_resp.status_code == 200
    logs_data = logs_resp.json()
    assert logs_data["total_count"] == 1
    assert len(logs_data["logs"]) == 1
    log_item = logs_data["logs"][0]
    assert log_item["prompt_text"] == "Hello jury inspection"
    assert log_item["client_key_name"] == "Jury Tester"
    assert log_item["status_code"] == 200
    assert "Dry-Run Simulation" in log_item["response_text"]
    assert isinstance(log_item["prompt_full"], dict)

    # Test blocked unauthorized access is logged
    bad_resp = await async_client.post(
        "/v1beta/models/antigravity:generateContent?dry_run=true",
        headers={"Authorization": "Bearer gkr-live-invalidtoken"},
        json={"contents": [{"parts": [{"text": "Blocked unauthorized request"}]}]},
    )
    assert bad_resp.status_code == 401

    logs_after_unauth = await async_client.get("/api/logs")
    assert logs_after_unauth.status_code == 200
    logs_unauth_data = logs_after_unauth.json()
    assert logs_unauth_data["total_count"] == 2
    unauth_log = logs_unauth_data["logs"][0]
    assert unauth_log["status_code"] == 401
    assert unauth_log["prompt_text"] == "Blocked unauthorized request"
    assert "Invalid Client API Key" in str(unauth_log["error_message"])

    # DELETE /api/logs
    del_resp = await async_client.delete("/api/logs")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"ok": True}

    # Verify logs are cleared
    empty_logs = await async_client.get("/api/logs")
    assert empty_logs.status_code == 200
    assert empty_logs.json()["total_count"] == 0
    assert empty_logs.json()["logs"] == []


@pytest.mark.asyncio
async def test_thought_signature_caching_and_restoration() -> None:
    """Verify caching of thought_signature and automatic restoration in requests."""
    await clear_thought_signature_cache()

    # 1. Cache signature from Gemini response payload
    gemini_response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "Tokyo", "units": "metric"},
                            },
                            "thought_signature": "sig_tokyo_metric_12345",
                        },
                        {
                            "functionCall": {
                                "name": "calculate_tax",
                                "args": {"amount": 100},
                            },
                            "thoughtSignature": "sig_tax_calc_67890",
                        },
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    }
    await cache_thought_signatures(gemini_response)

    # 2. Request body missing thought_signature (exact match)
    request_body_exact: dict[str, object] = {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Tokyo", "units": "metric"},
                        }
                    }
                ],
            }
        ]
    }

    contents_list = request_body_exact.get("contents")
    assert isinstance(contents_list, list) and len(contents_list) > 0
    first_item = contents_list[0]
    assert isinstance(first_item, dict)
    parts_list = first_item.get("parts")
    assert isinstance(parts_list, list) and len(parts_list) > 0
    part_ref = parts_list[0]
    assert isinstance(part_ref, dict)
    assert "thought_signature" not in part_ref

    restored = await restore_missing_thought_signatures(request_body_exact)
    assert restored is True
    assert part_ref["thought_signature"] == "sig_tokyo_metric_12345"

    # Restoring again should return False (already has signature)
    assert await restore_missing_thought_signatures(request_body_exact) is False

    # 3. Request body with different args (fallback to func_name match)
    request_body_fallback: dict[str, object] = {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "calculate_tax",
                            "args": {"amount": 250},
                        }
                    }
                ],
            }
        ]
    }
    fb_contents = request_body_fallback.get("contents")
    assert isinstance(fb_contents, list) and len(fb_contents) > 0
    fb_first = fb_contents[0]
    assert isinstance(fb_first, dict)
    fb_parts = fb_first.get("parts")
    assert isinstance(fb_parts, list) and len(fb_parts) > 0
    fb_part = fb_parts[0]
    assert isinstance(fb_part, dict)
    restored_fb = await restore_missing_thought_signatures(request_body_fallback)
    assert restored_fb is True
    assert fb_part["thought_signature"] == "sig_tax_calc_67890"

    # 4. Unknown function returns False
    unknown_req: dict[str, object] = {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "unknown_tool",
                            "args": {},
                        }
                    }
                ],
            }
        ]
    }
    assert await restore_missing_thought_signatures(unknown_req) is False

    # 5. Empty or None body
    assert await restore_missing_thought_signatures(None) is False
    assert await restore_missing_thought_signatures({}) is False

    await clear_thought_signature_cache()


@pytest.mark.asyncio
async def test_record_request_log_with_tool_calls(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """Verify tool_calls storage and extraction in database and API responses."""
    tool_calls = [
        {
            "name": "search_db",
            "args": {"query": "SELECT 1", "limit": 10},
            "id": "call_abc_123",
            "thought_signature": "sig_search_db_777",
        },
        {
            "name": "send_alert",
            "args": {"channel": "ops", "severity": "high"},
            "id": "call_alert_456",
            "thought_signature": "sig_alert_888",
        },
    ]

    log_id = await test_db.record_request_log(
        timestamp=time.time(),
        timestamp_iso=datetime.now(ZoneInfo(RESET_TIMEZONE)).isoformat(),
        model_id="gemini-3.1-flash-lite",
        action="generateContent",
        prompt_text="Execute operations with tools",
        tool_calls=tool_calls,
        tokens_in=50,
        tokens_out=120,
        status_code=200,
    )
    assert log_id > 0

    # 1. Verify direct DB extraction
    records, total_count, _ = await test_db.get_request_logs(limit=10)
    assert total_count >= 1
    found = next((r for r in records if r["id"] == log_id), None)
    assert found is not None
    assert found["tool_calls"] == tool_calls
    assert found["tool_calls_json"] is not None

    # 2. Verify /api/logs endpoint response
    api_resp = await async_client.get("/api/logs")
    assert api_resp.status_code == 200
    resp_data = api_resp.json()
    assert resp_data["total_count"] >= 1
    log_rec = next((l for l in resp_data["logs"] if l["id"] == log_id), None)
    assert log_rec is not None
    assert log_rec["tool_calls"] == tool_calls
    assert len(log_rec["tool_calls"]) == 2
    assert log_rec["tool_calls"][0]["name"] == "search_db"
    assert log_rec["tool_calls"][0]["thought_signature"] == "sig_search_db_777"


@pytest.mark.asyncio
async def test_stress_test_13_keys_pool(
    test_db: DatabaseManager, async_client: httpx.AsyncClient
) -> None:
    """Verify stress test execution across a pool of 13 keys."""
    from tests.stress_test_13_keys import (
        calculate_percentile,
        print_report,
        send_single_request,
    )

    # 1. Seed pool up to 13 keys with realistic labels
    extra_labels = [
        "Mama",
        "Nikitaloskutov",
        "artem1",
        "artem2",
        "artem3",
        "artem4",
        "artem5",
        "Key-Gamma-3",
        "Key-Delta-4",
        "Key-Epsilon-5",
        "Key-Zeta-6",
    ]
    for idx, lbl in enumerate(extra_labels, start=3):
        await test_db.add_key(key_value=f"AQ.Ab8RN6MockKey_{idx:03d}_{lbl}", label=lbl)

    all_keys = await test_db.get_all_keys()
    assert len(all_keys) == 13

    # 2. Run 50 concurrent requests
    total_requests = 50
    concurrency = 10
    semaphore = asyncio.Semaphore(concurrency)
    payload: dict[str, object] = {
        "contents": [{"parts": [{"text": "Testing 13 keys distribution"}]}],
        "dry_run": True,
    }

    start_wall = time.perf_counter()
    tasks = [
        asyncio.create_task(
            send_single_request(
                client=async_client,
                semaphore=semaphore,
                url="/v1beta/models/gemini-3.1-flash-lite:generateContent?dry_run=true",
                headers={"Content-Type": "application/json"},
                payload=payload,
            )
        )
        for _ in range(total_requests)
    ]
    metrics = await asyncio.gather(*tasks)
    elapsed_total = time.perf_counter() - start_wall

    assert len(metrics) == 50
    successes = [m for m in metrics if m.success]
    assert len(successes) == 50

    # Key distribution
    key_labels_used = {m.key_label for m in metrics}
    # All 13 keys should be utilized under least-loaded rotation
    assert len(key_labels_used) >= 10

    # Percentiles
    latencies = [m.latency_ms for m in metrics]
    p50 = calculate_percentile(latencies, 50.0)
    p95 = calculate_percentile(latencies, 95.0)
    assert p50 > 0.0
    assert p95 >= p50

    # Verify report prints properly without error
    print_report(
        total_requests=total_requests,
        concurrency=concurrency,
        elapsed_total_sec=elapsed_total,
        metrics=list(metrics),
    )






