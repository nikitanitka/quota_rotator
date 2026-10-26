"""High-performance stress test client for Gemini Key Pool Router.

Simulates concurrent and periodic requests to benchmark latency, connection pooling,
and dynamic key pool rotation under heavy load.
Supports CLI parameters --url, --dry-run, --requests, --duration, and --model.
"""

import argparse
import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Final
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("stress_test")

DEFAULT_BASE_URL: Final[str] = "http://localhost:8000"
DEFAULT_MODEL_NAME: Final[str] = "gemini-3.1-flash-lite"
DEFAULT_TOTAL_REQUESTS: Final[int] = 40
DEFAULT_TOTAL_DURATION_SEC: Final[float] = 60.0

DEFAULT_PAYLOAD: Final[dict[str, list[dict[str, list[dict[str, str]]]]]] = {
    "contents": [{"parts": [{"text": "Ping. Reply with pong."}]}]
}


@dataclass(slots=True)
class RequestResult:
    index: int
    status_code: int | None
    latency_ms: float
    success: bool
    error_message: str | None = None


async def send_request(
    client: httpx.AsyncClient,
    endpoint: str,
    headers: dict[str, str],
    request_id: int,
) -> RequestResult:
    """Send a single inference request to the router proxy endpoint and record latency."""
    start_time = time.perf_counter()
    try:
        response = await client.post(endpoint, json=DEFAULT_PAYLOAD, headers=headers)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        success = response.status_code == 200

        if not success:
            logger.warning(
                f"[{request_id:02d}] HTTP {response.status_code} in {latency_ms:.1f}ms: {response.text[:100]}"
            )
        else:
            logger.info(f"[{request_id:02d}] HTTP 200 in {latency_ms:.1f}ms")

        return RequestResult(
            index=request_id,
            status_code=response.status_code,
            latency_ms=latency_ms,
            success=success,
        )
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(f"[{request_id:02d}] Network error: {type(exc).__name__}: {exc}")
        return RequestResult(
            index=request_id,
            status_code=None,
            latency_ms=latency_ms,
            success=False,
            error_message=str(exc),
        )


def compute_percentile(sorted_data: list[float], percentile: float) -> float:
    """Calculate the p-th percentile from an ordered list of floats using linear interpolation."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * d


def print_summary(
    results: list[RequestResult], total_elapsed: float, target_duration: float
) -> None:
    """Output comprehensive benchmarking summary and latency percentiles."""
    latencies = sorted(r.latency_ms for r in results if r.success)
    successful = len(latencies)
    failed = len(results) - successful

    status_counts: dict[str, int] = {}
    for r in results:
        key = str(r.status_code) if r.status_code is not None else "Error"
        status_counts[key] = status_counts.get(key, 0) + 1

    logger.info("=" * 45)
    logger.info("STRESS TEST SUMMARY")
    logger.info(f"Total Sent:      {len(results)} requests")
    logger.info(f"Duration:        {total_elapsed:.2f}s (target: {target_duration:.2f}s)")
    pct_success = (successful / len(results) * 100) if results else 0.0
    logger.info(f"Successful:      {successful} ({pct_success:.1f}%)")
    logger.info(f"Failed:          {failed}")
    logger.info(f"Status codes:    {status_counts}")

    if latencies:
        logger.info(f"Min Latency:     {latencies[0]:.1f} ms")
        logger.info(f"Median (p50):    {compute_percentile(latencies, 50):.1f} ms")
        logger.info(f"p95 Latency:     {compute_percentile(latencies, 95):.1f} ms")
        logger.info(f"p99 Latency:     {compute_percentile(latencies, 99):.1f} ms")
        logger.info(f"Max Latency:     {latencies[-1]:.1f} ms")
    logger.info("=" * 45)


async def run_stress_test(
    base_url: str,
    model_name: str,
    total_requests: int,
    total_duration_sec: float,
    dry_run: bool,
) -> list[RequestResult]:
    """Execute the stress test against the router."""
    request_interval = total_duration_sec / total_requests if total_requests > 0 else 0.0
    endpoint = f"{base_url.rstrip('/')}/v1beta/models/{model_name}:generateContent"
    if dry_run:
        endpoint = f"{endpoint}?dry_run=true"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if dry_run:
        headers["X-Dry-Run"] = "true"

    logger.info(
        f"Starting test: {total_requests} requests over {total_duration_sec}s "
        f"(1 req every {request_interval:.2f}s, dry_run={dry_run}) to {endpoint}"
    )

    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
    timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)

    start_bench = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks: list[asyncio.Task[RequestResult]] = []
        for i in range(1, total_requests + 1):
            tasks.append(
                asyncio.create_task(send_request(client, endpoint, headers, i))
            )
            if i < total_requests and request_interval > 0:
                await asyncio.sleep(request_interval)

        results = await asyncio.gather(*tasks)

    total_elapsed = time.perf_counter() - start_bench
    print_summary(list(results), total_elapsed, total_duration_sec)
    return list(results)


def main() -> None:
    """CLI entrypoint with argument parsing."""
    parser = argparse.ArgumentParser(description="Gemini Key Pool Router Stress Test")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Base URL of router (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Target Gemini model (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_TOTAL_REQUESTS,
        help=f"Total requests to send (default: {DEFAULT_TOTAL_REQUESTS})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_TOTAL_DURATION_SEC,
        help=f"Total benchmark duration in seconds (default: {DEFAULT_TOTAL_DURATION_SEC})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute requests in simulated dry-run mode",
    )

    args = parser.parse_args()
    asyncio.run(
        run_stress_test(
            base_url=args.url,
            model_name=args.model,
            total_requests=args.requests,
            total_duration_sec=args.duration,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()

