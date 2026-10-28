"""Stress Testing Script for Gemini Key Pool Router.

Simulates concurrent load against the Gemini Key Pool Router,
evaluating key rotation, latency distribution (p50, p95, max),
and outgoing Google API key utilization across the key pool.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import NamedTuple
import httpx


class RequestMetrics(NamedTuple):
    """Metrics collected from a single request execution."""

    success: bool
    status_code: int
    latency_ms: float
    key_label: str
    error_message: str | None = None


def resolve_default_client_key() -> str:
    """Resolve client key from env, local database, or fallback constant."""
    env_key = os.getenv("ROUTER_CLIENT_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    db_path = os.getenv("DB_PATH", "data/router.db")
    if os.path.exists(db_path):
        try:
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT key_value FROM client_keys WHERE status = 'ACTIVE' ORDER BY id DESC LIMIT 1"
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return str(row[0]).strip()
        except Exception:
            pass

    return "gkr-live-3f5db109a982efcc492c9f51f5fb0585"


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate the given percentile (0-100) from a list of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    d = k - f
    return sorted_vals[f] * (1.0 - d) + sorted_vals[c] * d


async def send_single_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
) -> RequestMetrics:
    """Send a single generateContent request and collect execution metrics."""
    async with semaphore:
        start_time = time.perf_counter()
        try:
            resp = await client.post(url, headers=headers, json=payload)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = resp.status_code
            success = 200 <= status_code < 300

            key_label = resp.headers.get("x-selected-key-label", "").strip()

            error_msg: str | None = None
            if not success:
                error_msg = f"HTTP {status_code}: {resp.text[:80]}"
            else:
                if not key_label:
                    try:
                        data = resp.json()
                        text = ""
                        if isinstance(data, dict):
                            cands = data.get("candidates")
                            if isinstance(cands, list) and cands:
                                first_cand = cands[0]
                                if isinstance(first_cand, dict):
                                    content = first_cand.get("content")
                                    if isinstance(content, dict):
                                        parts = content.get("parts")
                                        if isinstance(parts, list) and parts:
                                            p0 = parts[0]
                                            if isinstance(p0, dict):
                                                text = str(p0.get("text", ""))
                        if "via " in text and "]" in text:
                            key_label = text.split("via ")[1].split("]")[0].strip()
                    except (json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
                        error_msg = f"Failed to parse key from text: {exc}"

            if not key_label:
                key_label = "Unassigned / Fallback"

            return RequestMetrics(
                success=success,
                status_code=status_code,
                latency_ms=latency_ms,
                key_label=key_label,
                error_message=error_msg,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return RequestMetrics(
                success=False,
                status_code=0,
                latency_ms=latency_ms,
                key_label="Error (Exception)",
                error_message=str(exc),
            )


def print_report(
    total_requests: int,
    concurrency: int,
    elapsed_total_sec: float,
    metrics: list[RequestMetrics],
    client_key: str | None = None,
) -> None:
    """Print beautifully formatted stress test execution summary and key distribution."""
    cyan = "\033[96m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    bold = "\033[1m"
    dim = "\033[2m"
    reset = "\033[0m"

    success_count = sum(1 for m in metrics if m.success)
    failure_count = len(metrics) - success_count
    success_rate = (success_count / total_requests * 100.0) if total_requests > 0 else 0.0

    latencies = [m.latency_ms for m in metrics]
    p50 = calculate_percentile(latencies, 50.0)
    p95 = calculate_percentile(latencies, 95.0)
    max_lat = max(latencies) if latencies else 0.0
    min_lat = min(latencies) if latencies else 0.0
    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
    rps = (total_requests / elapsed_total_sec) if elapsed_total_sec > 0 else 0.0

    key_counts: dict[str, int] = {}
    for m in metrics:
        key_counts[m.key_label] = key_counts.get(m.key_label, 0) + 1

    status_counts: dict[int, int] = {}
    for m in metrics:
        status_counts[m.status_code] = status_counts.get(m.status_code, 0) + 1

    print("\n" + bold + cyan + "═" * 70 + reset)
    print(bold + cyan + "       GEMINI KEY POOL ROUTER - STRESS TEST REPORT (13 KEYS POOL)       " + reset)
    print(bold + cyan + "═" * 70 + reset)

    key_display = f"{client_key[:12]}...{client_key[-4:]}" if client_key and len(client_key) > 16 else (client_key or "None")
    print(f"\n{bold}Configuration & Overview:{reset}")
    print(f"  • Total Requests Sent : {bold}{total_requests}{reset}")
    print(f"  • Concurrency Level   : {bold}{concurrency}{reset}")
    print(f"  • Total Wall Time     : {bold}{elapsed_total_sec:.2f}s{reset}")
    print(f"  • Throughput          : {bold}{rps:.2f} req/sec{reset}")
    print(f"  • Ingress Auth Key    : {bold}{cyan}{key_display}{reset}")

    rate_color = green if success_rate >= 95.0 else (yellow if success_rate >= 80.0 else red)
    print(f"\n{bold}Execution Outcomes:{reset}")
    print(f"  • Successful Responses: {green}{success_count}{reset}")
    print(f"  • Failed Responses    : {red if failure_count > 0 else green}{failure_count}{reset}")
    print(f"  • Success Rate        : {bold}{rate_color}{success_rate:.1f}%{reset}")

    if failure_count > 0:
        print(f"\n{bold}Failure Samples:{reset}")
        samples = [m.error_message for m in metrics if not m.success and m.error_message][:3]
        for s in samples:
            print(f"  • {red}{s}{reset}")

    print(f"\n{bold}Latency Profile (ms):{reset}")
    print(f"  • Min Latency         : {min_lat:.1f} ms")
    print(f"  • Mean Latency        : {mean_lat:.1f} ms")
    print(f"  • p50 (Median)        : {bold}{green}{p50:.1f} ms{reset}")
    print(f"  • p95                 : {bold}{yellow}{p95:.1f} ms{reset}")
    print(f"  • Max Latency         : {bold}{red if max_lat > 2000 else yellow}{max_lat:.1f} ms{reset}")

    print(f"\n{bold}HTTP Status Breakdown:{reset}")
    for code, cnt in sorted(status_counts.items()):
        sc_color = green if 200 <= code < 300 else (yellow if code == 429 else red)
        pct = (cnt / total_requests * 100.0) if total_requests > 0 else 0.0
        print(f"  • [{sc_color}{code}{reset}] : {cnt:3d} requests ({pct:5.1f}%)")

    print(f"\n{bold}Outgoing Google API Keys Distribution:{reset}")
    print(f"  ┌───────────────────────────┬───────────────┬────────────┐")
    print(f"  │ {bold}{'Key Label':<25}{reset} │ {bold}{'Calls Served':<13}{reset} │ {bold}{'Share (%)':<10}{reset} │")
    print(f"  ├───────────────────────────┼───────────────┼────────────┤")
    for key_label, count in sorted(key_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_requests * 100.0) if total_requests > 0 else 0.0
        bar = "█" * int(pct / 5.0)
        print(f"  │ {key_label:<25} │ {count:13d} │ {pct:9.1f}% {dim}{bar}{reset}│")
    print(f"  └───────────────────────────┴───────────────┴────────────┘")
    print(bold + cyan + "═" * 70 + reset + "\n")


async def run_stress_test(
    url: str,
    total_requests: int = 50,
    concurrency: int = 10,
    client_key: str | None = None,
    dry_run: bool = True,
) -> list[RequestMetrics]:
    """Execute load test against router and return collected metrics."""
    semaphore = asyncio.Semaphore(concurrency)
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if client_key:
        headers["Authorization"] = f"Bearer {client_key.strip()}"

    target_url = url
    if dry_run and "dry_run=" not in target_url:
        delimiter = "&" if "?" in target_url else "?"
        target_url = f"{target_url}{delimiter}dry_run=true"

    payload: dict[str, object] = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Explain quantum computing and superpositions in 1 sentence."
                    }
                ]
            }
        ],
        "dry_run": dry_run,
    }

    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=10.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        start_wall = time.perf_counter()
        tasks = [
            asyncio.create_task(
                send_single_request(
                    client=client,
                    semaphore=semaphore,
                    url=target_url,
                    headers=headers,
                    payload=payload,
                )
            )
            for _ in range(total_requests)
        ]
        results = await asyncio.gather(*tasks)
        elapsed_total = time.perf_counter() - start_wall

    print_report(
        total_requests=total_requests,
        concurrency=concurrency,
        elapsed_total_sec=elapsed_total,
        metrics=list(results),
        client_key=client_key,
    )
    return list(results)


def main() -> None:
    """CLI Entry point for stress testing Gemini Key Pool."""
    parser = argparse.ArgumentParser(
        description="Stress testing tool for Gemini Key Pool Router (13 Keys Pool)."
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default="http://localhost:8000/v1beta/models/gemini-3.1-flash-lite:generateContent",
        help="Router endpoint URL (default: http://localhost:8000/v1beta/models/gemini-3.1-flash-lite:generateContent)",
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=50,
        help="Total number of requests to execute (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Concurrency level (default: 10)",
    )
    parser.add_argument(
        "--key",
        "-k",
        type=str,
        default=resolve_default_client_key(),
        help="Client API key for router authentication (default: auto-detected from DB or ROUTER_CLIENT_KEY)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute real upstream calls instead of dry-run simulation",
    )

    args = parser.parse_args()
    asyncio.run(
        run_stress_test(
            url=args.url,
            total_requests=args.requests,
            concurrency=args.concurrency,
            client_key=args.key,
            dry_run=not args.live,
        )
    )


if __name__ == "__main__":
    main()
