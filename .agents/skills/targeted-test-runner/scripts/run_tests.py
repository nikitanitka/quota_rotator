#!/usr/bin/env python3
import json
import subprocess
import sys

cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--no-header"]
proc = subprocess.run(cmd, capture_output=True, text=True)

ok = proc.returncode == 0
output_lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()

print(
    json.dumps(
        {
            "ok": ok,
            "exit_code": proc.returncode,
            "summary": output_lines[-1] if output_lines else "No tests found",
            "failure_details": output_lines[-25:] if not ok else [],
        },
        ensure_ascii=False,
        indent=2,
    )
)
sys.exit(0 if ok else 1)
