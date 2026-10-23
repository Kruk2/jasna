"""Opt-in driver check: fresh HIP allocations must not contain a killed process's marker.

Run on an idle AMD GPU with JASNA_TEST_AMD_VRAM_CLEANUP=1. This checks the
installed driver on the selected GPU.
"""
import json
import os
import subprocess
import sys
import time

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("JASNA_TEST_AMD_VRAM_CLEANUP") != "1", reason="exclusive AMD GPU cleanup qualification"
)

PROGRAM = r'''
import ctypes
import json
from pathlib import Path
import sys
import torch
from jasna.media.cuda_kernel import hip_runtime, check_hip

assert torch.version.hip is not None
torch.cuda.init()
hip = hip_runtime()
signatures = {
    "hipMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
    "hipMemset": [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
    "hipMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    "hipFree": [ctypes.c_void_p],
    "hipDeviceSynchronize": [],
}
for name, arguments in signatures.items():
    function = getattr(hip, name)
    function.argtypes = arguments
    function.restype = ctypes.c_int

mode, report_path = sys.argv[1:]
report_path = Path(report_path)
chunk = 64 * 1024**2
if mode == "victim":
    free, total = torch.cuda.mem_get_info()
    count = min(free - 1024**3, total * 4 // 5) // chunk
    assert count >= 16, "not enough free VRAM for cleanup qualification"
else:
    count = json.loads(report_path.read_text())["chunks"]

allocations = []
try:
    for _ in range(count):
        pointer = ctypes.c_void_p()
        check_hip(hip.hipMalloc(ctypes.byref(pointer), chunk), "allocate cleanup probe")
        allocations.append(pointer)
        if mode == "victim":
            check_hip(hip.hipMemset(pointer, 0xa7, chunk), "write synthetic marker")
    check_hip(hip.hipDeviceSynchronize(), "finish cleanup allocations")
    if mode == "victim":
        pending = report_path.with_suffix(".tmp")
        pending.write_text(json.dumps({"chunks": count, "bytes": count * chunk}))
        pending.replace(report_path)
        sys.stdin.read()
    else:
        host = ctypes.create_string_buffer(chunk)
        matches = 0
        for pointer in allocations:
            check_hip(hip.hipMemcpy(host, pointer, chunk, 2), "read fresh allocation")
            matches += host.raw.find(b"\xa7" * 32) >= 0
        print(json.dumps({"scanned_bytes": count * chunk, "marker_chunks": matches}))
finally:
    for pointer in allocations:
        check_hip(hip.hipFree(pointer), "free cleanup probe")
'''


def test_new_hip_allocations_do_not_expose_a_killed_process(tmp_path):
    ready = tmp_path / "ready.json"
    victim = subprocess.Popen(
        [sys.executable, "-c", PROGRAM, "victim", str(ready)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path,
    )
    try:
        deadline = time.monotonic() + 120
        while not ready.exists() and victim.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert ready.exists(), "VRAM marker process did not become ready"
        victim.kill()
        victim.wait(timeout=30)
        assert victim.returncode != 0
        scanner = subprocess.run(
            [sys.executable, "-c", PROGRAM, "scanner", str(ready)],
            capture_output=True, text=True, timeout=180, cwd=tmp_path,
        )
        assert scanner.returncode == 0, scanner.stdout + scanner.stderr
        result = json.loads(scanner.stdout)
        assert result["scanned_bytes"] == json.loads(ready.read_text())["bytes"]
        assert result["marker_chunks"] == 0, result
        print(json.dumps(result))
    finally:
        if victim.poll() is None:
            victim.kill()
        output, errors = victim.communicate(timeout=30)
        if errors:
            print(output + errors)
