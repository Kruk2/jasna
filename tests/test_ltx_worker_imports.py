"""The LTX worker must not load OpenCV: the release LTX worker fails to start with it."""
from __future__ import annotations

import subprocess
import sys


def test_ltx_worker_backend_import_does_not_load_cv2():
    code = "import sys\nfrom jasna.ltx.inference import WorkerBackend\nassert 'cv2' not in sys.modules\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
