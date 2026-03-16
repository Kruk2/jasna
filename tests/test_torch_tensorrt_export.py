import os
from unittest.mock import patch

from jasna.trt.torch_tensorrt_export import engine_system_suffix, engine_precision_name


class TestEngineSystemSuffix:
    def test_windows(self):
        with patch.object(os, "name", "nt"):
            assert engine_system_suffix() == ".win"

    def test_linux(self):
        with patch.object(os, "name", "posix"):
            assert engine_system_suffix() == ".linux"


class TestEnginePrecisionName:
    def test_fp16(self):
        assert engine_precision_name(fp16=True) == "fp16"

    def test_fp32(self):
        assert engine_precision_name(fp16=False) == "fp32"
