"""Ensure tensorrt_libs DLLs are loaded before nvvfx can override them."""
import tensorrt_libs  # noqa: F401 — locks in tensorrt_libs nvinfer_10.dll before nvvfx
