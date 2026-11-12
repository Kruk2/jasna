# Python 3.10 compatibility (AutoDL / MTT S4000)

This branch makes **a compatibility attempt** for Python 3.10, keeping the Python 3.11+ packaging dependencies unchanged. The original upstream release targets Python 3.12+; this branch is **not yet a validated full-featured port**.

## Package changes

- Accept Python 3.10 in `pyproject.toml`.
- Use `av==16.1.0` for Python 3.10 (PyAV 18.1 requires Python >=3.11).
- Limit `diffusers<0.42` on Python 3.10 (0.42+ requires Python >=3.11).
- Provide a Python 3.10 fallback for `enum.StrEnum`.
- Add a Python 3.10 compile/build metadata check to CI.

**Video/VALI caveat:** Jasna's original PyAV 18.1 pin matches a release VALI backend. PyAV 16.1 does **not** guarantee ABI compatibility with that VALI binary. Use PyAV decoding or a matching VALI build and test the selected decoder.

## AutoDL recommended workflow

Avoid uninstalling the image's system MUSA or replacing its Python; the original image's system Python 3.10 already has PyTorch MUSA installed.

```bash
/usr/bin/python3 -c "import torch, torch_musa; print(torch.__version__)"
/usr/bin/python3 -m pip install -e . --no-deps
```

The `--no-deps` form is for controlled provisioning, **not a complete install**: supply suitable Python 3.10 versions of `mmengine==0.10.7`, `numpy`, `opencv-python`, `Pillow`, `safetensors`, `tqdm`, and any packages the chosen entry point imports. Running `pip install -e .` normally may try to resolve a new GPU `torch` through dependencies; verify the wheel selection before installing.

For BasicVSR++ specifically, start with an import test, then test `torchvision.ops.deform_conv2d` and the actual model in a **GPU-attached** session. AutoDL's no-GPU mode may show `torch.musa.is_available() == False` even when the Python package imports.

This branch **does not** implement MUSA vendor dispatch throughout Jasna, enable NVIDIA CUDA/ROCm-only kernels on MUSA, or repair unrelated `libmccl.so`/torch_musa ABI mismatches. These are separate from Python-version support.
