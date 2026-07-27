"""Bake the FP8-calibrated upsample ONNX shipped as a model_weights asset.

Dev-side only (needs nvidia-modelopt[onnx]). Exports the BasicVSR++ upsample
wrapper as plain ONNX, runs modelopt FP8 PTQ with real clip activations, and
writes standard QuantizeLinear/DequantizeLinear nodes with baked scales. User
machines compile the result with the plain TensorRT ONNX parser — no modelopt.

    ~/.virtualenvs/model-opt/bin/python scripts/export_upsample_fp8_onnx.py \
        --calib-dir <calib-clip-dir>
"""

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(REPO_ROOT / "model_weights" / "lada_mosaic_restoration_model_generic_v1.2.pth"))
    parser.add_argument("--calib-dir", required=True, help="dir with clip_*.pt calibration clips (see quantize_basicvsrpp.py synth/capture)")
    parser.add_argument("--calib-clips", type=int, default=8)
    parser.add_argument("--calib-calls", type=int, default=16)
    parser.add_argument("--out", default=None, help="default: <weights stem>_upsample_fp8.onnx next to weights")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import numpy as np
    import torch

    from jasna.engine_paths import get_basicvsrpp_fp8_upsample_onnx_path
    from quantize_basicvsrpp import (
        FEATURE_SIZE,
        MID_CHANNELS,
        _build_wrappers,
        _capture_wrapper_inputs,
        _load_clips,
        _load_generator,
    )

    device = torch.device(args.device)
    out_path = args.out or get_basicvsrpp_fp8_upsample_onnx_path(args.weights)

    _model, gen = _load_generator(args.weights, device, fp16=False)
    wrappers = _build_wrappers(gen, device, torch.float32)
    clips = _load_clips(Path(args.calib_dir), args.calib_clips)
    captured = _capture_wrapper_inputs(
        gen, wrappers, clips, device, torch.float32, ["upsample"], args.calib_calls,
    )
    calib = np.concatenate(
        [t[0].float().cpu().numpy() for t in captured["upsample"]], axis=0,
    )[:256]

    with tempfile.TemporaryDirectory() as tmp:
        plain = str(Path(tmp) / "upsample_plain.onnx")
        sample = torch.randn(30, 5 * MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE, device=device)
        with torch.no_grad():
            torch.onnx.export(
                wrappers["upsample"], (sample,), plain, opset_version=17, dynamo=False,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        from modelopt.onnx.quantization import quantize

        quantize(plain, quantize_mode="fp8", calibration_data={"input": calib},
                 output_path=str(out_path), simplify=False)
    print(f"FP8 QDQ ONNX written to {out_path} (calibrated on {calib.shape[0]} activations)")


if __name__ == "__main__":
    main()
