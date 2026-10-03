"""Synthetic Windows RF-DETR precision diagnostics, not a quality certificate.

Uses the product Math SDPA entry point. Permutation-invariant box/class matching
separates proposal ordering from numerical differences; it cannot certify real
mosaic detection/restoration quality or new/old performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def compare_proposals(reference, candidate, *, rtol=.001, atol=.001, threshold=.35):
    from scipy.optimize import linear_sum_assignment

    names = ("dets", "labels", "masks")
    a = {name: reference[name].detach().cpu().float() for name in names}
    b = {name: candidate[name].detach().cpu().float() for name in names}
    if any(a[name].shape != b[name].shape for name in names):
        raise ValueError("Proposal shapes changed; comparison is not admissible")
    if any(not torch.isfinite(tensor).all() for tensor in [*a.values(), *b.values()]):
        raise ValueError("Nonfinite detector output")
    if any(tensor.ndim < 3 or tensor.shape[:2] != a["dets"].shape[:2]
           for tensor in [*a.values(), *b.values()]):
        raise ValueError("Expected B,Q detector outputs with shared query axes")
    if a["dets"].shape[-1] != 4 or a["dets"].shape[1] == 0:
        raise ValueError("Expected nonempty cxcywh proposal boxes")
    rows = []
    for batch in range(a["dets"].shape[0]):
        # Matching is a diagnostic cost, not an assertion that every selected
        # top-k proposal survives a precision change. Report missed active
        # proposals separately instead of treating every permutation as equal.
        cost = torch.cdist(a["dets"][batch], b["dets"][batch], p=1)
        cost += torch.cdist(a["labels"][batch].sigmoid(), b["labels"][batch].sigmoid(), p=1)
        left, right = linear_sum_assignment(cost.numpy())
        aa = {name: a[name][batch, left] for name in names}
        bb = {name: b[name][batch, right] for name in names}
        numerical = {name: dict(max_abs_error=float((aa[name] - bb[name]).abs().max()),
                                allclose=bool(torch.allclose(aa[name], bb[name], rtol=rtol, atol=atol)))
                     for name in names}
        def corners(box):
            return torch.cat((box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2), dim=1)
        x, y = corners(aa["dets"]), corners(bb["dets"])
        intersection = (torch.minimum(x[:, 2:], y[:, 2:]) - torch.maximum(x[:, :2], y[:, :2])).clamp_min(0).prod(-1)
        areas = (x[:, 2:] - x[:, :2]).clamp_min(0).prod(-1) + (y[:, 2:] - y[:, :2]).clamp_min(0).prod(-1)
        box_iou = intersection / (areas - intersection).clamp_min(1e-12)
        score_a, class_a = aa["labels"].sigmoid().max(-1)
        score_b, class_b = bb["labels"].sigmoid().max(-1)
        active_a, active_b = score_a > threshold, score_b > threshold
        ma, mb = aa["masks"] > 0, bb["masks"] > 0
        dimensions = tuple(range(1, ma.ndim))
        union = (ma | mb).sum(dim=dimensions)
        mask_iou = torch.where(union > 0, (ma & mb).sum(dim=dimensions) / union.clamp_min(1), 1.)
        matching = active_a & active_b & (class_a == class_b) & (box_iou >= .95) & (mask_iou >= .95)
        rows.append(dict(query_count=len(left), reordered_queries=int((left != right).sum()),
                         numerical=numerical, reference_active=int(active_a.sum()),
                         candidate_active=int(active_b.sum()), high_overlap_matches=int(matching.sum()),
                         reference_active_without_high_overlap=int((active_a & ~matching).sum()),
                         candidate_active_without_high_overlap=int((active_b & ~matching).sum())))
    return dict(rtol=rtol, atol=atol, score_threshold=threshold, diagnostic_overlap_threshold=.95,
                strict_matched_numbers=all(v["allclose"] for row in rows for v in row["numerical"].values()),
                batches=rows, quality_acceptance="NOT_CERTIFIED: synthetic/task overlap diagnostics only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4], choices=[1, 4])
    args = parser.parse_args()
    if sys.platform != "win32" or torch.__version__ != "2.12.0+rocm10.0.0" or not torch.version.hip:
        raise RuntimeError("This probe requires the isolated Windows Torch 2.12 ROCm 10 candidate")
    if args.output.exists():
        raise FileExistsError("Refusing to replace existing precision evidence")
    from jasna.media.hip_kernel import hip_runtime, hip_runtime_identity
    from jasna.mosaic.rfdetr_torch_runner import RfDetrTorchRunner
    import ctypes

    runtime = hip_runtime()
    peek = runtime.hipPeekAtLastError
    peek.argtypes, peek.restype = [], ctypes.c_int
    report = dict(schema="jasna.windows.rfdetr.precision.v1", source_kind="synthetic",
                  seed=20261003, torch=torch.__version__, hip_identity=hip_runtime_identity(), cases=[],
                  status="RUNNING", quality_acceptance="NOT_CERTIFIED", performance_acceptance="NOT_RUN",
                  note="CPU is an explicit numerical reference, never a native failure fallback")
    cpu, gpu = None, None
    try:
        weights = args.weights.resolve(strict=True)
        gpu = RfDetrTorchRunner(weights, [(4, 3, 576, 576)], torch.device("cuda:0"),
                               fp16=False, resolution=576, variant="medium")
        if gpu.sdpa_policy["resolved"] != "math":
            raise RuntimeError("The exact verified GPU Math SDPA policy was not selected")
        report["sdpa_policy"] = gpu.sdpa_policy
        cpu = RfDetrTorchRunner(weights, [(4, 3, 576, 576)], torch.device("cpu"),
                               fp16=False, resolution=576, variant="medium")
        for batch in args.batches:
            generator = torch.Generator().manual_seed(report["seed"])
            inputs = torch.randn((batch, 3, 576, 576), generator=generator)
            reference = cpu.infer({"input": inputs})
            outputs = []
            for fp16 in (False, True):
                gpu.fp16 = fp16  # Same checkpoint/model; only product autocast changes.
                output = gpu.infer({"input": inputs})
                torch.cuda.synchronize()
                status = int(peek())  # Does not clear/consume evidence.
                if status:
                    raise RuntimeError(f"HIP error state {status}; stop, preserve evidence, restart GPU process")
                outputs.append({name: tensor.detach().cpu().clone() for name, tensor in output.items()})
            report["cases"].append(dict(batch=batch, execution_fp32_fp16="PASS", hip_peek_status=0,
                cpu_fp32_vs_gpu_fp32=compare_proposals(reference, outputs[0]),
                gpu_fp32_vs_gpu_fp16=compare_proposals(outputs[0], outputs[1])))
        report["status"] = "EXECUTED: read numerical/task diagnostics; not release acceptance"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # Evidence even on failure. Do not synchronize or retry a poisoned GPU.
        for runner in (cpu, gpu):
            if runner is not None:
                runner.close()
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(str(args.output))


if __name__ == "__main__":
    main()
