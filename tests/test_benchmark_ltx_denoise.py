import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ltx_denoise as bench


def test_kernels_are_grouped_by_kind():
    assert bench.kernel_kind("void qk_int_sv_f8_attn_kernel<128u, 64u>") == "attention"
    assert bench.kernel_kind("cudnn_generated_fort_native_sdpa_sm80_flash_fprop") == "attention"
    assert bench.kernel_kind("void cutlass::Kernel2<cutlass_80_tensorop_i16832gemm_s8>") == "gemm"
    assert bench.kernel_kind("triton_poi_fused__fused_rms_norm_addcmul_sage_attention_slice_8") == "fused triton"
    assert bench.kernel_kind("nvjet_tst_192x192_64x3_2x1_v_bz_TNT") == "gemm"
    assert bench.kernel_kind("triton_tem_fused__int_mm_mul_0") == "gemm"
    assert bench.kernel_kind("triton_poi_fused_abs_amax_round_1") == "fused triton"
    assert bench.kernel_kind("Memcpy HtoD (Pinned -> Device)") == "other"


def test_inductor_flags_are_set_from_literals(monkeypatch):
    monkeypatch.setattr(torch._inductor.config, "force_fuse_int_mm_with_mul", False)
    monkeypatch.setattr(torch._inductor.config.triton, "cudagraphs", False)
    bench.set_inductor_flags(["force_fuse_int_mm_with_mul=True", "triton.cudagraphs=True"])
    assert torch._inductor.config.force_fuse_int_mm_with_mul is True
    assert torch._inductor.config.triton.cudagraphs is True
