import torch
import logging

logger = logging.getLogger(__name__)

def _chw_rgb_to_p010_bt709_limited_fallback(img_chw: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch fallback implementation."""
    C, H, W = img_chw.shape
    assert C == 3 and H % 2 == 0 and W % 2 == 0
    
    if img_chw.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        img_chw = img_chw.float() / 255.0
    
    R = img_chw[0].float()
    G = img_chw[1].float()
    B = img_chw[2].float()

    # 10-bit BT.709 limited range: Y: 64-940, U/V: 64-960
    Yf = 64.0 + 876.0 * (0.2126 * R + 0.7152 * G + 0.0722 * B)
    Uf = 512.0 + 896.0 * (-0.114572 * R - 0.385428 * G + 0.500000 * B)
    Vf = 512.0 + 896.0 * (0.500000 * R - 0.454153 * G - 0.045847 * B)

    # Clamp and shift left by 6 for P010 (10-bit in upper bits of 16-bit)
    Y = (Yf.round().clamp(64, 940) * 64).to(torch.int16)

    # Subsample UV (4:2:0)
    U00, U01, U10, U11 = Uf[0::2, 0::2], Uf[0::2, 1::2], Uf[1::2, 0::2], Uf[1::2, 1::2]
    V00, V01, V10, V11 = Vf[0::2, 0::2], Vf[0::2, 1::2], Vf[1::2, 0::2], Vf[1::2, 1::2]

    U_ds = (((U00 + U01 + U10 + U11) * 0.25).round().clamp(64, 960) * 64).to(torch.int16)
    V_ds = (((V00 + V01 + V10 + V11) * 0.25).round().clamp(64, 960) * 64).to(torch.int16)

    # Interleave U and V
    uv = torch.stack((U_ds, V_ds), dim=-1).reshape(U_ds.shape[0], -1)

    return torch.cat([Y, uv], dim=0).contiguous()


def chw_rgb_to_p010_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010_bt709_limited_fallback(img_chw)
