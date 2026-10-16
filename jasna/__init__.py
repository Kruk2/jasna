__all__ = ["__version__"]

__version__ = "0.11.1"
# Local test build (F tree only). Contents so far:
# - long-run VRAM consolidation - a light cold-cache release every
#   JASNA_VRAM_LIGHT_VIDEO_S of restored video (default 20 min) and a deep sweep
#   every JASNA_VRAM_DEEP_VIDEO_S (default 1 h);
# - mosaic-leak fix at stream/segment end (short trailing clip is no longer
#   dropped as a false positive);
# - AMD decode defaults to FFmpeg software (measured ~1.95x faster than AMF for
#   H.264; hevc_amf cannot decode Main10 at all). AMF is still reachable with
#   JASNA_DECODE_BACKEND=pyav-hw;
# - the AMD panel "sharpness" control also drives a patch-CAS sharpen on the
#   network engines (real-esr / realesrgan), scoped to the mosaic patch only;
# - secondary models: 2xParimgCompact added to the real-esr family; the unused
#   x4plus_anime_6B checkpoint was dropped.
# The "-local.N" suffix is a local build number - bump it per test build. Do NOT
# carry this suffix into a PR/upstream; upstream is plain "0.11.0".
__version__ = "0.11.0-local.5"
