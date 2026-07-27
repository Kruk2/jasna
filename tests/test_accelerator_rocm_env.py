"""ROCm-only environment defaults applied when jasna.accelerator is imported."""
from jasna.accelerator import apply_rocm_env_defaults

ROCM_DEFAULTS = {
    "MIOPEN_FIND_MODE": "FAST",
    "PYTORCH_ALLOC_CONF": "expandable_segments:False",
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:False",
}


def test_rocm_pins_the_allocator_and_miopen_find_mode():
    environ: dict[str, str] = {}
    apply_rocm_env_defaults(environ)
    assert environ == ROCM_DEFAULTS


def test_a_user_setting_wins():
    environ = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    apply_rocm_env_defaults(environ)
    assert environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert environ["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:False"
