from __future__ import annotations

from pathlib import Path

import torch


def test_compile_clip10_compiles_requested_engine(monkeypatch, tmp_path: Path) -> None:
    import jasna.models.basicvsrpp.inference as inf
    import jasna.restorer.basicvrspp_tenorrt_compilation as comp

    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(comp, "_get_approx_max_tensorrt_clip_length", lambda _dev: (8.0, 300))
    monkeypatch.setattr(inf, "load_model", lambda *_a, **_k: object())

    created: list[Path] = []

    def _fake_compile(_model, _device, _dtype, output_path: str, _max_clip: int) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("engine", encoding="utf-8")
        created.append(p)
        return str(p)

    monkeypatch.setattr(comp, "_compile_basicvsrpp_model", _fake_compile)

    out = comp.compile_mosaic_restoration_model(
        mosaic_restoration_model_path=str(Path("model_weights") / "lada_mosaic_restoration_model_generic_v1.2.pth"),
        clip_length=10,
        device=torch.device("cuda:0"),
        fp16=True,
        interactive=False,
    )

    assert Path(out).suffix == ".engine"
    assert Path(out).is_file()
    assert len(created) == 1
    assert created[0] == Path(out)

