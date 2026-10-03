from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from jasna.migraphx_artifact import (
    MigraphxArtifactManifest,
    MigraphxArtifactRunner,
    MigraphxTensorContract,
    _RuntimeBindings,
    migraphx_manifest_sha256,
)
from jasna.mosaic.rfdetr_migraphx_runner import RfDetrMigraphxRunner


class _FakeShape:
    def __init__(
        self,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: str = "float_type",
    ) -> None:
        self._shape = shape
        self._strides = strides
        self._dtype = dtype

    def lens(self):
        return self._shape

    def strides(self):
        return self._strides

    def type_string(self):
        return self._dtype


class _FakeProgram:
    def __init__(self, parameters: dict[str, _FakeShape], outputs: list[_FakeShape]) -> None:
        self.parameters = parameters
        self.outputs = outputs
        self.run_calls: list[tuple[dict[str, object], int, str]] = []

    def is_compiled(self) -> bool:
        return True

    def get_parameter_shapes(self):
        return self.parameters

    def get_output_shapes(self):
        return self.outputs

    def run_async(self, arguments, stream, stream_type):
        self.run_calls.append((dict(arguments), stream, stream_type))
        return object()


class _FakeDevice:
    type = "cuda"
    index = 0

    def __str__(self) -> str:
        return "cuda:0"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDevice)


class _FakeTensor:
    _next_pointer = 100

    def __init__(
        self,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: torch.dtype,
        device: _FakeDevice,
    ) -> None:
        self.shape = shape
        self._strides = strides
        self.dtype = dtype
        self.device = device
        self._pointer = _FakeTensor._next_pointer
        _FakeTensor._next_pointer += 1

    def stride(self):
        return self._strides

    def data_ptr(self) -> int:
        return self._pointer


class _FakeTorch:
    float16 = torch.float16
    float32 = torch.float32
    __version__ = "torch-test"
    version = SimpleNamespace(hip="hip-test")

    def __init__(self) -> None:
        self.device_value = _FakeDevice()
        self.allocations: list[_FakeTensor] = []
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _device: SimpleNamespace(gcnArchName="gfx-test"),
            current_stream=lambda _device: SimpleNamespace(cuda_stream=1234),
        )

    def device(self, _value) -> _FakeDevice:
        return self.device_value

    def empty_strided(self, shape, strides, *, dtype, device) -> _FakeTensor:
        tensor = _FakeTensor(tuple(shape), tuple(strides), dtype, device)
        self.allocations.append(tensor)
        return tensor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_dict(
    artifact: Path,
    source: Path,
    *,
    artifact_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "migraphx",
        "purpose": "test-purpose",
        "artifact": {
            "path": artifact.name,
            "sha256": artifact_sha256 or _sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        },
        "source": {"sha256": _sha256(source)},
        "target": {
            "platform": "linux",
            "device_arch": "gfx-test",
            "torch_version": "torch-test",
            "torch_hip_version": "hip-test",
            "migraphx_version": "migraphx-test",
            "torch_migraphx_version": "torch-migraphx-test",
        },
        "program": {
            "internal_precision": "fp16",
            "stream_type": "ihipStream_t",
            "inputs": [
                {
                    "logical_name": "input",
                    "parameter_name": "input",
                    "dtype": "float32",
                    "shape": [2, 3, 4, 4],
                    "strides": [48, 16, 4, 1],
                }
            ],
            "outputs": [
                {
                    "logical_name": "dets",
                    "parameter_name": "main:#output_0",
                    "dtype": "float32",
                    "shape": [2, 2, 4],
                    "strides": [8, 4, 1],
                },
                {
                    "logical_name": "labels",
                    "parameter_name": "main:#output_1",
                    "dtype": "float32",
                    "shape": [2, 2, 3],
                    "strides": [6, 3, 1],
                },
                {
                    "logical_name": "masks",
                    "parameter_name": "main:#output_2",
                    "dtype": "float32",
                    "shape": [2, 2, 2, 2],
                    "strides": [8, 4, 2, 1],
                },
            ],
        },
    }


def _write_fixture(tmp_path: Path):
    artifact = tmp_path / "model.mxr"
    source = tmp_path / "model.pt"
    artifact.write_bytes(b"compiled")
    source.write_bytes(b"weights")
    payload = _manifest_dict(artifact, source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, artifact, source, payload


def _fake_program(payload: dict[str, object]) -> _FakeProgram:
    program = payload["program"]
    contracts = [*program["inputs"], *program["outputs"]]
    parameters = {
        item["parameter_name"]: _FakeShape(tuple(item["shape"]), tuple(item["strides"]))
        for item in contracts
    }
    outputs = [parameters[item["parameter_name"]] for item in program["outputs"]]
    return _FakeProgram(parameters, outputs)


def _fake_bindings(program: _FakeProgram, fake_torch: _FakeTorch) -> _RuntimeBindings:
    def argument_from_ptr(pointer, shape):
        return (pointer, tuple(shape.lens()), tuple(shape.strides()))

    return _RuntimeBindings(
        migraphx=SimpleNamespace(load=lambda _path: program),
        mgx_argument_from_ptr=argument_from_ptr,
        tensors_from_mgx_arguments=lambda _raw, _shapes: tuple(fake_torch.allocations),
        migraphx_version="migraphx-test",
        torch_migraphx_version="torch-migraphx-test",
    )


def _install_fake_artifact_runtime(monkeypatch, payload: dict[str, object]) -> tuple[_FakeTorch, _FakeProgram]:
    import jasna.migraphx_artifact as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    fake_torch = _FakeTorch()
    program = _fake_program(payload)
    monkeypatch.setattr(module, "torch", fake_torch)
    monkeypatch.setattr(module, "_load_runtime_bindings", lambda: _fake_bindings(program, fake_torch))
    return fake_torch, program


def test_manifest_is_strict_and_exposes_sidecar_digest(tmp_path: Path) -> None:
    manifest_path, artifact, _source, payload = _write_fixture(tmp_path)

    manifest = MigraphxArtifactManifest.load(manifest_path)

    assert manifest.artifact_path == artifact
    assert manifest.inputs[0].shape == (2, 3, 4, 4)
    assert migraphx_manifest_sha256(manifest_path) == _sha256(manifest_path)

    payload["unexpected"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="root keys mismatch"):
        MigraphxArtifactManifest.load(manifest_path)


def test_manifest_rejects_invalid_hash_rank_and_empty_outputs(tmp_path: Path) -> None:
    manifest_path, _artifact, _source, payload = _write_fixture(tmp_path)
    payload["artifact"]["sha256"] = "not-a-sha"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lowercase SHA256"):
        MigraphxArtifactManifest.load(manifest_path)

    payload["artifact"]["sha256"] = "0" * 64
    payload["program"]["inputs"][0]["strides"] = [1]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="shape/stride rank mismatch"):
        MigraphxArtifactManifest.load(manifest_path)

    payload["program"]["inputs"][0]["strides"] = [48, 16, 4, 1]
    payload["program"]["outputs"] = []
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="at least one output"):
        MigraphxArtifactManifest.load(manifest_path)


def test_runtime_loader_discovers_abi_matched_official_rocm_binding(monkeypatch, tmp_path: Path) -> None:
    import jasna.migraphx_artifact as module

    rocm = tmp_path / "rocm"
    library_dir = rocm / "lib"
    library_dir.mkdir(parents=True)
    suffix = str(module.sysconfig.get_config_var("EXT_SUFFIX"))
    (library_dir / f"migraphx{suffix}").touch()
    monkeypatch.setenv("ROCM_PATH", str(rocm))

    fake_migraphx = SimpleNamespace(__version__="migraphx-test")
    fake_utils = SimpleNamespace(
        mgx_argument_from_ptr=object(),
        tensors_from_mgx_arguments=object(),
    )
    calls: list[tuple[str, str]] = []

    def import_module(name: str):
        calls.append((name, module.os.environ.get("PATH", "")))
        if name == "migraphx":
            if str(library_dir) not in module.sys.path:
                error = ModuleNotFoundError("missing migraphx")
                error.name = "migraphx"
                raise error
            return fake_migraphx
        if name == "torch_migraphx":
            return SimpleNamespace()
        if name == "torch_migraphx.fx.utils":
            return fake_utils
        raise AssertionError(name)

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    monkeypatch.setattr(
        module,
        "_package_version",
        lambda name: "torch-migraphx-test" if name == "torch-migraphx" else "unused",
    )

    bindings = module._load_runtime_bindings()

    assert bindings.migraphx is fake_migraphx
    assert bindings.mgx_argument_from_ptr is fake_utils.mgx_argument_from_ptr
    assert bindings.tensors_from_mgx_arguments is fake_utils.tensors_from_mgx_arguments
    assert calls[0][0] == "migraphx"
    assert calls[1][0] == "migraphx"
    expected_bin = str(Path(module.sys.executable).parent)
    torch_calls = [path for name, path in calls if name.startswith("torch_migraphx")]
    assert torch_calls and all(path.split(module.os.pathsep)[0] == expected_bin for path in torch_calls)


def test_pointer_runner_uses_owned_output_buffers_and_closes_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, _artifact, source, payload = _write_fixture(tmp_path)
    fake_torch, program = _install_fake_artifact_runtime(monkeypatch, payload)
    runner = MigraphxArtifactRunner(
        manifest_path,
        source_path=source,
        device=_FakeDevice(),
        expected_purpose="test-purpose",
    )
    input_tensor = _FakeTensor(
        (2, 3, 4, 4),
        (48, 16, 4, 1),
        torch.float32,
        fake_torch.device_value,
    )

    outputs = runner.infer({"input": input_tensor})

    assert list(outputs) == ["dets", "labels", "masks"]
    assert program.run_calls[0][1:] == (1234, "ihipStream_t")
    assert program.run_calls[0][0]["input"][0] == input_tensor.data_ptr()
    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.infer({"input": input_tensor})


def test_pointer_runner_rejects_foreign_output_and_input_dtype(tmp_path: Path, monkeypatch) -> None:
    import jasna.migraphx_artifact as module

    manifest_path, _artifact, source, payload = _write_fixture(tmp_path)
    fake_torch, program = _install_fake_artifact_runtime(monkeypatch, payload)

    def foreign_outputs(_raw, _shapes):
        return tuple(
            _FakeTensor(
                tuple(contract["shape"]),
                tuple(contract["strides"]),
                torch.float32,
                fake_torch.device_value,
            )
            for contract in payload["program"]["outputs"]
        )

    bindings = _fake_bindings(program, fake_torch)
    bindings = replace(bindings, tensors_from_mgx_arguments=foreign_outputs)
    monkeypatch.setattr(module, "_load_runtime_bindings", lambda: bindings)
    runner = MigraphxArtifactRunner(
        manifest_path,
        source_path=source,
        device=_FakeDevice(),
        expected_purpose="test-purpose",
    )
    wrong_dtype = _FakeTensor(
        (2, 3, 4, 4),
        (48, 16, 4, 1),
        torch.float16,
        fake_torch.device_value,
    )
    with pytest.raises(RuntimeError, match="input dtype mismatch"):
        runner.infer({"input": wrong_dtype})

    input_tensor = _FakeTensor(
        (2, 3, 4, 4),
        (48, 16, 4, 1),
        torch.float32,
        fake_torch.device_value,
    )
    with pytest.raises(RuntimeError, match="output pointer mismatch"):
        runner.infer({"input": input_tensor})


def test_pointer_runner_rejects_hash_source_runtime_and_program_abi(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import jasna.migraphx_artifact as module

    manifest_path, artifact, source, payload = _write_fixture(tmp_path)
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        MigraphxArtifactRunner(
            manifest_path,
            source_path=source,
            device=torch.device("cpu"),
            expected_purpose="test-purpose",
        )

    artifact.write_bytes(b"compiled")
    source.write_bytes(b"different weights")
    with pytest.raises(RuntimeError, match="source model SHA256 mismatch"):
        MigraphxArtifactRunner(
            manifest_path,
            source_path=source,
            device=torch.device("cpu"),
            expected_purpose="test-purpose",
        )

    source.write_bytes(b"weights")
    fake_torch, program = _install_fake_artifact_runtime(monkeypatch, payload)
    bad_bindings = replace(
        _fake_bindings(program, fake_torch),
        migraphx_version="wrong-version",
    )
    monkeypatch.setattr(module, "_load_runtime_bindings", lambda: bad_bindings)
    with pytest.raises(RuntimeError, match="runtime version mismatch"):
        MigraphxArtifactRunner(
            manifest_path,
            source_path=source,
            device=_FakeDevice(),
            expected_purpose="test-purpose",
        )

    good_bindings = _fake_bindings(program, fake_torch)
    program.parameters["input"] = _FakeShape((1, 3, 4, 4), (48, 16, 4, 1))
    monkeypatch.setattr(module, "_load_runtime_bindings", lambda: good_bindings)
    with pytest.raises(RuntimeError, match="tensor ABI mismatch"):
        MigraphxArtifactRunner(
            manifest_path,
            source_path=source,
            device=_FakeDevice(),
            expected_purpose="test-purpose",
        )

    program.parameters["input"] = _FakeShape((2, 3, 4, 4), (48, 16, 4, 1))
    program.outputs[0] = _FakeShape((2, 2, 4), (8, 1, 2))
    with pytest.raises(RuntimeError, match="output ABI mismatch"):
        MigraphxArtifactRunner(
            manifest_path,
            source_path=source,
            device=_FakeDevice(),
            expected_purpose="test-purpose",
        )


def _rfdetr_product_manifest(
    *,
    batch_size: int = 1,
    internal_precision: str = "mixed-dot-projector-convolution-fp16",
) -> SimpleNamespace:
    return SimpleNamespace(
        platform="linux",
        device_arch="gfx1100",
        internal_precision=internal_precision,
        stream_type="ihipStream_t",
        inputs=(
            MigraphxTensorContract(
                logical_name="input",
                parameter_name="input",
                dtype_name="float32",
                shape=(batch_size, 3, 576, 576),
                strides=(995328, 331776, 576, 1),
            ),
        ),
        outputs=(
            MigraphxTensorContract(
                logical_name="dets",
                parameter_name="main:#output_0",
                dtype_name="float32",
                shape=(batch_size, 200, 4),
                strides=(800, 4, 1),
            ),
            MigraphxTensorContract(
                logical_name="labels",
                parameter_name="main:#output_1",
                dtype_name="float32",
                shape=(batch_size, 200, 3),
                strides=(600, 3, 1),
            ),
            MigraphxTensorContract(
                logical_name="masks",
                parameter_name="main:#output_2",
                dtype_name="float32",
                shape=(batch_size, 200, 144, 144),
                strides=(4147200, 20736, 144, 1),
            ),
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rfdetr_runner_accepts_proven_static_contract(batch_size: int) -> None:
    RfDetrMigraphxRunner._validate_contract(
        SimpleNamespace(manifest=_rfdetr_product_manifest(batch_size=batch_size))
    )


@pytest.mark.parametrize(
    "internal_precision",
    [
        "fp32",
        "mixed-dot-fp16",
        "mixed-convolution-fp16",
        "mixed-backbone-linear-fp16",
    ],
)
def test_rfdetr_runner_rejects_unproven_precision(internal_precision: str) -> None:
    with pytest.raises(RuntimeError, match="outside the proven set"):
        RfDetrMigraphxRunner._validate_contract(
            SimpleNamespace(
                manifest=_rfdetr_product_manifest(
                    internal_precision=internal_precision
                )
            )
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("platform", "win32"),
        ("device_arch", "gfx1151"),
        ("input_parameter", "wrong_input"),
        ("input_shape", (1, 3, 288, 1152)),
        ("input_strides", (995328, 1, 1728, 3)),
        ("output_parameter", "wrong_output"),
        ("output_shape", (1, 200, 72, 288)),
        ("output_strides", (4147200, 1, 28800, 200)),
    ],
)
def test_rfdetr_runner_rejects_artifact_outside_proven_contract(
    mutation: str,
    value: object,
) -> None:
    manifest = _rfdetr_product_manifest()
    if mutation in {"platform", "device_arch"}:
        setattr(manifest, mutation, value)
    elif mutation == "input_parameter":
        manifest.inputs = (replace(manifest.inputs[0], parameter_name=value),)
    elif mutation == "input_shape":
        manifest.inputs = (replace(manifest.inputs[0], shape=value),)
    elif mutation == "input_strides":
        manifest.inputs = (replace(manifest.inputs[0], strides=value),)
    elif mutation == "output_parameter":
        manifest.outputs = (
            *manifest.outputs[:2],
            replace(manifest.outputs[2], parameter_name=value),
        )
    elif mutation == "output_shape":
        manifest.outputs = (
            *manifest.outputs[:2],
            replace(manifest.outputs[2], shape=value),
        )
    elif mutation == "output_strides":
        manifest.outputs = (
            *manifest.outputs[:2],
            replace(manifest.outputs[2], strides=value),
        )
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match="RF-DETR MIGraphX"):
        RfDetrMigraphxRunner._validate_contract(SimpleNamespace(manifest=manifest))


def test_rfdetr_runner_rejects_windows_before_artifact_runtime_load(monkeypatch) -> None:
    import jasna.mosaic.rfdetr_migraphx_runner as module

    monkeypatch.setattr(module.sys, "platform", "win32")
    artifact_runner = MagicMock(side_effect=AssertionError("must not load artifact runtime"))
    monkeypatch.setattr(module, "MigraphxArtifactRunner", artifact_runner)

    with pytest.raises(RuntimeError, match="Linux only"):
        RfDetrMigraphxRunner(
            Path("manifest.json"),
            weights_path=Path("rfdetr-v6.pt"),
            device=torch.device("cuda:0"),
            fp16=True,
            resolution=576,
            variant="medium",
        )

    artifact_runner.assert_not_called()


def test_rfdetr_model_uses_explicit_artifact_runner_without_torch_fallback(monkeypatch) -> None:
    import jasna.mosaic.rfdetr as module
    import jasna.mosaic.rfdetr_migraphx_runner as migraphx_module
    import jasna.mosaic.rfdetr_torch_runner as torch_module

    runner = MagicMock()
    runner.input_names = ["input"]
    runner.input_dtypes = {"input": torch.float32}
    runner.output_names = ["dets", "labels", "masks"]
    runner.outputs = {
        "dets": torch.empty((1, 200, 4)),
        "labels": torch.empty((1, 200, 3)),
        "masks": torch.empty((1, 200, 144, 144)),
    }
    runner.batch_size = 1
    runner.artifact_path = Path("detector.mxr")
    monkeypatch.setattr(module, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "is_nvidia_device", lambda _device: False)
    monkeypatch.setattr(migraphx_module, "RfDetrMigraphxRunner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(
        torch_module,
        "RfDetrTorchRunner",
        lambda *args, **kwargs: pytest.fail("PyTorch fallback must not be constructed"),
    )

    model = module.RfDetrMosaicDetectionModel(
        weights_path=Path("rfdetr-v6.pt"),
        batch_size=4,
        device=torch.device("cpu"),
        resolution=576,
        dynamic_batch=True,
        torch_variant="medium",
        fp16=True,
        amd_migraphx_manifest_path=Path("manifest.json"),
    )

    assert model.runner is runner
    assert model.batch_size == 1
    assert model.dynamic_batch is False
    assert model.engine_path == Path("detector.mxr")


def test_rfdetr_model_rejects_windows_manifest_before_runner_import(monkeypatch) -> None:
    import jasna.mosaic.rfdetr as module

    monkeypatch.setattr(module, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(module.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="Linux AMD/ROCm"):
        module.RfDetrMosaicDetectionModel(
            weights_path=Path("rfdetr-v6.pt"),
            batch_size=4,
            device=torch.device("cpu"),
            resolution=576,
            dynamic_batch=True,
            torch_variant="medium",
            fp16=True,
            amd_migraphx_manifest_path=Path("manifest.json"),
        )


def test_b4_input_splits_static_b1_and_clones_reused_pointer_outputs() -> None:
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel

    model = object.__new__(RfDetrMosaicDetectionModel)
    model.batch_size = 1
    model.dynamic_batch = False
    model._input_name = "input"

    class ReusingRunner:
        output_names = ["dets", "labels", "masks"]

        def __init__(self) -> None:
            self.buffers = {
                "dets": torch.empty((1, 1, 4)),
                "labels": torch.empty((1, 1, 1)),
                "masks": torch.empty((1, 1, 1, 1)),
            }
            self.dispatches: list[float] = []

        def infer(self, inputs):
            value = float(inputs["input"][0, 0, 0, 0])
            self.dispatches.append(value)
            for buffer in self.buffers.values():
                buffer.fill_(value)
            return self.buffers

    runner = ReusingRunner()
    model.runner = runner
    values = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1)

    outputs = model._infer(values)

    assert runner.dispatches == [1.0, 2.0, 3.0, 4.0]
    assert outputs["dets"][:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert outputs["labels"][:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert outputs["masks"][:, 0, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
