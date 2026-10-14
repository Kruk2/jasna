"""YOLO checkpoints must load under the frozen application's restricted policy."""
import os
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement


def test_ultralytics_dependency_requires_restricted_checkpoint_support():
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    requirement = next(Requirement(item) for item in dependencies if Requirement(item).name == "ultralytics")
    assert requirement.specifier
    assert "8.4.60" not in requirement.specifier
    assert "8.5.0" not in requirement.specifier
    assert metadata.version("ultralytics") in requirement.specifier


def test_yolo_checkpoint_loads_with_frozen_policy_and_rejects_unknown_types(tmp_path):
    environment = dict(os.environ, TORCH_FORCE_WEIGHTS_ONLY_LOAD="1", ULTRALYTICS_SAFE_LOAD="1")
    environment.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    result = subprocess.run(
        [sys.executable, "-c", """
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

root = Path(sys.argv[1])
checkpoint = root / 'segment.pt'
model = SegmentationModel({
    'nc': 1,
    'backbone': [[-1, 1, 'Conv', [8, 3, 2]]],
    'head': [[[0], 1, 'Segment', [1, 8, 8]]],
}, verbose=False)
torch.save({'model': model}, checkpoint)
loaded = YOLO(str(checkpoint), task='segment', verbose=False)
assert isinstance(loaded.model, SegmentationModel)
assert next(loaded.model.parameters()).device.type == 'cpu'

unknown = root / 'unknown.pt'
torch.save({'model': SimpleNamespace()}, unknown)
try:
    YOLO(str(unknown), task='segment', verbose=False)
except TypeError as error:
    assert isinstance(error.__cause__, pickle.UnpicklingError)
else:
    raise AssertionError('Unsupported checkpoint type was accepted')
assert os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] == '1'
assert os.environ['ULTRALYTICS_SAFE_LOAD'] == '1'
""", str(tmp_path)],
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
