import sys
from contextlib import nullcontext
from unittest.mock import patch

import torch

from factories import cli_args, cli_environment


def test_cli_runs_the_job_on_the_chosen_device(tmp_path, nvidia_vendor) -> None:
    from jasna.main import main

    devices = []

    def record_device(device):
        devices.append(torch.device(device))
        return nullcontext()

    with (
        cli_environment() as pipeline_cls,
        patch("jasna.main.check_supported_gpu", return_value=(True, "Fake GPU")) as check_gpu,
        patch("jasna.accelerator.device_context", side_effect=record_device),
        patch("torch.cuda.Stream"),
        patch.object(sys, "argv", ["jasna", *cli_args(tmp_path, "--device", "cuda:1")]),
    ):
        main()

    assert torch.device("cuda:1") in devices
    check_gpu.assert_called_once_with("cuda:1")
    assert pipeline_cls.call_args.kwargs["session"].device == torch.device("cuda:1")
