import numpy as np
import torch

from jasna.models.basicvsrpp import register_all_modules
from jasna.models.basicvsrpp.inference import get_default_gan_inference_config
from jasna.models.basicvsrpp.mmagic.data_sample import DataSample
from jasna.models.basicvsrpp.mmagic.registry import MODELS


def test_default_inference_components_build_and_normalize_frames():
    register_all_modules()
    config = get_default_gan_inference_config()
    loss = MODELS.build(config["pixel_loss"])
    torch.testing.assert_close(
        loss(torch.ones(1, 3, 2, 2), torch.zeros(1, 3, 2, 2)),
        torch.tensor(1.0),
    )

    frames = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    sample = DataSample()
    sample.set_tensor_data({"gt_img": list(frames)})
    inputs = sample.gt_img.unsqueeze(0)
    expected = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float()
    preprocessor = MODELS.build(config["data_preprocessor"])

    normalized = preprocessor({"inputs": inputs}, training=False)["inputs"]

    torch.testing.assert_close(normalized, expected / 255)
    torch.testing.assert_close(preprocessor.destruct(normalized), expected)
