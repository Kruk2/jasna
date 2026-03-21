# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os
import warnings

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from jasna.models.basicvsrpp import register_all_modules
from jasna.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from jasna.models.basicvsrpp.mmagic.basicvsr import BasicVSR
from jasna.models.basicvsrpp.mmagic.registry import MODELS

logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="The given buffer is not writable, and PyTorch does not support non-writable tensors",
    module="torch",
)

def get_default_gan_inference_config() -> dict:
    return dict(
        type='BasicVSRPlusPlusGan',
        generator=dict(
            type='BasicVSRPlusPlusGanNet',
            mid_channels=64,
            num_blocks=15,
            spynet_pretrained=None),
        pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
        is_use_ema=True,
        data_preprocessor=dict(
            type='DataPreprocessor',
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
        ))


def load_model(config: str | dict | None, checkpoint_path: str, device: torch.device, fp16: bool) -> BasicVSRPlusPlusGan | BasicVSR:
    register_all_modules()
    if device and type(device) == str:
        device = torch.device(device)

    if config is None:
        config = get_default_gan_inference_config()
    elif type(config) == str:
        config = Config.fromfile(config).model
    elif type(config) == dict:
        pass
    else:
        raise Exception("unsupported value for 'config', Must be either a file path to a config file or a dict definition of the model")
    model = MODELS.build(config)
    assert isinstance(model, BasicVSRPlusPlusGan) or isinstance(model, BasicVSR), "Unknown model config. Must be either stage1 (BasicVSR) or stage2 (BasicVSRPlusPlusGan)"
    _mmengine_logger = logging.getLogger("mmengine")
    _prev_level = _mmengine_logger.level
    _mmengine_logger.setLevel(logging.WARNING)
    try:
        load_checkpoint(model, checkpoint_path, map_location='cpu', logger=logger)
    finally:
        _mmengine_logger.setLevel(_prev_level)
    model.cfg = config
    model = model.to(device).eval()
    if fp16:
        model = model.half()
    return model
