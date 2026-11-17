# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

"""The MMagic model registry the vendored networks register into."""

from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

MODELS = Registry(
    'model',
    parent=MMENGINE_MODELS,
    locations=['jasna.models.basicvsrpp.mmagic'],
    scope='jasna.models.basicvsrpp.mmagic',
)
