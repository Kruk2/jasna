# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

from mmengine import DefaultScope

SCOPE = 'jasna.models.basicvsrpp.mmagic'

def register_all_modules():
    from .base_edit_model import BaseEditModel
    from .basicvsr_plusplus_net import BasicVSRPlusPlusNet
    from .basicvsr import BasicVSR
    from .data_preprocessor import DataPreprocessor
    from .pixelwise_loss import CharbonnierLoss
    from .real_basicvsr import RealBasicVSR

    never_created = DefaultScope.get_current_instance() is None or not DefaultScope.check_instance_created(SCOPE)
    if never_created:
        DefaultScope.get_instance(SCOPE, scope_name=SCOPE)
        return
