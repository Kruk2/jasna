# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from jasna.models.basicvsrpp.mmagic.registry import MODELS
from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

@MODELS.register_module()
class BasicVSRPlusPlusGanNet(BasicVSRPlusPlusNet):
    def __init__(self,
                **kwargs):

        super().__init__(**kwargs)
        self.spynet.requires_grad_(False)


    def forward(self, lqs, return_lqs=False):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            return_lqs (bool): Whether to return LQ sequence. Default: False.

        Returns:
            Tensor: Output HR sequence.
        """
        outputs = super().forward(lqs)

        if return_lqs:
            return outputs, lqs
        else:
            return outputs
