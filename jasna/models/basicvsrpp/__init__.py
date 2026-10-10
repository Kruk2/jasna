# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from jasna.models.basicvsrpp.mmengine_compat import prepare_mmengine_for_windows_rocm


# MMEngine imports distributed-only symbols before exposing the inference
# helpers used below. Prepare that dependency boundary first on Windows ROCm.
prepare_mmengine_for_windows_rocm()


def register_all_modules():
    from jasna.models.basicvsrpp.mmagic import register_all_modules
    register_all_modules()
    from jasna.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet, BasicVSRPlusPlusGan
