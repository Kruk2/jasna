// SPDX-License-Identifier: Apache-2.0
// Raw and prepared buffers for short dense-mask preparation fused with Q quantization.
#pragma once
#include <cstdint>
struct SageDenseMask16 {
    const uint16_t* data = nullptr;
    uint16_t* packed = nullptr;
    int batches = 0, heads = 0, q = 0, k = 0, dtype = 0;
    int64_t sb = 0, sh = 0, sq = 0, sk = 0;
};
