#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "block_kernels.h"

namespace jasna_ltx {

struct Config {
  int64_t device = 0, heads = 0, chunk_rows = 0;
  double eps = 0;
  std::vector<int64_t> narrow_tile, wide_tile, small_tile;
  int64_t small_max_n = 0, wide_k = 0, group_m = 0, stream_k_max_waves = 0, programs = 0;
  bool has_tma = false, cublas_int8 = false;
  std::vector<int64_t> epilogue_tile;  // (BM, BN) of the cuBLAS-path epilogue
  double fp4_tensor_scale_divisor = 0;
  int64_t fp4_block = 0, fp4_quantize_rows = 0, fp4_quantize_cols = 0, fp4_scale_tile_rows = 0,
          fp4_scale_tile_cols = 0;
  std::vector<int64_t> fp4_recipe, fp4_swizzle;
  pybind11::list kernels;
  std::string kernel_directory;
};

enum LinearIndex { SelfQ, SelfK, SelfV, SelfOut, SelfGate, CrossQ, CrossK, CrossV, CrossOut, CrossGate, FfUp, FfDown,
                   LinearCount };

class Block {
 public:
  explicit Block(const Config& config);
  Block(const Config& config, std::vector<KernelImage> kernels);
  ~Block();
  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;

  at::Tensor forward(const std::vector<std::optional<at::Tensor>>& linear_tensors, const at::Tensor& q_norm,
                     const at::Tensor& k_norm, const at::Tensor& cross_q_norm, const at::Tensor& scale_shift_table,
                     const at::Tensor& x, const at::Tensor& block_rows, const at::Tensor& prompt_key,
                     const at::Tensor& prompt_value, const at::Tensor& rope_cos, const at::Tensor& rope_sin,
                     int64_t cta_k, bool skip_self_attention);
  at::Tensor self_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int64_t rows, int64_t heads,
                            int64_t head_dim, int64_t cta_k);
  at::Tensor normed_self_attention(const at::Tensor& q_projection, const at::Tensor& k_projection, const at::Tensor& v,
                                   const at::Tensor& q_norm, const at::Tensor& k_norm, const at::Tensor& rope_cos,
                                   const at::Tensor& rope_sin, int64_t cta_k);
  std::pair<at::Tensor, at::Tensor> prompt_key_value(
      const std::vector<std::optional<at::Tensor>>& linear_tensors, const at::Tensor& cross_k_norm,
      const at::Tensor& prompt_scale_shift_table, const at::Tensor& prompt_rows, const at::Tensor& context);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jasna_ltx
