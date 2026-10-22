#include "block.h"
#include "glue.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using jasna_ltx::Block;
  using jasna_ltx::Config;
  py::class_<Config>(m, "Config")
      .def(py::init<>())
      .def_readwrite("device", &Config::device).def_readwrite("heads", &Config::heads)
      .def_readwrite("eps", &Config::eps).def_readwrite("chunk_rows", &Config::chunk_rows)
      .def_readwrite("narrow_tile", &Config::narrow_tile).def_readwrite("wide_tile", &Config::wide_tile)
      .def_readwrite("small_tile", &Config::small_tile).def_readwrite("small_max_n", &Config::small_max_n)
      .def_readwrite("wide_k", &Config::wide_k).def_readwrite("group_m", &Config::group_m)
      .def_readwrite("stream_k_max_waves", &Config::stream_k_max_waves).def_readwrite("programs", &Config::programs)
      .def_readwrite("has_tma", &Config::has_tma).def_readwrite("cublas_int8", &Config::cublas_int8)
      .def_readwrite("epilogue_tile", &Config::epilogue_tile)
      .def_readwrite("fp4_tensor_scale_divisor", &Config::fp4_tensor_scale_divisor)
      .def_readwrite("fp4_block", &Config::fp4_block).def_readwrite("fp4_quantize_rows", &Config::fp4_quantize_rows)
      .def_readwrite("fp4_quantize_cols", &Config::fp4_quantize_cols)
      .def_readwrite("fp4_scale_tile_rows", &Config::fp4_scale_tile_rows)
      .def_readwrite("fp4_scale_tile_cols", &Config::fp4_scale_tile_cols)
      .def_readwrite("fp4_recipe", &Config::fp4_recipe).def_readwrite("fp4_swizzle", &Config::fp4_swizzle)
      .def_readwrite("kernels", &Config::kernels).def_readwrite("kernel_directory", &Config::kernel_directory);
  py::class_<Block>(m, "Block").def(py::init<const Config&>()).def("forward", &Block::forward)
      .def("self_attention", &Block::self_attention).def("normed_self_attention", &Block::normed_self_attention)
      .def("prompt_key_value", &Block::prompt_key_value);
  m.def("norm_rope", [](const at::Tensor& x, const at::Tensor& weight, const at::Tensor& cos, const at::Tensor& sin,
                        const at::Tensor& out, int64_t head_dim, double eps) {
    jasna_ltx::norm_rope(x, weight, &cos, &sin, out, head_dim, eps);
  });
}
