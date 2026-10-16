// One LTX transformer block (``jasna.ltx.transformer.block_forward``) as a single native call.
//
// GEMMs and quantization run the Triton kernels of ``jasna/ltx/kernels.py``, compiled ahead of
// time to cubins (``aot.py``) and launched through the CUDA driver; NVFP4 GEMMs and the prompt
// cross-attention are the same ATen ops the torch path calls; video self-attention is the INT8
// attention vendored from comfy-kitchen (``sage_attention/``); the rest is ``glue.cu``. Every model constant
// and launch policy value comes from the shared typed configuration.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <utility>

#include "glue.h"
#include "block.h"
#include "sage_attention/launchers.h"

namespace py = pybind11;

namespace jasna_ltx {
namespace {

void check_cu(CUresult result, const char* what) {
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    TORCH_CHECK(false, "LTX native block: ", what, " failed: ", message ? message : "unknown CUDA error");
  }
}

int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

class ModuleOwner {
 public:
  ModuleOwner() = default;
  ModuleOwner(const ModuleOwner&) = delete;
  ModuleOwner& operator=(const ModuleOwner&) = delete;
  ~ModuleOwner() {
    if (module_ && cuModuleUnload(std::exchange(module_, nullptr)) != CUDA_SUCCESS)
      std::fputs("LTX native block: unloading a kernel module failed\n", stderr);
  }
  CUmodule* address() { return &module_; }
  CUmodule get() const { return module_; }
  CUmodule release() { return std::exchange(module_, nullptr); }

 private:
  CUmodule module_ = nullptr;
};

// One ahead-of-time compiled Triton kernel and the argument list its launcher expects.
class Kernel {
 public:
  explicit Kernel(KernelSpec spec, KernelSource source) : source_(std::move(source)) {
    validate_kernel_spec(spec);
    TORCH_CHECK(source_.kind() == KernelSource::Kind::Image ? !source_.image().empty() : !source_.path().empty(),
                "LTX native block: empty kernel source");
    function_name_ = std::move(spec.function);
    shared_ = spec.shared;
    threads_ = spec.warps * 32;
    args_ = std::move(spec.args);
    divisible_ = std::move(spec.divisible);
    descriptors_ = std::move(spec.tensordesc);
    ModuleOwner loaded;
    if (source_.kind() == KernelSource::Kind::Image)
      check_cu(cuModuleLoadData(loaded.address(), source_.image().data()), "loading a kernel image");
    else
      check_cu(cuModuleLoad(loaded.address(), source_.path().c_str()), "loading a kernel cubin");
    check_cu(cuModuleGetFunction(&function_, loaded.get(), function_name_.c_str()), "finding a kernel");
    if (shared_ > 49152) {  // as Triton's own loader: opt in to more than 48 KiB of dynamic shared memory
      CUdevice device;
      check_cu(cuCtxGetDevice(&device), "querying the device");
      int optin = 0, static_shared = 0;
      check_cu(cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device), "shared");
      check_cu(cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function_), "shared");
      TORCH_CHECK(shared_ <= optin, "LTX native block: ", function_name_, " needs more shared memory than the GPU has");
      check_cu(cuFuncSetCacheConfig(function_, CU_FUNC_CACHE_PREFER_SHARED), "shared");
      check_cu(cuFuncSetAttribute(function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin - static_shared),
               "shared");
    }
    module_ = loaded.release();
  }
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;
  ~Kernel() {
    if (module_ && cuModuleUnload(std::exchange(module_, nullptr)) != CUDA_SUCCESS)
      std::fputs("LTX native block: unloading a kernel module failed\n", stderr);
  }

  friend class Launch;

 private:
  KernelSource source_;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
  std::string function_name_;
  int shared_ = 0, threads_ = 0;
  std::vector<std::string> args_;
  std::vector<bool> divisible_;
  std::vector<TensorDescriptorLayout> descriptors_;
};

// Packs a kernel's arguments in launcher order, checking each against the compiled signature
// and its divisibility promises, then launches it with Triton's two trailing scratch pointers.
class Launch {
 public:
  explicit Launch(const Kernel& kernel) : kernel_(kernel), slots_(kernel.args_.size() + 2) {}

  Launch& ptr(const void* p) {
    const auto& kind = next();
    TORCH_CHECK(kind[0] == '*', where(), " expects ", kind, ", got a pointer");
    TORCH_CHECK(!kernel_.divisible_[index_] || reinterpret_cast<uintptr_t>(p) % 16 == 0, where(),
                " needs a 16-byte aligned pointer");
    return put(&p, sizeof(p));
  }
  Launch& ptr(const at::Tensor& t) { return ptr(t.data_ptr()); }

  Launch& i32(int64_t value) {
    const auto& kind = next();
    TORCH_CHECK(kind == "i32", where(), " expects ", kind, ", got i32");
    TORCH_CHECK(value == static_cast<int32_t>(value), where(), " overflows i32: ", value);
    TORCH_CHECK(!kernel_.divisible_[index_] || value % 16 == 0, where(), " must be a multiple of 16, got ", value);
    const int32_t v = static_cast<int32_t>(value);
    return put(&v, sizeof(v));
  }

  Launch& i64(int64_t value) {
    TORCH_CHECK(next() == "i64", where(), " expects i64");
    return put(&value, sizeof(value));
  }

  // A host TensorDescriptor of a row-major (rows, cols) tensor, expanded the way Triton's
  // launcher expands it: the CUtensorMap, then the shape (i32) and strides (i64).
  Launch& tensor_descriptor(const at::Tensor& t) {
    TORCH_CHECK(next() == "nvTmaDesc", where(), " expects a tensor descriptor");
    TORCH_CHECK(t.dim() == 2 && t.stride(1) == 1, "tensor descriptors need a row-major matrix");
    const auto& layout = kernel_.descriptors_.at(descriptor_++);
    alignas(64) CUtensorMap map;
    const cuuint64_t shape[2] = {static_cast<cuuint64_t>(t.size(1)), static_cast<cuuint64_t>(t.size(0))};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(t.stride(0) * layout.elem_size)};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(layout.block[1]), static_cast<cuuint32_t>(layout.block[0])};
    const cuuint32_t element_strides[2] = {1, 1};
    check_cu(cuTensorMapEncodeTiled(&map, static_cast<CUtensorMapDataType>(layout.elem_type), 2, t.data_ptr(),
                                    shape, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    static_cast<CUtensorMapSwizzle>(layout.swizzle), CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
             "encoding a TMA descriptor");
    put(&map, sizeof(map));
    i32(t.size(0)).i32(t.size(1));
    return i64(t.stride(0)).i64(1);
  }

  void operator()(int64_t x, int64_t y, int64_t z) {
    TORCH_CHECK(index_ == kernel_.args_.size(), "LTX native block: ", kernel_.function_name_, " got ", index_, " of ",
                kernel_.args_.size(), " arguments");
    const CUdeviceptr scratch = 0;
    put(&scratch, sizeof(scratch));
    put(&scratch, sizeof(scratch));
    std::vector<void*> params;
    for (auto& slot : slots_) params.push_back(slot.bytes);
    if (x * y * z == 0) return;
    check_cu(cuLaunchKernel(kernel_.function_, x, y, z, kernel_.threads_, 1, 1, kernel_.shared_,
                            at::cuda::getCurrentCUDAStream(), params.data(), nullptr),
             kernel_.function_name_.c_str());
  }

 private:
  struct alignas(64) Slot {
    unsigned char bytes[128];
  };

  const std::string& next() {
    TORCH_CHECK(index_ < kernel_.args_.size(), "LTX native block: too many arguments for ", kernel_.function_name_);
    return kernel_.args_[index_];
  }
  std::string where() const { return kernel_.function_name_ + " argument " + std::to_string(index_); }
  Launch& put(const void* value, size_t size) {
    std::memcpy(slots_[filled_++].bytes, value, size);
    ++index_;
    return *this;
  }

  const Kernel& kernel_;
  std::vector<Slot> slots_;
  size_t index_ = 0, filled_ = 0, descriptor_ = 0;
};

}  // namespace

class Block::Impl {
  struct Activation;

 public:
  explicit Impl(const Config& config) : Impl(config, path_kernels(config)) {}

  Impl(const Config& config, std::vector<KernelImage> kernels) : Impl(config, image_kernels(std::move(kernels))) {}

 private:
  struct KernelInput {
    KernelSpec spec;
    KernelSource source;
  };

  static std::vector<KernelInput> path_kernels(const Config& config) {
    std::vector<KernelInput> result;
    for (const auto& item : config.kernels) {
      const auto entry = item.cast<py::dict>();
      KernelSpec spec;
      spec.family = entry["family"].cast<std::string>();
      spec.constexprs = entry["constexprs"].cast<std::map<std::string, int64_t>>();
      spec.function = entry["function"].cast<std::string>();
      spec.shared = entry["shared"].cast<int>();
      spec.warps = entry["warps"].cast<int>();
      spec.args = entry["args"].cast<std::vector<std::string>>();
      spec.divisible = entry["divisible"].cast<std::vector<bool>>();
      for (const auto& descriptor : entry["tensordesc"].cast<py::list>()) {
        const auto t = descriptor.cast<py::tuple>();
        spec.tensordesc.push_back({t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(),
                                 t[3].cast<std::vector<int64_t>>()});
      }
      validate_kernel_spec(spec);
      result.push_back({std::move(spec), KernelSource::from_path(config.kernel_directory + "/" + entry["cubin"].cast<std::string>())});
    }
    return result;
  }

  static std::vector<KernelInput> image_kernels(std::vector<KernelImage> kernels) {
    validate_kernel_images(kernels);
    std::vector<KernelInput> result;
    result.reserve(kernels.size());
    for (auto& image : kernels)
      result.push_back({std::move(image.spec), KernelSource::from_image(std::move(image.cubin))});
    return result;
  }

  Impl(const Config& config, std::vector<KernelInput> kernels) : config_(config) {
    TORCH_CHECK(config_.narrow_tile.size() == 5 && config_.wide_tile.size() == 5 && config_.small_tile.size() == 5,
                "GEMM tiles are (BM, BN, BK, stages, warps)");
    TORCH_CHECK(config_.epilogue_tile.size() == 2, "the cuBLAS-path epilogue tile is (BM, BN)");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(config_.device));
    C10_CUDA_CHECK(cudaFree(nullptr));  // makes the device's primary context current for the driver API
    for (auto& input : kernels) {
      const auto identity = kernel_key(input.spec.family, input.spec.constexprs);
      kernels_.emplace(identity, std::make_unique<Kernel>(std::move(input.spec), std::move(input.source)));
    }
  }

 public:

  at::Tensor forward(const std::vector<std::optional<at::Tensor>>& linear_tensors, const at::Tensor& q_norm,
                     const at::Tensor& k_norm, const at::Tensor& cross_q_norm, const at::Tensor& scale_shift_table,
                     const at::Tensor& x, const at::Tensor& block_rows, const at::Tensor& prompt_key,
                     const at::Tensor& prompt_value, const at::Tensor& rope_cos, const at::Tensor& rope_sin,
                     int64_t cta_k, bool skip_self_attention) {
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(linear_tensors.size() == LinearCount * 4, "expected 4 tensors for each of ", LinearCount, " linears");
    std::vector<Linear> w;
    for (int i = 0; i < LinearCount; ++i) w.push_back(Linear::from(linear_tensors, i));
    TORCH_CHECK(x.dim() == 3 && x.size(0) == 1 && x.is_contiguous() && x.scalar_type() == at::kBFloat16,
                "x must be a contiguous bf16 [1, rows, dim] block input");
    const int64_t rows = x.size(1), dim = x.size(2);
    TORCH_CHECK(rows % 2 == 0, "a block input holds a target and a reference half");
    const bool fp4 = w[SelfV].kind == Kind::Fp4;
    // AdaLN rows per table entry: [9, 2 halves, dim], each entry contiguous for the kernels
    const at::Tensor ada = (scale_shift_table.unsqueeze(0) + block_rows[0]).transpose(0, 1).contiguous();
    const at::Tensor xin = x.view({rows, dim});

    // gated video self-attention
    Activation attn_in = modulated(xin, ada[0], ada[1], fp4, rows);
    at::Tensor v = linear(attn_in, w[SelfV]);
    at::Tensor attention;  // (rows, heads, head_dim) view
    const int64_t heads = config_.heads, head_dim = dim / heads;
    if (skip_self_attention) {
      attention = v.view({rows, heads, head_dim});
    } else {
      const at::Tensor q = at::empty_like(v), k = at::empty_like(v);
      norm_rope(linear(attn_in, w[SelfQ]), q_norm, &rope_cos, &rope_sin, q, head_dim, config_.eps);
      norm_rope(linear(attn_in, w[SelfK]), k_norm, &rope_cos, &rope_sin, k, head_dim, config_.eps);
      attention = self_attention(q, k, v, rows, heads, head_dim, cta_k);
    }
    at::Tensor h = project_gated(attention, linear(attn_in, w[SelfGate]), w[SelfOut], xin, ada[2], rows, fp4);

    // gated prompt cross-attention
    Activation query = modulated(h, ada[6], ada[7], fp4, rows);
    const at::Tensor q = at::empty({rows, dim}, x.options());
    norm_rope(linear(query, w[CrossQ]), cross_q_norm, nullptr, nullptr, q, head_dim, config_.eps);
    const at::Tensor q_heads = q.view({1, rows, heads, head_dim}).transpose(1, 2).to(prompt_key.scalar_type());
    const at::Tensor cross =
        at::scaled_dot_product_attention(q_heads, prompt_key, prompt_value).to(at::kBFloat16)[0].transpose(0, 1);
    h = project_gated(cross, linear(query, w[CrossGate]), w[CrossOut], h, ada[8], rows, fp4);

    // feed-forward with gated residual, quantized or modulated over all rows, then in row chunks
    return (fp4 ? feed_forward_fp4(h, ada, w[FfUp], w[FfDown], rows) : feed_forward_int8(h, ada, w[FfUp], w[FfDown], rows))
        .view({1, rows, dim});
  }

  // comfy-kitchen's INT8 attention of token-major (rows, heads * head_dim) q/k/v, launched the way
  // comfy_kitchen.sage_attention._int8_attention_cuda launches it unmasked; exposed for the
  // bit-exactness test against comfy_kitchen.int8_attention.
  at::Tensor self_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int64_t rows, int64_t heads,
                            int64_t head_dim, int64_t cta_k) {
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256, "INT8 attention needs a head size of 64/128/256");
    for (const auto* t : {&q, &k, &v}) {
      TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 2 && t->size(0) == rows &&
                  t->size(1) == heads * head_dim && t->stride(1) == 1, "q/k/v must be (rows, heads * head_dim) bf16");
    }
    const int L = static_cast<int>(rows), H = static_cast<int>(heads), D = static_cast<int>(head_dim);
    const int64_t padded = cdiv(L, cta_k) * cta_k;
    const auto i8 = q.options().dtype(at::kChar), f32 = q.options().dtype(at::kFloat);
    const at::Tensor q_int8 = at::empty({1, H, L, D}, i8), k_int8 = at::empty({1, H, L, D}, i8);
    const at::Tensor q_scale = at::empty({1, H, cdiv(L, 128) * (D == 256 ? 64 : 32)}, f32);
    const at::Tensor k_scale = at::empty({1, H, cdiv(L, cta_k) * 4}, f32);
    const at::Tensor v_int8 = at::empty({H * D, padded}, i8), v_scale = at::empty({H * D}, f32);
    const at::Tensor anchors = at::empty({1, H}, q.options().dtype(at::kInt));
    const at::Tensor out = at::empty({1, H, L, D}, q.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t token_stride = q.stride(0);
    const int code = sage_attention::kBFloat16;
    sage_attention::launch_quant_qk_per_thread_int8(
        q.data_ptr(), q_int8.data_ptr(), q_scale.data_ptr(), k.data_ptr(), k_int8.data_ptr(), k_scale.data_ptr(), 1, H,
        L, H, L, D, 128, D == 256 ? 16 : 32, static_cast<int>(cta_k), static_cast<int>(cta_k), q.numel(), D,
        token_stride, k.numel(), D, k.stride(0), code, anchors.data_ptr(), stream);
    sage_attention::launch_quant_v_int8_kernel(v.data_ptr(), v_int8.data_ptr(), v_scale.data_ptr(), 1, H, L, D,
                                               static_cast<int>(padded), v.numel(), D, v.stride(0), code, stream);
    const int64_t head_stride = static_cast<int64_t>(L) * D, v_head_stride = D * padded;
    sage_attention::launch_sage_attn_kernel(
        q_int8.data_ptr(), k_int8.data_ptr(), v_int8.data_ptr(), out.data_ptr(), q_scale.data_ptr(), k_scale.data_ptr(),
        v_scale.data_ptr(), nullptr, 0, 0, 0, 0, -1, static_cast<int>(cta_k), 1, L, L, H, H, D, H * head_stride, D,
        head_stride, H * head_stride, D, head_stride, H * v_head_stride, v_head_stride, static_cast<int>(padded),
        H * head_stride, D, head_stride, static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), code, stream,
        nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view({H, L, D}).transpose(0, 1);
  }

  std::pair<at::Tensor, at::Tensor> prompt_key_value(
      const std::vector<std::optional<at::Tensor>>& linear_tensors, const at::Tensor& cross_k_norm,
      const at::Tensor& prompt_scale_shift_table, const at::Tensor& prompt_rows, const at::Tensor& context) {
    std::vector<Linear> w;
    for (int i = 0; i < LinearCount; ++i) w.push_back(Linear::from(linear_tensors, i));
    const at::Tensor kv = prompt_scale_shift_table.unsqueeze(0).unsqueeze(0) + prompt_rows;
    const at::Tensor input = context * (1 + kv.select(2, 1)) + kv.select(2, 0);
    const bool fp4 = w[CrossV].kind == Kind::Fp4;
    const Activation prepared = plain_activation(input.view({-1, input.size(-1)}).contiguous(), fp4);
    const int64_t dim = input.size(-1), heads = config_.heads, head_dim = dim / heads;
    const at::Tensor key = at::empty({input.size(1), dim}, input.options());
    norm_rope(linear(prepared, w[CrossK]), cross_k_norm, nullptr, nullptr, key, head_dim, config_.eps);
    const at::ScalarType dtype = prompt_attention_dtype(context);
    return {key.view({1, input.size(1), heads, head_dim}).transpose(1, 2).to(dtype),
            linear(prepared, w[CrossV]).view({1, input.size(1), heads, head_dim}).transpose(1, 2).to(dtype)};
  }

  // transformer.prompt_attention_dtype: no fused attention kernel takes bf16 before SM 8.0.
  static at::ScalarType prompt_attention_dtype(const at::Tensor& context) {
    return at::cuda::getDeviceProperties(context.get_device())->major < 8 ? at::kHalf : at::kBFloat16;
  }

 private:
  enum class Kind { Int8, Fp4, Dense };

  struct Linear {
    Kind kind;
    at::Tensor weight, scale, scale2;
    std::optional<at::Tensor> bias;

    static Linear from(const std::vector<std::optional<at::Tensor>>& t, int index) {
      const auto& weight = t[index * 4].value();
      const Kind kind = weight.scalar_type() == at::kChar ? Kind::Int8
                        : weight.scalar_type() == at::kByte ? Kind::Fp4 : Kind::Dense;
      return {kind, weight, kind == Kind::Dense ? at::Tensor() : t[index * 4 + 1].value(),
              kind == Kind::Fp4 ? t[index * 4 + 2].value() : at::Tensor(), t[index * 4 + 3]};
    }
  };

  // One block activation quantized once for every linear that reads it.
  struct Activation {
    at::Tensor values;                               // bf16 rows (NVFP4 blocks; dense linears read these)
    at::Tensor codes, scales;                        // INT8 ConvRot codes and row scales
    at::Tensor packed, block_scales, tensor_scale;   // NVFP4
  };

  Activation plain_activation(const at::Tensor& x, bool fp4) const {
    if (!fp4) {
      auto [codes, scales] = rotate_quantize(x, nullptr, nullptr);
      return {x, codes, scales, {}, {}, {}};
    }
    const at::Tensor amax = at::amax(at::abs(x)).to(at::kFloat).reshape({1});
    return fp4_activation(x, tensor_scales(amax));
  }

  static std::string key(const std::string& family, const std::map<std::string, int64_t>& constexprs) {
    return kernel_key(family, constexprs);
  }

  const Kernel& kernel(const std::string& family, const std::map<std::string, int64_t>& constexprs) const {
    const auto found = kernels_.find(key(family, constexprs));
    TORCH_CHECK(found != kernels_.end(), "LTX native block: no compiled kernel ", key(family, constexprs));
    return *found->second;
  }

  std::pair<at::Tensor, at::Tensor> rotate_quantize(const at::Tensor& x, const at::Tensor* shift,
                                                    const at::Tensor* scale) const {
    const int64_t rows = x.size(0), k = x.size(1);
    TORCH_CHECK(x.is_contiguous(), "rotate_quantize needs contiguous rows");
    const at::Tensor codes = at::empty({rows, k}, x.options().dtype(at::kChar));
    const at::Tensor scales = at::empty({rows}, x.options().dtype(at::kFloat));
    Launch(kernel("rotate_quantize", {{"K", k}, {"MODULATE", shift != nullptr}}))
        .ptr(x).ptr(shift ? *shift : x).ptr(scale ? *scale : x).ptr(codes).ptr(scales)
        .i32(x.stride(0)).i32(std::max<int64_t>(rows / 2, 1))(rows, 1, 1);
    return {codes, scales};
  }

  // FP4 block quantization of bf16 (rows, k) under tensor_scale (a 1-element fp32 tensor).
  std::pair<at::Tensor, at::Tensor> quantize_fp4(const at::Tensor& x, const at::Tensor& tensor_scale, bool gelu) const {
    const int64_t rows = x.size(0), k = x.size(1);
    TORCH_CHECK(x.is_contiguous() && k % config_.fp4_quantize_cols == 0, "quantize_fp4 needs contiguous rows of a width divisible by ",
                config_.fp4_quantize_cols);
    const at::Tensor packed = at::empty({rows, k / 2}, x.options().dtype(at::kByte));
    const int64_t scale_rows = cdiv(rows, config_.fp4_scale_tile_rows) * config_.fp4_scale_tile_rows;
    const int64_t scale_cols = cdiv(k / config_.fp4_block, config_.fp4_scale_tile_cols) * config_.fp4_scale_tile_cols;
    const at::Tensor scales = at::zeros({scale_rows, scale_cols}, x.options().dtype(at::kByte));
    Launch(kernel("quantize_fp4", {{"GELU", gelu}, {"BM", config_.fp4_quantize_rows}, {"BK", config_.fp4_quantize_cols}}))
        .ptr(x).ptr(tensor_scale).ptr(packed).ptr(scales).i32(rows).i32(k).i32(x.stride(0))(
            cdiv(rows, config_.fp4_quantize_rows), k / config_.fp4_quantize_cols, 1);
    return {packed, scales.view(at::kFloat8_e4m3fn)};
  }

  at::Tensor zeros_f32(int64_t count, const at::Tensor& like) const {
    return at::zeros({count}, like.options().dtype(at::kFloat));
  }

  at::Tensor tensor_scales(const at::Tensor& amax) const {
    const at::Tensor scales = at::empty_like(amax);
    fp4_tensor_scales(amax, scales, config_.fp4_tensor_scale_divisor);
    return scales;
  }

  Activation fp4_activation(const at::Tensor& values, const at::Tensor& tensor_scale) const {
    auto [packed, block_scales] = quantize_fp4(values, tensor_scale, false);
    return {values, {}, {}, packed, block_scales, tensor_scale};
  }

  // rms_norm(x) * (1 + scale) + shift, quantized for the linears that read it.
  Activation modulated(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale, bool fp4,
                       int64_t rows) const {
    if (!fp4) {
      auto [codes, scales] = rotate_quantize(x, &shift, &scale);
      return {{}, codes, scales};
    }
    const at::Tensor values = at::empty_like(x), amax = zeros_f32(1, x);
    rms_modulate(x, shift, scale, values, amax.data_ptr<float>(), rows, config_.eps);
    return fp4_activation(values, tensor_scales(amax));
  }

  at::Tensor int8_mm(const at::Tensor& codes, const at::Tensor& act_scales, const Linear& w, const at::Tensor& out,
                     bool gelu, const at::Tensor* residual, const at::Tensor* gate, int64_t half_rows,
                     int64_t row_offset) const {
    const int64_t m = codes.size(0), k = codes.size(1), n = w.weight.size(0);
    TORCH_CHECK(w.weight.size(1) == k && out.size(0) == m && out.size(1) == n && out.is_contiguous(),
                "int8 GEMM shapes do not match");
    const bool has_bias = w.bias.has_value();
    const void* bias = has_bias ? w.bias->data_ptr() : w.scale.data_ptr();
    const void* res = residual ? residual->data_ptr() : out.data_ptr();
    const void* gate_rows = gate ? gate->data_ptr() : w.scale.data_ptr();
    const std::map<std::string, int64_t> flags = {{"HAS_BIAS", has_bias}, {"GELU", gelu}, {"RESIDUAL", residual != nullptr}};
    auto epilogue_args = [&](Launch& launch) -> Launch& {
      return launch.ptr(act_scales).ptr(w.scale).ptr(bias).ptr(res).ptr(gate_rows).ptr(out);
    };
    if (config_.cublas_int8) {
      // torch._int_mm needs more than 16 rows
      const at::Tensor acc = at::_int_mm(m <= 16 ? at::constant_pad_nd(codes, {0, 0, 0, 17 - m}) : codes, w.weight.t());
      auto constexprs = flags;
      constexprs["BM"] = config_.epilogue_tile[0];
      constexprs["BN"] = config_.epilogue_tile[1];
      Launch launch(kernel("epilogue", constexprs));
      epilogue_args(launch.ptr(acc)).i32(m).i32(n).i32(half_rows).i32(row_offset)(
          cdiv(m, config_.epilogue_tile[0]), cdiv(n, config_.epilogue_tile[1]), 1);
      return out;
    }
    const auto& tile = n < config_.small_max_n ? config_.small_tile : k > config_.wide_k ? config_.wide_tile
                                                                                         : config_.narrow_tile;
    const int64_t bm = tile[0], bn = tile[1], bk = tile[2];
    TORCH_CHECK(k % bk == 0, "int8 GEMM needs a width divisible by ", bk);
    auto constexprs = flags;
    constexprs.insert({{"BM", bm}, {"BN", bn}, {"BK", bk}, {"GROUP_M", config_.group_m}});
    const int64_t tiles = cdiv(m, bm) * cdiv(n, bn), sms = config_.programs, tail = tiles % sms;
    // kernels.stream_k_splits
    const bool stream_k = config_.has_tma && n >= config_.small_max_n && tiles < config_.stream_k_max_waves * sms &&
                          tail > 0 && tail <= sms / 2;
    if (stream_k) {
      const at::Tensor workspace = at::zeros({tail * bm * bn + tail}, codes.options().dtype(at::kInt));
      Launch launch(kernel("mm_stream_k", constexprs));
      launch.tensor_descriptor(codes).tensor_descriptor(w.weight);
      epilogue_args(launch)
          .ptr(workspace).ptr(workspace.data_ptr<int32_t>() + tail * bm * bn)
          .i32(m).i32(n).i32(k).i32(half_rows).i32(row_offset).i32(tiles - tail)
          .i32(std::min(sms / tail, k / bk)).i32(sms)(sms, 1, 1);
    } else {
      Launch launch(kernel("mm", constexprs));
      epilogue_args(launch.ptr(codes).ptr(w.weight)).i32(m).i32(n).i32(k).i32(half_rows).i32(row_offset)(tiles, 1, 1);
    }
    return out;
  }

  at::Tensor fp4_mm(const at::Tensor& packed, const at::Tensor& block_scales, const at::Tensor& tensor_scale,
                    const Linear& w) const {
    return at::_scaled_mm_v2(packed.view(at::kFloat4_e2m1fn_x2), w.weight.view(at::kFloat4_e2m1fn_x2).t(),
                             {block_scales, tensor_scale.reshape({1})}, config_.fp4_recipe, config_.fp4_swizzle,
                             {w.scale, w.scale2.reshape({1})}, config_.fp4_recipe, config_.fp4_swizzle, w.bias,
                             at::kBFloat16);
  }

  at::Tensor linear(const Activation& a, const Linear& w) const {
    switch (w.kind) {
      case Kind::Int8: {
        const at::Tensor out = at::empty({a.codes.size(0), w.weight.size(0)}, a.scales.options().dtype(at::kBFloat16));
        return int8_mm(a.codes, a.scales, w, out, false, nullptr, nullptr, 1, 0);
      }
      case Kind::Fp4:
        return fp4_mm(a.packed, a.block_scales, a.tensor_scale, w);
      case Kind::Dense:
        return at::linear(a.values, w.weight, w.bias);
    }
    TORCH_CHECK(false, "unreachable");
  }

  // residual + gate[half] * out_projection(attention * 2 * sigmoid(gate_logits)): a new tensor.
  at::Tensor project_gated(const at::Tensor& attention, const at::Tensor& logits, const Linear& w,
                           const at::Tensor& residual, const at::Tensor& gate, int64_t rows, bool fp4) const {
    const at::Tensor gated = at::empty({rows, attention.size(1) * attention.size(2)}, residual.options());
    if (!fp4) {
      gate_heads(attention, logits, gated, nullptr);
      auto [codes, scales] = rotate_quantize(gated, nullptr, nullptr);
      const at::Tensor out = at::empty_like(residual);
      return int8_mm(codes, scales, w, out, false, &residual, &gate, rows / 2, 0);
    }
    const at::Tensor amax = zeros_f32(1, residual);
    gate_heads(attention, logits, gated, amax.data_ptr<float>());
    const Activation a = fp4_activation(gated, tensor_scales(amax));
    const at::Tensor out = at::empty_like(residual);
    gated_residual(out, residual, fp4_mm(a.packed, a.block_scales, a.tensor_scale, w), gate, rows / 2, 0);
    return out;
  }

  at::Tensor feed_forward_int8(const at::Tensor& x, const at::Tensor& ada, const Linear& up, const Linear& down,
                               int64_t rows) const {
    const at::Tensor shift = ada[3], scale = ada[4], gate = ada[5];
    auto [codes, scales] = rotate_quantize(x, &shift, &scale);
    const at::Tensor out = at::empty_like(x);
    for (int64_t start = 0; start < rows; start += config_.chunk_rows) {
      const int64_t stop = std::min(start + config_.chunk_rows, rows);
      const at::Tensor hidden = at::empty({stop - start, up.weight.size(0)}, x.options());
      int8_mm(codes.slice(0, start, stop), scales.slice(0, start, stop), up, hidden, true, nullptr, nullptr, 1, 0);
      auto [hidden_codes, hidden_scales] = rotate_quantize(hidden, nullptr, nullptr);
      const at::Tensor residual = x.slice(0, start, stop);
      int8_mm(hidden_codes, hidden_scales, down, out.slice(0, start, stop), false, &residual, &gate, rows / 2, start);
    }
    return out;
  }

  at::Tensor feed_forward_fp4(const at::Tensor& x, const at::Tensor& ada, const Linear& up, const Linear& down,
                              int64_t rows) const {
    // as transformer.block_forward: modulate every row, then per chunk one tensor scale per linear call
    const int64_t chunks = cdiv(rows, config_.chunk_rows);
    const at::Tensor values = at::empty_like(x), amax = zeros_f32(chunks, x);
    rms_modulate(x, ada[3], ada[4], values, amax.data_ptr<float>(), config_.chunk_rows, config_.eps);
    const at::Tensor chunk_scales = tensor_scales(amax);
    const at::Tensor out = at::empty_like(x);
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
      const int64_t start = chunk * config_.chunk_rows, stop = std::min(start + config_.chunk_rows, rows);
      const Activation a = fp4_activation(values.slice(0, start, stop), chunk_scales.slice(0, chunk, chunk + 1));
      const at::Tensor hidden = fp4_mm(a.packed, a.block_scales, a.tensor_scale, up);
      const at::Tensor hidden_amax = zeros_f32(1, x);
      gelu_amax(hidden, hidden_amax.data_ptr<float>());
      const at::Tensor hidden_scale = tensor_scales(hidden_amax);
      auto [packed, block_scales] = quantize_fp4(hidden, hidden_scale, true);
      gated_residual(out.slice(0, start, stop), x.slice(0, start, stop), fp4_mm(packed, block_scales, hidden_scale, down),
                     ada[5], rows / 2, start);
    }
    return out;
  }

  Config config_;
  std::map<std::string, std::unique_ptr<Kernel>> kernels_;
};

Block::Block(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
Block::Block(const Config& config, std::vector<KernelImage> kernels)
    : impl_(std::make_unique<Impl>(config, std::move(kernels))) {}
Block::~Block() = default;
at::Tensor Block::forward(const std::vector<std::optional<at::Tensor>>& linear_tensors, const at::Tensor& q_norm,
                          const at::Tensor& k_norm, const at::Tensor& cross_q_norm,
                          const at::Tensor& scale_shift_table, const at::Tensor& x, const at::Tensor& block_rows,
                          const at::Tensor& prompt_key, const at::Tensor& prompt_value, const at::Tensor& rope_cos,
                          const at::Tensor& rope_sin, int64_t cta_k, bool skip_self_attention) {
  return impl_->forward(linear_tensors, q_norm, k_norm, cross_q_norm, scale_shift_table, x, block_rows, prompt_key,
                        prompt_value, rope_cos, rope_sin, cta_k, skip_self_attention);
}
at::Tensor Block::self_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int64_t rows,
                                 int64_t heads, int64_t head_dim, int64_t cta_k) {
  return impl_->self_attention(q, k, v, rows, heads, head_dim, cta_k);
}
std::pair<at::Tensor, at::Tensor> Block::prompt_key_value(
    const std::vector<std::optional<at::Tensor>>& linear_tensors, const at::Tensor& cross_k_norm,
    const at::Tensor& prompt_scale_shift_table, const at::Tensor& prompt_rows, const at::Tensor& context) {
  return impl_->prompt_key_value(linear_tensors, cross_k_norm, prompt_scale_shift_table, prompt_rows, context);
}

}  // namespace jasna_ltx
