// One LTX transformer block (``jasna.ltx.transformer.block_forward``) as a single native call.
//
// GEMMs and quantization run the Triton kernels of ``jasna/ltx/kernels.py``, compiled ahead of
// time to cubins / AMD code objects (``aot.py``) and launched through the GPU driver; NVFP4 GEMMs
// and the NVIDIA prompt cross-attention are the same ATen ops the torch path calls; video
// self-attention is the INT8 attention vendored from comfy-kitchen (``sage_attention/``, on AMD
// ``sage_attention_hip/``, which as in the AMD torch path also runs the prompt cross-attention);
// the rest is ``glue.cu``. Every model constant and launch policy value comes from the shared
// typed configuration.

#include <torch/extension.h>

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
#include "gpu_runtime.h"
#ifdef USE_ROCM
#include "norm_rope_quant.h"
#include "sage_attention_hip/launchers.h"
#else
#include "sage_attention/launchers.h"
#endif

namespace py = pybind11;

namespace jasna_ltx {
namespace {

void check_gpu(gpu::Result result, const char* what) {
  TORCH_CHECK(result == gpu::kSuccess, "LTX native block: ", what, " failed: ", gpu::error_string(result));
}

int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

class ModuleOwner {
 public:
  ModuleOwner() = default;
  ModuleOwner(const ModuleOwner&) = delete;
  ModuleOwner& operator=(const ModuleOwner&) = delete;
  ~ModuleOwner() {
    if (module_ && gpu::module_unload(std::exchange(module_, nullptr)) != gpu::kSuccess)
      std::fputs("LTX native block: unloading a kernel module failed\n", stderr);
  }
  gpu::Module* address() { return &module_; }
  gpu::Module get() const { return module_; }
  gpu::Module release() { return std::exchange(module_, nullptr); }

 private:
  gpu::Module module_ = nullptr;
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
      check_gpu(gpu::module_load_data(loaded.address(), source_.image().data()), "loading a kernel image");
    else
      check_gpu(gpu::module_load(loaded.address(), source_.path().c_str()), "loading a kernel cubin");
    check_gpu(gpu::module_get_function(&function_, loaded.get(), function_name_.c_str()), "finding a kernel");
#ifndef USE_ROCM  // AMD launches take a workgroup's whole LDS without opting in
    if (shared_ > 49152) {  // as Triton's own loader: opt in to more than 48 KiB of dynamic shared memory
      CUdevice device;
      check_gpu(cuCtxGetDevice(&device), "querying the device");
      int optin = 0, static_shared = 0;
      check_gpu(cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device), "shared");
      check_gpu(cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function_), "shared");
      TORCH_CHECK(shared_ <= optin, "LTX native block: ", function_name_, " needs more shared memory than the GPU has");
      check_gpu(cuFuncSetCacheConfig(function_, CU_FUNC_CACHE_PREFER_SHARED), "shared");
      check_gpu(cuFuncSetAttribute(function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin - static_shared),
                "shared");
    }
#endif
    module_ = loaded.release();
  }
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;
  ~Kernel() {
    if (module_ && gpu::module_unload(std::exchange(module_, nullptr)) != gpu::kSuccess)
      std::fputs("LTX native block: unloading a kernel module failed\n", stderr);
  }

  friend class Launch;

 private:
  KernelSource source_;
  gpu::Module module_ = nullptr;
  gpu::Function function_ = nullptr;
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
#ifdef USE_ROCM
    TORCH_CHECK(false, "LTX native block: AMD GPUs have no TMA descriptors");
#else
    TORCH_CHECK(t.dim() == 2 && t.stride(1) == 1, "tensor descriptors need a row-major matrix");
    const auto& layout = kernel_.descriptors_.at(descriptor_++);
    alignas(64) CUtensorMap map;
    const cuuint64_t shape[2] = {static_cast<cuuint64_t>(t.size(1)), static_cast<cuuint64_t>(t.size(0))};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(t.stride(0) * layout.elem_size)};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(layout.block[1]), static_cast<cuuint32_t>(layout.block[0])};
    const cuuint32_t element_strides[2] = {1, 1};
    check_gpu(cuTensorMapEncodeTiled(&map, static_cast<CUtensorMapDataType>(layout.elem_type), 2, t.data_ptr(),
                                     shape, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     static_cast<CUtensorMapSwizzle>(layout.swizzle), CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
              "encoding a TMA descriptor");
    put(&map, sizeof(map));
    i32(t.size(0)).i32(t.size(1));
    return i64(t.stride(0)).i64(1);
#endif
  }

  void operator()(int64_t x, int64_t y, int64_t z) {
    TORCH_CHECK(index_ == kernel_.args_.size(), "LTX native block: ", kernel_.function_name_, " got ", index_, " of ",
                kernel_.args_.size(), " arguments");
    const void* scratch = nullptr;
    put(&scratch, sizeof(scratch));
    put(&scratch, sizeof(scratch));
    std::vector<void*> params;
    for (auto& slot : slots_) params.push_back(slot.bytes);
    if (x * y * z == 0) return;
    check_gpu(gpu::launch(kernel_.function_, x, y, z, kernel_.threads_, kernel_.shared_,
                          at::cuda::getCurrentCUDAStream(), params.data()),
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
    gpu::bind_primary_context();
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
      attention = normed_self_attention([&] { return linear(attn_in, w[SelfQ]); },
                                        [&] { return linear(attn_in, w[SelfK]); }, v, q_norm, k_norm, rope_cos,
                                        rope_sin, cta_k);
    }
    at::Tensor h = project_gated(attention, linear(attn_in, w[SelfGate]), w[SelfOut], xin, ada[2], rows, fp4);

    // gated prompt cross-attention
    Activation query = modulated(h, ada[6], ada[7], fp4, rows);
#ifdef USE_ROCM
    const at::Tensor cross = normed_cross_attention(linear(query, w[CrossQ]), cross_q_norm, prompt_key, prompt_value);
#else
    // in the prompt key/value dtype; project_gated takes fp16 attention as it is
    const at::Tensor q = at::empty({rows, dim}, x.options().dtype(prompt_key.scalar_type()));
    norm_rope(linear(query, w[CrossQ]), cross_q_norm, nullptr, nullptr, q, head_dim, config_.eps);
    const at::Tensor q_heads = q.view({1, rows, heads, head_dim}).transpose(1, 2);
    const at::Tensor cross = at::scaled_dot_product_attention(q_heads, prompt_key, prompt_value)[0].transpose(0, 1);
#endif
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
#ifdef USE_ROCM
    TORCH_CHECK(cta_k == kHipAttentionCtaK, "AMD INT8 attention runs ", kHipAttentionCtaK, "-key tiles");
    const auto heads_first = [&](const at::Tensor& t) { return t.view({1, rows, heads, head_dim}).transpose(1, 2); };
    return int8_attention(heads_first(q), heads_first(k), heads_first(v))[0].transpose(0, 1);
#else
    const int L = static_cast<int>(rows), H = static_cast<int>(heads), D = static_cast<int>(head_dim);
    const int64_t padded = cdiv(L, cta_k) * cta_k;
    const auto i8 = q.options().dtype(at::kChar), f32 = q.options().dtype(at::kFloat);
    const at::Tensor q_int8 = at::empty({1, H, L, D}, i8), k_int8 = at::empty({1, H, L, D}, i8);
    const at::Tensor q_scale = at::empty({1, H, cdiv(L, 128) * (D == 256 ? 64 : 32)}, f32);
    const at::Tensor k_scale = at::empty({1, H, cdiv(L, cta_k) * 4}, f32);
    const at::Tensor v_int8 = at::empty({H * D, padded}, i8), v_scale = at::empty({H * D}, f32);
    const at::Tensor anchors = at::empty({1, H}, q.options().dtype(at::kInt));
    const at::Tensor out = at::empty({1, H, L, D}, q.options());
    const gpu::Stream stream = at::cuda::getCurrentCUDAStream();
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
#endif
  }

  // self_attention of the q/k projections (rows, heads * head_dim) after norm_rope, each projection
  // made when it is normalized; on AMD the norm runs inside comfy-kitchen's Q/K quantization
  // (norm_rope_quant.h), with the same result.
  template <class ProjectQ, class ProjectK>
  at::Tensor normed_self_attention(const ProjectQ& q_projection, const ProjectK& k_projection, const at::Tensor& v,
                                   const at::Tensor& q_norm, const at::Tensor& k_norm, const at::Tensor& rope_cos,
                                   const at::Tensor& rope_sin, int64_t cta_k) {
    const int64_t rows = v.size(0), heads = config_.heads, head_dim = v.size(1) / heads;
#ifdef USE_ROCM
    TORCH_CHECK(cta_k == kHipAttentionCtaK, "AMD INT8 attention runs ", kHipAttentionCtaK, "-key tiles");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16 && v.dim() == 2 && v.stride(1) == 1, "v must be (rows, heads * head_dim) bf16");
    const Int8QK qk = int8_qk(heads, rows, rows, head_dim, v.options());
    const gpu::Stream stream = at::cuda::getCurrentCUDAStream();
    sage_attention_hip::launch_norm_rope_quant_q(norm_rope_rows(q_projection(), q_norm, &rope_cos, &rope_sin, head_dim),
                                                 qk.q.data_ptr(), qk.q_scale.data_ptr(),
                                                 static_cast<int>(qk.padded_queries()), qk.rotation(), stream);
    const at::Tensor samples = at::empty({kAnchorSamples, v.size(1)}, v.options());
    sage_attention_hip::launch_norm_rope_quant_k(norm_rope_rows(k_projection(), k_norm, &rope_cos, &rope_sin, head_dim),
                                                 qk.k.data_ptr(), qk.k_scale.data_ptr(), samples.data_ptr(),
                                                 qk.anchors.data_ptr(), static_cast<int>(qk.key_groups()),
                                                 qk.rotation(), stream);
    return int8_attend(qk, v.view({1, rows, heads, head_dim}).transpose(1, 2))[0].transpose(0, 1);
#else
    const at::Tensor q = at::empty_like(v), k = at::empty_like(v);
    norm_rope(q_projection(), q_norm, &rope_cos, &rope_sin, q, head_dim, config_.eps);
    norm_rope(k_projection(), k_norm, &rope_cos, &rope_sin, k, head_dim, config_.eps);
    return self_attention(q, k, v, rows, heads, head_dim, cta_k);
#endif
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

#ifdef USE_ROCM
  // comfy-kitchen's HIP tiles: 128 queries, 64 keys, one key scale per 16 keys; nine anchor candidates.
  static constexpr int64_t kHipAttentionCtaQ = 128, kHipAttentionCtaK = 64, kHipKeyGroup = 16, kAnchorSamples = 9;

  // comfy-kitchen's INT8 Q/K of queries against keys, in the layouts launch_quant_qk_int8 writes.
  struct Int8QK {
    at::Tensor q, q_scale, k, k_scale, anchors;
    int64_t heads, queries, keys, head_dim;

    int64_t padded_queries() const { return cdiv(queries, kHipAttentionCtaQ) * kHipAttentionCtaQ; }
    int64_t padded_keys() const { return cdiv(keys, kHipAttentionCtaK) * kHipAttentionCtaK; }
    int64_t key_groups() const { return padded_keys() / kHipKeyGroup; }
    // comfy-kitchen's sage_rotation: short keys H4, longer ones the widest Hadamard of the head
    int rotation() const { return keys <= 256 ? 4 : head_dim >= 256 ? 129 : head_dim >= 128 ? 128 : 64; }
  };

  static Int8QK int8_qk(int64_t heads, int64_t queries, int64_t keys, int64_t head_dim, const at::TensorOptions& like) {
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256, "INT8 attention needs a head size of 64/128/256");
    Int8QK qk{{}, {}, {}, {}, {}, heads, queries, keys, head_dim};
    const auto i8 = like.dtype(at::kChar), f32 = like.dtype(at::kFloat);
    qk.q = at::empty({1, heads, queries, head_dim}, i8);
    qk.q_scale = at::empty({1, heads, qk.padded_queries()}, f32);
    qk.k = at::empty({1, heads, keys, head_dim}, i8);
    qk.k_scale = at::empty({1, heads, qk.key_groups()}, f32);
    qk.anchors = at::empty({1, heads}, like.dtype(at::kInt));
    return qk;
  }

  // comfy-kitchen's HIP INT8 attention of quantized q/k against v [1, heads, keys, d] (any token/head
  // strides, contiguous d), launched the way comfy_kitchen.backends.hip.sage_int8_sdpa launches it
  // unmasked: a contiguous [1, heads, queries, d] result.
  static at::Tensor int8_attend(const Int8QK& qk, const at::Tensor& v) {
    namespace sage = sage_attention_hip;
    TORCH_CHECK(v.scalar_type() == at::kBFloat16 && v.dim() == 4 && v.size(0) == 1 && v.size(1) == qk.heads &&
                v.size(2) == qk.keys && v.size(3) == qk.head_dim && v.stride(3) == 1,
                "INT8 attention values must be [1, heads, keys, head_dim] bf16 with a contiguous head_dim");
    const int64_t H = qk.heads, Lq = qk.queries, Lk = qk.keys, D = qk.head_dim, padded_k = qk.padded_keys();
    const at::Tensor v_int8 = at::empty({H * D, padded_k}, v.options().dtype(at::kChar));
    const at::Tensor v_scale = at::empty({H * D}, v.options().dtype(at::kFloat));
    const at::Tensor out = at::empty({1, H, Lq, D}, v.options());
    const gpu::Stream stream = at::cuda::getCurrentCUDAStream();
    const int h = static_cast<int>(H), lq = static_cast<int>(Lq), lk = static_cast<int>(Lk), d = static_cast<int>(D);
    sage::launch_quant_v_int8(v.data_ptr(), v_int8.data_ptr(), v_scale.data_ptr(), 1, h, lk, d,
                              static_cast<int>(padded_k), v.stride(0), v.stride(1), v.stride(2), sage::kBFloat16,
                              stream);
    sage::launch_int8_attn(qk.q.data_ptr(), qk.k.data_ptr(), v_int8.data_ptr(), out.data_ptr(), qk.q_scale.data_ptr(),
                           qk.k_scale.data_ptr(), v_scale.data_ptr(), nullptr, 0, 0, 0, 0, 0,
                           static_cast<int>(kHipAttentionCtaK), 1, lq, lk, static_cast<int>(qk.padded_queries()), h, h, d,
                           static_cast<int>(qk.key_groups()), H * Lq * D, Lq * D, H * Lk * D, Lk * D, H * D * padded_k,
                           D * padded_k, padded_k, H * Lq * D, Lq * D, D,
                           static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), sage::kBFloat16, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  // comfy-kitchen's HIP INT8 attention of q [1, heads, rows, d] against k/v [1, heads, keys, d]
  // (any token/head strides, contiguous d), as comfy_kitchen.backends.hip.sage_int8_sdpa runs it unmasked.
  static at::Tensor int8_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    namespace sage = sage_attention_hip;
    for (const auto* t : {&q, &k, &v})
      TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 4 && t->size(0) == 1 && t->stride(3) == 1,
                  "INT8 attention takes [1, heads, tokens, head_dim] bf16 with a contiguous head_dim");
    TORCH_CHECK(k.sizes() == v.sizes() && k.size(1) == q.size(1) && k.size(3) == q.size(3),
                "INT8 attention keys and values must match the query heads");
    const Int8QK qk = int8_qk(q.size(1), q.size(2), k.size(2), q.size(3), q.options());
    const int h = static_cast<int>(qk.heads);
    sage::launch_quant_qk_int8(q.data_ptr(), qk.q.data_ptr(), qk.q_scale.data_ptr(), k.data_ptr(), qk.k.data_ptr(),
                               qk.k_scale.data_ptr(), qk.anchors.data_ptr(), 1, h, static_cast<int>(qk.queries),
                               static_cast<int>(qk.padded_queries()), h, static_cast<int>(qk.keys),
                               static_cast<int>(qk.key_groups()), static_cast<int>(qk.head_dim), q.stride(0),
                               q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), sage::kBFloat16,
                               qk.rotation(), at::cuda::getCurrentCUDAStream());
    return int8_attend(qk, v);
  }

  // comfy-kitchen's HIP INT8 attention of the cross-attention queries after norm_rope (no RoPE) against
  // the prompt keys/values [1, heads, keys, d]: a (rows, heads, d) view.
  at::Tensor normed_cross_attention(const at::Tensor& q_projection, const at::Tensor& q_norm, const at::Tensor& key,
                                    const at::Tensor& value) const {
    TORCH_CHECK(key.scalar_type() == at::kBFloat16 && key.dim() == 4 && key.size(0) == 1 && key.stride(3) == 1 &&
                key.sizes() == value.sizes(), "prompt keys/values must be [1, heads, keys, head_dim] bf16");
    const int64_t rows = q_projection.size(0), heads = key.size(1), head_dim = key.size(3);
    const Int8QK qk = int8_qk(heads, rows, key.size(2), head_dim, key.options());
    const gpu::Stream stream = at::cuda::getCurrentCUDAStream();
    sage_attention_hip::launch_norm_rope_quant_q(norm_rope_rows(q_projection, q_norm, nullptr, nullptr, head_dim),
                                                 qk.q.data_ptr(), qk.q_scale.data_ptr(),
                                                 static_cast<int>(qk.padded_queries()), qk.rotation(), stream);
    sage_attention_hip::launch_quant_k_int8(key.data_ptr(), qk.k.data_ptr(), qk.k_scale.data_ptr(),
                                            qk.anchors.data_ptr(), static_cast<int>(heads), static_cast<int>(qk.keys),
                                            static_cast<int>(qk.key_groups()), static_cast<int>(head_dim),
                                            key.stride(1), key.stride(2), qk.rotation(), stream);
    return int8_attend(qk, value)[0].transpose(0, 1);
  }

  // norm_rope's inputs as norm_rope_quant.h takes them, checked as glue.cu's norm_rope checks them.
  sage_attention_hip::NormRope norm_rope_rows(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* cos,
                                              const at::Tensor* sin, int64_t head_dim) const {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous() &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be contiguous 16-byte aligned bf16 rows");
    const int64_t rows = x.size(0), dim = x.size(1), tokens = rows / 2;
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16 && weight.is_contiguous() && weight.numel() == dim &&
                dim % head_dim == 0, "norm_rope needs a (dim,) bf16 weight and whole heads");
    sage_attention_hip::NormRope p{x.data_ptr(), weight.data_ptr(), nullptr, nullptr, 0, 0, static_cast<int>(rows),
                                   static_cast<int>(std::max<int64_t>(tokens, 1)), static_cast<int>(dim / head_dim),
                                   static_cast<int>(head_dim), static_cast<float>(config_.eps)};
    if (cos != nullptr) {
      // [1, heads, rows / 2, head_dim / 2], innermost contiguous, both halves of the rows at the same positions
      TORCH_CHECK(cos->scalar_type() == at::kBFloat16 && sin->scalar_type() == at::kBFloat16 && cos->dim() == 4 &&
                  cos->size(1) * head_dim == dim && cos->size(2) == tokens && cos->size(3) * 2 == head_dim &&
                  cos->stride(3) == 1 && cos->strides() == sin->strides() && cos->stride(1) % 4 == 0 &&
                  cos->stride(2) % 4 == 0 && reinterpret_cast<uintptr_t>(cos->data_ptr()) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(sin->data_ptr()) % 8 == 0,
                  "rope tables must be [1, heads, rows / 2, head_dim / 2]");
      p.cos = cos->data_ptr();
      p.sin = sin->data_ptr();
      p.rope_stride_head = cos->stride(1);
      p.rope_stride_token = cos->stride(2);
    }
    return p;
  }
#endif

  // tensor_utils.half_dtype: no fused attention kernel takes bf16 before SM 8.0.
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

  // rotate_quantize of attention heads (heads, rows, head_dim) read as token rows, each head times
  // 2 * sigmoid(logits[row, head]) (kernels.gated_rotate_quantize).
  std::pair<at::Tensor, at::Tensor> gated_rotate_quantize(const at::Tensor& heads, const at::Tensor& logits) const {
    const int64_t count = heads.size(0), rows = heads.size(1), head_dim = heads.size(2), k = count * head_dim;
    TORCH_CHECK(heads.is_contiguous() && logits.is_contiguous() && logits.size(0) == rows && logits.size(1) == count,
                "gated_rotate_quantize needs contiguous (heads, rows, head_dim) heads and (rows, heads) logits");
    const at::Tensor codes = at::empty({rows, k}, heads.options().dtype(at::kChar));
    const at::Tensor scales = at::empty({rows}, heads.options().dtype(at::kFloat));
    Launch(kernel("gated_rotate_quantize", {{"K", k}, {"HEAD_DIM", head_dim}}))
        .ptr(heads).ptr(logits).ptr(codes).ptr(scales).i32(rows)(rows, 1, 1);
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
#ifdef USE_ROCM
    // as the AMD torch path: an attention output is gated and quantized straight from its heads
    if (!fp4 && attention.transpose(0, 1).is_contiguous()) {
      auto [codes, scales] = gated_rotate_quantize(attention.transpose(0, 1), logits);
      const at::Tensor out = at::empty_like(residual);
      return int8_mm(codes, scales, w, out, false, &residual, &gate, rows / 2, 0);
    }
#endif
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
at::Tensor Block::normed_self_attention(const at::Tensor& q_projection, const at::Tensor& k_projection,
                                        const at::Tensor& v, const at::Tensor& q_norm, const at::Tensor& k_norm,
                                        const at::Tensor& rope_cos, const at::Tensor& rope_sin, int64_t cta_k) {
  return impl_->normed_self_attention([&] { return q_projection; }, [&] { return k_projection; }, v, q_norm, k_norm,
                                      rope_cos, rope_sin, cta_k);
}
std::pair<at::Tensor, at::Tensor> Block::prompt_key_value(
    const std::vector<std::optional<at::Tensor>>& linear_tensors, const at::Tensor& cross_k_norm,
    const at::Tensor& prompt_scale_shift_table, const at::Tensor& prompt_rows, const at::Tensor& context) {
  return impl_->prompt_key_value(linear_tensors, cross_k_norm, prompt_scale_shift_table, prompt_rows, context);
}

}  // namespace jasna_ltx
