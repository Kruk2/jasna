#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jasna_ltx {

struct TensorDescriptorLayout {
  int swizzle, elem_size, elem_type;
  std::vector<int64_t> block;
};

struct KernelSpec {
  std::string family;
  std::map<std::string, int64_t> constexprs;
  std::string function;
  int shared = 0, warps = 0;
  std::vector<std::string> args;
  std::vector<bool> divisible;
  std::vector<TensorDescriptorLayout> tensordesc;
};

struct KernelImage {
  KernelSpec spec;
  std::vector<unsigned char> cubin;
};

inline bool kernel_identifier(const std::string& value) {
  if (value.empty()) return false;
  for (unsigned char c : value)
    if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') && c != '_')
      return false;
  return true;
}

inline void validate_kernel_spec(const KernelSpec& spec) {
  if (!kernel_identifier(spec.family) || !kernel_identifier(spec.function) || spec.shared < 0 ||
      spec.warps <= 0 || spec.warps > 32 || spec.args.empty() || spec.args.size() != spec.divisible.size())
    throw std::invalid_argument("LTX native block: invalid kernel descriptor");
  for (const auto& [name, value] : spec.constexprs)
    if (!kernel_identifier(name)) throw std::invalid_argument("LTX native block: invalid kernel constant name");
  size_t descriptors = 0;
  for (const auto& arg : spec.args) {
    if (arg.empty() || arg.find('\0') != std::string::npos)
      throw std::invalid_argument("LTX native block: invalid kernel argument");
    descriptors += arg == "nvTmaDesc";
  }
  if (descriptors != spec.tensordesc.size())
    throw std::invalid_argument("LTX native block: missing tensor descriptor layout");
  for (const auto& descriptor : spec.tensordesc)
    if (descriptor.block.size() != 2 || descriptor.block[0] <= 0 || descriptor.block[1] <= 0 ||
        descriptor.elem_size <= 0)
      throw std::invalid_argument("LTX native block: invalid tensor descriptor layout");
}

inline std::string kernel_key(const std::string& family, const std::map<std::string, int64_t>& constexprs) {
  std::string result = family;
  for (const auto& [name, value] : constexprs) result += "|" + name + "=" + std::to_string(value);
  return result;
}

inline void validate_kernel_images(const std::vector<KernelImage>& kernels) {
  if (kernels.empty()) throw std::invalid_argument("LTX native block: no kernel images");
  std::set<std::string> identities;
  for (const auto& image : kernels) {
    validate_kernel_spec(image.spec);
    if (image.cubin.empty()) throw std::invalid_argument("LTX native block: empty kernel image");
    if (!identities.insert(kernel_key(image.spec.family, image.spec.constexprs)).second)
      throw std::invalid_argument("LTX native block: duplicate kernel identity");
  }
}

class KernelSource {
 public:
  enum class Kind { Path, Image };
  static KernelSource from_path(std::string path) {
    if (path.empty() || path.find('\0') != std::string::npos)
      throw std::invalid_argument("LTX native block: invalid kernel path");
    return KernelSource(Kind::Path, std::move(path), {});
  }
  static KernelSource from_image(std::vector<unsigned char> image) {
    if (image.empty()) throw std::invalid_argument("LTX native block: empty kernel image");
    return KernelSource(Kind::Image, {}, std::move(image));
  }
  Kind kind() const { return kind_; }
  const std::string& path() const { return path_; }
  const std::vector<unsigned char>& image() const { return image_; }

 private:
  KernelSource(Kind kind, std::string path, std::vector<unsigned char> image)
      : kind_(kind), path_(std::move(path)), image_(std::move(image)) {}
  Kind kind_;
  std::string path_;
  std::vector<unsigned char> image_;
};

}  // namespace jasna_ltx
