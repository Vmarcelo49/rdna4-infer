// rdna4-infer (PLAN.md M0): device gate + VRAM budget check. Full CLI lands in M4.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

#include "rdna4/device.h"
#include "rdna4/loader.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: rdna4-infer [-m model.gguf] [--ctx-size N] [--cache-type-k TYPE] "
               "[--cache-type-v TYPE] [--list-tensors]\n  cache types: f16 (default), q8_0, q4_0\n");
}

bool valid_kv_type(const char *s) {
  return std::strcmp(s, "f16") == 0 || std::strcmp(s, "q8_0") == 0 ||
         std::strcmp(s, "q4_0") == 0;
}

bool parse_u64(const char *s, std::uint64_t *out) {
  char *end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') {
    return false;
  }
  *out = static_cast<std::uint64_t>(v);
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  const char *model_path = nullptr;
  std::uint64_t ctx_size = 4096;
  const char *cache_k = "f16";
  const char *cache_v = "f16";
  bool list_tensors = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
      if (!parse_u64(argv[++i], &ctx_size) || ctx_size == 0) {
        usage();
        return 1;
      }
    } else if (std::strcmp(argv[i], "--cache-type-k") == 0 && i + 1 < argc) {
      cache_k = argv[++i];
      if (!valid_kv_type(cache_k)) {
        usage();
        return 1;
      }
    } else if (std::strcmp(argv[i], "--cache-type-v") == 0 && i + 1 < argc) {
      cache_v = argv[++i];
      if (!valid_kv_type(cache_v)) {
        usage();
        return 1;
      }
    } else if (std::strcmp(argv[i], "--list-tensors") == 0) {
      list_tensors = true;
    } else {
      usage();
      return 1;
    }
  }

  char arch[64] = {};
  std::size_t total_vram = 0;
  if (rdna4::query_device(arch, sizeof(arch), &total_vram) != 0) {
    std::fprintf(stderr, "rdna4-infer: no HIP device found\n");
    return 1;
  }
  std::printf("device arch: %s\nvram total: %.2f GiB\n", arch,
              static_cast<double>(total_vram) / (1u << 30));
  if (!rdna4::is_gfx1201(arch)) {
    std::fprintf(stderr, "rdna4-infer: requires gfx1201, found %s\n", arch);
    return 2;
  }

  if (model_path == nullptr) {
    std::printf("rdna4-infer: gfx1201 OK (no model given, budget check skipped)\n");
    return 0;
  }

  struct stat st;
  if (stat(model_path, &st) != 0) {
    std::fprintf(stderr, "rdna4-infer: cannot stat %s\n", model_path);
    return 1;
  }
  const auto file_bytes = static_cast<std::uint64_t>(st.st_size);

  rdna4::GgufLoader loader;
  std::string gerr;
  if (!loader.open(model_path, gerr)) {
    std::fprintf(stderr, "rdna4-infer: gguf load failed: %s\n", gerr.c_str());
    return 1;
  }
  const rdna4::gguf::File &gf = loader.meta();
  std::printf("gguf v%u: arch=%s tensors=%llu kv=%llu data=%.2f GiB\n", gf.version,
              rdna4::gguf::kv_str(gf, "general.architecture", "?").c_str(),
              (unsigned long long)gf.tensors.size(), (unsigned long long)gf.kv.size(),
              static_cast<double>(loader.total_bytes()) / (1u << 30));
  if (list_tensors) {
    for (std::size_t i = 0; i < gf.tensors.size(); ++i) {
      const auto &t = gf.tensors[i];
      std::printf("  %-40s %-8s [", t.name.c_str(), rdna4::dtype_name(loader.dtype(i)));
      for (std::size_t d = 0; d < t.dims.size(); ++d) {
        std::printf("%s%lld", d ? ", " : "", (long long)t.dims[d]);
      }
      std::printf("]\n");
    }
    return 0;
  }
  const double gib = static_cast<double>(1u << 30);
  const auto kv_k =
      static_cast<std::uint64_t>(ctx_size * rdna4::kQwen35KvElemsPerToken *
                                 rdna4::kv_bytes_per_elem(cache_k) / 2);
  const auto kv_v =
      static_cast<std::uint64_t>(ctx_size * rdna4::kQwen35KvElemsPerToken *
                                 rdna4::kv_bytes_per_elem(cache_v) / 2);
  const std::uint64_t need = file_bytes + kv_k + kv_v + rdna4::kOverheadBytes;
  std::printf("model: %.2f GiB + kv(ctx=%llu, k=%s/v=%s): %.2f GiB + overhead: %.2f GiB = need %.2f GiB\n",
              file_bytes / gib, (unsigned long long)ctx_size, cache_k, cache_v,
              (kv_k + kv_v) / gib, rdna4::kOverheadBytes / gib, need / gib);
  if (need > total_vram) {
    std::fprintf(stderr, "rdna4-infer: insufficient VRAM (need %.2f GiB)\n", need / gib);
    return 3;
  }
  std::printf("rdna4-infer: budget OK\n");
  return 0;
}
