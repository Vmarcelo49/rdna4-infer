// M3 step 1 — device primitives vs the llama.cpp oracle.
//
// llama.cpp's `llama-eval-callback` dumps every named graph node with its first
// and last three values plus the element sum (see reference/oracle_hello_cpu.txt,
// captured with `-ngl 0` for determinism). This test reproduces the first two
// nodes of the qwen35 graph on the GPU and compares:
//
//   model.input_embed = GET_ROWS(token_embd.weight, token)      (llama.cpp)
//   norm-0            = RMS_NORM(model.input_embed)
//   attn_norm-0       = MUL(norm-0, blk.0.attn_norm.weight)
//
// usage: check-nn-gpu <file.gguf> <oracle_dump.txt> [token_id]
#include "rdna4/nn.cuh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "rdna4/dtype.h"
#include "rdna4/loader.h"
#include "rdna4/model.h"

extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem);

namespace {

// First three + last three values and the element sum, as printed by
// llama-eval-callback for a node.
struct OracleNode {
  std::vector<float> sampled;  // v0,v1,v2, v[n-3],v[n-2],v[n-1]
  double sum = 0.0;
  bool found = false;
  std::string shape;
};

OracleNode parse_oracle(const char *path, const std::string &name) {
  OracleNode out;
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot open oracle dump %s\n", path);
    return out;
  }
  std::string line;
  const std::string needle = "common_debug_cb_eval:";
  while (std::getline(f, line)) {
    if (line.find(needle) == std::string::npos) continue;
    // node label: everything between the marker and " = "
    const std::size_t eq = line.find(" = ");
    if (eq == std::string::npos) continue;
    std::string label = line.substr(needle.size(), eq - needle.size());
    while (!label.empty() && label.front() == ' ') label.erase(label.begin());
    while (!label.empty() && label.back() == ' ') label.pop_back();
    if (label != name) continue;

    out.found = true;
    const std::size_t lb = line.find('{', eq);
    const std::size_t rb = line.find('}', lb);
    if (lb != std::string::npos && rb != std::string::npos) out.shape = line.substr(lb, rb - lb + 1);

    // values follow on the subsequent lines until the "sum =" line
    std::vector<float> vals;
    while (std::getline(f, line)) {
      if (line.find("sum =") != std::string::npos) {
        out.sum = std::strtod(line.c_str() + line.find("sum =") + 5, nullptr);
        break;
      }
      const char *p = line.c_str();
      char *endp = nullptr;
      while (*p) {
        const double v = std::strtod(p, &endp);
        if (endp == p) {
          ++p;
          continue;
        }
        vals.push_back((float)v);
        p = endp;
      }
    }
    // the dump prints [v0, v1, v2, ..., v[n-3], v[n-2], v[n-1]]
    if (vals.size() >= 6) {
      out.sampled.assign(vals.begin(), vals.begin() + 3);
      out.sampled.insert(out.sampled.end(), vals.end() - 3, vals.end());
    } else {
      out.sampled = vals;
    }
    return out;
  }
  return out;
}

struct Compare {
  double max_abs = 0;
  double rel_sum = 0;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: check-nn-gpu <file.gguf> <oracle_dump.txt> [token_id]\n");
    return 2;
  }
  const std::int64_t token_id = argc >= 4 ? std::atoll(argv[3]) : 9419;  // "Hello"

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open %s: %s\n", argv[1], err.c_str());
    return 1;
  }

  const auto *emb = ld.find("token_embd.weight");
  const auto *nrm = ld.find("blk.0.attn_norm.weight");
  if (!emb || !nrm) {
    std::fprintf(stderr, "missing token_embd.weight / blk.0.attn_norm.weight\n");
    return 1;
  }
  const std::size_t i_emb = (std::size_t)(emb - ld.meta().tensors.data());
  const std::size_t i_nrm = (std::size_t)(nrm - ld.meta().tensors.data());

  // config-derived dims
  rdna4::Qwen35Config cfg{};
  if (!rdna4::parse_qwen35_config(ld.meta(), cfg, err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 1;
  }
  const std::int64_t n_embd = (std::int64_t)cfg.embedding_length;
  if (emb->dims.size() != 2 || emb->dims[0] != n_embd) {
    std::fprintf(stderr, "unexpected token_embd dims\n");
    return 1;
  }
  if (token_id < 0 || token_id >= emb->dims[1]) {
    std::fprintf(stderr, "token %lld out of range\n", (long long)token_id);
    return 1;
  }

  // ---- CPU side: the embedding row for `token_id`, dequantized with the oracle
  const std::uint32_t be = rdna4::dtype_block_elems(ld.dtype(i_emb));
  const std::uint32_t bb = rdna4::dtype_block_bytes(ld.dtype(i_emb));
  const std::uint64_t row_bytes = (std::uint64_t)(n_embd / be) * bb;
  std::vector<std::uint8_t> raw;
  if (!ld.load_tensor_range(i_emb, (std::uint64_t)token_id * row_bytes, row_bytes, raw, err)) {
    std::fprintf(stderr, "load embed row: %s\n", err.c_str());
    return 1;
  }
  std::vector<float> x(n_embd);
  if (!rdna4_cpu_dequant_row((int)ld.dtype(i_emb), raw.data(), x.data(), n_embd)) {
    std::fprintf(stderr, "no CPU oracle for the embedding dtype\n");
    return 1;
  }
  std::vector<float> w(n_embd);
  if (!rdna4_cpu_dequant_row((int)ld.dtype(i_nrm), (const std::uint8_t *)nullptr, w.data(), 0)) {
    // attn_norm is F32: read it straight
  }
  {
    std::vector<std::uint8_t> wraw;
    if (!ld.load_tensor(i_nrm, wraw, err)) {
      std::fprintf(stderr, "load attn_norm: %s\n", err.c_str());
      return 1;
    }
    if (ld.dtype(i_nrm) != rdna4::DType::F32 || wraw.size() < (std::size_t)n_embd * 4) {
      std::fprintf(stderr, "blk.0.attn_norm.weight is not F32[%lld] (dtype %s)\n", (long long)n_embd,
                   rdna4::dtype_name(ld.dtype(i_nrm)));
      return 1;
    }
    std::memcpy(w.data(), wraw.data(), (std::size_t)n_embd * sizeof(float));
  }

  // ---- GPU side
  float *d_x = nullptr, *d_w = nullptr, *d_norm = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_x, n_embd * sizeof(float)) != hipSuccess ||
      hipMalloc(&d_w, n_embd * sizeof(float)) != hipSuccess ||
      hipMalloc(&d_norm, n_embd * sizeof(float)) != hipSuccess ||
      hipMalloc(&d_out, n_embd * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed\n");
    return 1;
  }
  hipMemcpy(d_x, x.data(), n_embd * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_w, w.data(), n_embd * sizeof(float), hipMemcpyHostToDevice);

  const float eps = (float)cfg.rms_norm_eps;
  if (!rdna4::rms_norm_launch(d_x, nullptr, d_norm, 1, n_embd, eps) ||
      !rdna4::mul_launch(d_norm, d_w, d_out, n_embd)) {
    std::fprintf(stderr, "kernel launch failed\n");
    return 1;
  }
  std::vector<float> g_norm(n_embd), g_out(n_embd);
  hipMemcpy(g_norm.data(), d_norm, n_embd * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(g_out.data(), d_out, n_embd * sizeof(float), hipMemcpyDeviceToHost);

  // ---- compare with the oracle
  int failures = 0;
  auto check_node = [&](const char *name, const std::vector<float> &gpu) {
    const OracleNode o = parse_oracle(argv[2], name);
    if (!o.found) {
      std::printf("%-14s NOT FOUND in %s\n", name, argv[2]);
      ++failures;
      return;
    }
    double gsum = 0.0;
    for (float v : gpu) gsum += v;
    const double rel_sum = std::fabs(gsum - o.sum) / (std::fabs(o.sum) > 1e-6 ? std::fabs(o.sum) : 1.0);
    double max_abs = 0;
    const std::int64_t n = (std::int64_t)gpu.size();
    const std::int64_t idx[6] = {0, 1, 2, n - 3, n - 2, n - 1};
    std::printf("%-14s %s  oracle[", name, o.shape.c_str());
    for (int k = 0; k < 6 && k < (int)o.sampled.size(); ++k) {
      const double d = std::fabs((double)gpu[idx[k]] - (double)o.sampled[k]);
      max_abs = std::fmax(max_abs, d);
      std::printf("%s%.4f/%.4f", k ? " " : "", o.sampled[k], (double)gpu[idx[k]]);
    }
    const bool ok = max_abs < 2e-3 && rel_sum < 2e-3;
    std::printf("]  sum %.6f vs %.6f (rel %.2e)  max|d|=%.2e  %s\n", o.sum, gsum, rel_sum, max_abs,
                ok ? "OK" : "MISMATCH");
    if (!ok) ++failures;
  };

  check_node("norm-0", g_norm);
  check_node("attn_norm-0", g_out);

  hipFree(d_x);
  hipFree(d_w);
  hipFree(d_norm);
  hipFree(d_out);

  std::printf("check-nn-gpu: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
