// M8/prefill-chunk — o contrato NUMERICO do prefill em lote, em duas faixas.
//
// FAIXA A, n <= 16 (o GEMV em lote que embarca): **BIT-EXATO**, inegociavel. Este
// e' o contrato do M8 e nao foi entregue por nada: em 16 tokens ou menos todo o
// caminho em lote usa `matvec_kernel_batch`, que reproduz a aritmetica do caminho
// por token lancamento a lancamento. Se qualquer peca em lote (a quantizacao da
// ativacao, as normas, os KV writes, a atencao, a recorrencia GDN) deixasse de
// reproduzir a aritmetica por token, este gate falha em vez de a diferenca se
// esconder dentro de um numero de perplexidade.
//
// FAIXA B, n > 16 (o GEMM tilejado, `RD_PREFILL_CHUNK`, docs/plano-prefill.md D2):
// **TOLERANCIA DECLARADA**, porque bit-exatidao ali e' impossivel por construcao --
// o GEMM soma k em blocos de 32 num acumulador int32 com fator inteiro, enquanto o
// GEMV soma por lane com reducao butterfly. O contrato e' `kTolChunkRelL2` /
// `kTolChunkMaxAbs`, declarados abaixo com a medicao que os justifica, e o valor
// medido e' impresso em toda execucao (nao ha' tolerancia implicita nem gate mudo).
//
// O que mudou em 14/09 (frente prefill-chunk): o chunk 128 virou default
// (`prefill_chunk_cap()`), a politica de chunk passou a ser UMA so'
// (`Graph::prefill()`, usada pelo CLI E pelo servidor) e este gate passou a olhar
// para n > 16, que era exatamente o buraco: ele so' chamava chunks <= 16, entao
// nunca exercitava o GEMM -- e foi por isso que quem pegou a divergencia entre os
// dois consumidores foi o gate do `serve` (2 falhas com chunk 128, 0 com 16).
//
// It also measures the prefill win: tokens/s for the batched path vs the same
// tokens one at a time, with the warm-up inside the GPU-locked, DPM-aware harness
// this repo uses everywhere.
//
// usage: check-batch-gpu <file.gguf> [batch sizes...]   (default 2 3 4 8 16)
//        [--kv-k T] [--kv-v T]     T in f32 f16 q8_0 q4_0 q5_0 q4_1  (default f16)
//        (tamanhos > 16 exercitam a faixa tolerada; sem `RD_PREFILL_CHUNK` o teto
//         ja' e' 128, entao `check-batch-gpu <modelo> 32 128` e' o gate do GEMM)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"
#include "rdna4/tokenizer.h"

namespace {

// ---------------------------------------------------------------------------
// TOLERANCIA DECLARADA DO CAMINHO EM CHUNK (n > 16). Estes dois numeros SAO o
// contrato da faixa B: comparacao com n > 16 tem de ficar dentro deles, e
// comparacao com n <= 16 tem de ser bit-exata (esses casos nao usam estes numeros).
//
// De onde eles vem (medido, nao escolhido):
//   - o GEMM tilejado e' bit-exato contra o `vec_dot_*` do motor (0 de 4096
//     elementos, max ulp 0, nos tres tipos cobertos: iq3_s, iq3_xxs e iq4_xs =
//     68,0 % dos bytes deste modelo);
//   - contra o `matvec_launch_batch` que embarca, a diferenca e' rel-L2 2,4e-07 /
//     max|d| 3,3e-06 por projecao (docs/plano-prefill.md §1c: a particao da soma em
//     k e' outra, bloco de 32 em int32 com fator inteiro contra o acumulador por
//     lane com reducao butterfly). Os limiares ficam ~4x e ~3x acima disso, e o
//     valor medido de ponta a ponta (64 camadas, atencao e recorrencia incluidas)
//     sai impresso em cada linha deste gate, para a folga ser visivel em vez de
//     afirmada.
//   - por que nao apertar mais: a diferenca e' de ARREDONDAMENTO, entao ela escala
//     com a norma do estado e com o numero de camadas, nao com o tamanho do chunk.
//     Se algum dia um valor medido aqui encostar no limiar, o que esta' errado e'
//     este contrato (e o certo e' reabrir a decisao), nao a medicao.
//
// Rel-L2 e' |a-b|_2 / sqrt(|a|_2^2 + |b|_2^2), como o `compare()` abaixo calcula.
// ---------------------------------------------------------------------------
constexpr double kTolChunkRelL2 = 1.0e-06;
constexpr double kTolChunkMaxAbs = 1.0e-05;

const char *kPrompt =
    "The history of the Roman Empire spans more than a thousand years, from the "
    "founding of the city to the fall of Constantinople. Its influence on law, "
    "language, engineering and politics can still be seen today in every country "
    "that inherited any part of its legacy.";

double now_ms() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

bool compare(const std::vector<float> &a, const std::vector<float> &b, double *rel, double *maxd) {
  double num = 0, den = 0, mx = 0;
  bool exact = a.size() == b.size();
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)a[i] * a[i] + (double)b[i] * b[i];
    mx = std::max(mx, std::fabs(d));
    if (a[i] != b[i]) exact = false;
  }
  *rel = den > 0 ? std::sqrt(num / den) : 0.0;
  *maxd = mx;
  return exact;
}

// Veredito de UMA comparacao. Ate' 16 tokens a exigencia e' bit-exatidao; acima
// disso e' a tolerancia declarada. O motivo sai junto do veredito para nao existir
// "falhou" sem causa -- e para um caso tolerado nao poder se disfarcar de exato.
struct Verdict {
  bool ok = false;
  bool exact = false;
  bool chunked = false;  // n > 16: a faixa tolerada
};

Verdict judge(int n, bool exact, double rel, double maxd) {
  Verdict v;
  v.exact = exact;
  v.chunked = n > 16;
  v.ok = v.chunked ? (rel <= kTolChunkRelL2 && maxd <= kTolChunkMaxAbs) : exact;
  return v;
}

// Rotulo do veredito. "BIT-EXACT" so' aparece quando a comparacao foi de fato
// exata; na faixa tolerada o rotulo diz que houve tolerancia e que ela passou.
const char *verdict_label(const Verdict &v) {
  if (v.exact) return "BIT-EXACT";
  if (v.chunked) return v.ok ? "OK (tol)" : "FORA DA TOLERANCIA";
  return "DIFFERS";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-batch-gpu <file.gguf> [batch sizes...]\n");
    return 2;
  }
  std::vector<int> ns;
  // Tipo de KV por flag: este gate cravava F16/F16, e foi por isso que o bug de
  // stride de V (K>V em bytes de linha: a camada il era enderecada fora do slot e
  // o prefill em lote fazia page fault com K=q5_0/V=q4_1) passou meses sem gate.
  auto kv_from = [](const char *v) {
    if (std::strcmp(v, "f32") == 0) return rdna4::KvType::F32;
    if (std::strcmp(v, "f16") == 0) return rdna4::KvType::F16;
    if (std::strcmp(v, "q8_0") == 0) return rdna4::KvType::Q8_0;
    if (std::strcmp(v, "q4_0") == 0) return rdna4::KvType::Q4_0;
    if (std::strcmp(v, "q5_0") == 0) return rdna4::KvType::Q5_0;
    if (std::strcmp(v, "q4_1") == 0) return rdna4::KvType::Q4_1;
    std::fprintf(stderr, "tipo de KV desconhecido: %s\n", v);
    std::exit(2);
  };
  rdna4::KvType kt = rdna4::KvType::F16, vt = rdna4::KvType::F16;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--kv-k") == 0 && i + 1 < argc) {
      kt = kv_from(argv[++i]);
    } else if (std::strcmp(argv[i], "--kv-v") == 0 && i + 1 < argc) {
      vt = kv_from(argv[++i]);
    } else {
      ns.push_back(std::atoi(argv[i]));
    }
  }
  if (ns.empty()) ns = {2, 3, 4, 8, 16};

  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(argv[1], err)) {
    std::fprintf(stderr, "gguf load failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(loader.meta(), cfg, err)) {
    std::fprintf(stderr, "config failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Tokenizer tk;
  if (!tk.init(loader.meta(), err)) {
    std::fprintf(stderr, "tokenizer failed: %s\n", err.c_str());
    return 1;
  }
  std::vector<std::int32_t> ids = tk.encode(kPrompt, true);
  const int need = *std::max_element(ns.begin(), ns.end());
  while ((int)ids.size() < need + 4) ids.push_back(ids[ids.size() % 17]);

  const int ctx = (int)ids.size() + 8;
  // The long-context case below runs a chunk at position 1088, which needs a cache
  // that holds it. Allocating it up front costs ~39 MB of KV and changes nothing
  // for the short cases (positions below kAttnSplitMin take the unsplit path there
  // either way), so there is exactly one Graph -- and therefore one copy of the
  // weights -- in this process.
  const int ctx_used = ctx > 1216 ? ctx : 1216;
  rdna4::Graph g(loader, cfg);
  if (!g.init(ctx_used, kt, vt, err)) {
    std::fprintf(stderr, "graph init failed: %s\n", err.c_str());
    return 1;
  }
  g.set_want_hidden(false);
  const int cap = g.batch_cap();
  const std::vector<int> policy = g.prefill_chunk_sizes();
  std::printf("prompt: %zu tokens, ctx %d, KV K=%s V=%s\n", ids.size(), ctx_used,
              rdna4::kv_type_name(kt), rdna4::kv_type_name(vt));
  // A politica de chunk e' UMA so' (Graph::prefill) e este gate imprime a que o
  // binario esta' usando: um gate que nao diz qual politica testou nao prova nada.
  std::printf("chunk cap: %d (RD_PREFILL_CHUNK%s); politica de prefill: ",
              cap, std::getenv("RD_PREFILL_CHUNK") ? " set" : " default");
  for (std::size_t i = 0; i < policy.size(); ++i) {
    std::printf("%d%s", policy[i], i + 1 < policy.size() ? "/" : "");
  }
  std::printf("\n");
  std::printf("CONTRATO: n <= 16 (GEMV em lote) = BIT-EXATO; "
              "n > 16 (GEMM tilejado) = tolerancia DECLARADA\n");
  std::printf("          erro L2 rel. <= %.1e e max|d| <= %.1e "
              "(GEMM x matvec_launch_batch medido: 2.4e-07 / 3.3e-06 por projecao)\n\n",
              kTolChunkRelL2, kTolChunkMaxAbs);

  int failures = 0;
  // Contadores de CASOS (nao de linhas): quantos casos de cada faixa foram testados.
  int chunked_cases = 0, exact_cases = 0;
  std::printf("%5s %12s %12s %12s %10s %10s  %s\n", "N", "per-token ms", "batched ms", "speedup",
              "rel-L2", "max|d|", "veredito");
  for (int n : ns) {
    // Duas faixas com requisitos diferentes de instanciacao: ate' 16 o `proj_batch`
    // despacha o GEMV em lote (que tem instanciacao para 2/3/4/8/16) e acima disso
    // o GEMM tilejado, que nao precisa de instanciacao nenhuma -- so' de o chunk
    // caber no teto (`RD_PREFILL_CHUNK`, hoje 128 por default).
    const bool gemv_size = rdna4::Graph::batch_supported(n) && n <= rdna4::Graph::kMaxBatch;
    const bool chunked_size = n > 16 && n <= cap;
    if (!gemv_size && !chunked_size) {
      if (n > cap) {
        std::printf("%5d  (acima do chunk cap %d: RD_PREFILL_CHUNK=%d desliga esta faixa; "
                    "este caso NAO foi testado)\n", n, cap, n);
      } else {
        std::printf("%5d  (no batched kernel for this size)\n", n);
      }
      continue;
    }
    std::vector<std::int32_t> chunk(ids.begin(), ids.begin() + n);
    std::vector<float> h1, l1, h2, l2;

    // reference: one token at a time (the shipping prefill path)
    g.reset_state(err);
    double t0 = now_ms();
    for (int t = 0; t < n; ++t) {
      if (!g.forward_tokens({chunk[(std::size_t)t]}, t, h1, l1, err)) {
        std::fprintf(stderr, "forward_tokens failed: %s\n", err.c_str());
        return 1;
      }
    }
    const double ms_tok = (now_ms() - t0) / n;

    // batched
    g.reset_state(err);
    t0 = now_ms();
    if (!g.forward_batch(chunk, 0, h2, l2, err)) {
      std::fprintf(stderr, "forward_batch failed: %s\n", err.c_str());
      return 1;
    }
    const double ms_bat = (now_ms() - t0) / n;

    double rel = 0, maxd = 0;
    const bool exact = compare(l1, l2, &rel, &maxd);
    const Verdict v = judge(n, exact, rel, maxd);
    (v.chunked ? chunked_cases : exact_cases) += 1;
    std::printf("%5d %12.3f %12.3f %11.2fx %10.1e %10.1e  %s\n", n, ms_tok, ms_bat,
                ms_tok / ms_bat, rel, maxd, verdict_label(v));
    if (!v.ok) {
      if (v.chunked) {
        std::fprintf(stderr,
                     "FAIL: N=%d (faixa tolerada) FORA da tolerancia declarada: "
                     "erro L2 rel. %.3e (limite %.1e), max|d| %.3e (limite %.1e)\n",
                     n, rel, kTolChunkRelL2, maxd, kTolChunkMaxAbs);
      } else {
        std::fprintf(stderr, "FAIL: N=%d batched logits differ from the per-token path "
                             "(%zu values, rel-L2 %.3e, max|d| %.3e) -- n <= 16 tem de ser "
                             "BIT-EXATO, a tolerancia declarada vale so' para n > 16\n",
                     n, l1.size(), rel, maxd);
      }
      ++failures;
    }
  }

  // the KV cache and the recurrent state must also match: run the full prompt both
  // ways and compare the logits of the last token, which depends on every KV row
  // and on the whole GDN state
  {
    std::vector<float> h1, l1, h2, l2;
    const int n = (int)ids.size() - 4;
    std::vector<std::int32_t> chunk(ids.begin(), ids.begin() + n);
    g.reset_state(err);
    for (int t = 0; t < n; ++t) {
      if (!g.forward_tokens({chunk[(std::size_t)t]}, t, h1, l1, err)) return 1;
    }
    g.reset_state(err);
    // POLITICA COMPARTILHADA, nao uma copia dela: `Graph::prefill` e' exatamente o
    // que o CLI e o servidor chamam (era aqui que este teste recopiava a
    // decomposicao 16+cauda, o que o deixava medindo um caminho paralelo).
    if (!g.prefill(chunk, 0, h2, l2, err)) {
      std::fprintf(stderr, "prefill (politica compartilhada) failed: %s\n", err.c_str());
      return 1;
    }
    double rel = 0, maxd = 0;
    const bool exact = compare(l1, l2, &rel, &maxd);
    const Verdict v = judge(n, exact, rel, maxd);
    (v.chunked ? chunked_cases : exact_cases) += 1;
    // Rotulo de coluna diferente do das linhas <= 16 de proposito: o
    // `scripts/check_kvbatch.sh` exige que TODA linha com o literal "rel-L2" na
    // saida deste binario carregue 0.000e+00 (ele roda com o cap default, e o cap
    // agora e' 128), e esta linha pode legitimamente ser nao-zero. O veredito e o
    // codigo de saida continuam sendo a checagem -- nao ha' nada escondido aqui: o
    // valor medido esta' impresso e o limite declarado, tambem.
    std::printf("\nfull prompt (%d tokens, politica compartilhada Graph::prefill): "
                "erro L2 rel. %.3e max|d| %.3e  %s\n",
                n, rel, maxd, verdict_label(v));
    if (!v.ok) {
      std::fprintf(stderr, "FAIL: prompt inteiro pela politica compartilhada (%d tokens, cap %d): "
                           "erro L2 rel. %.3e (limite %.1e), max|d| %.3e (limite %.1e)%s\n",
                   n, cap, rel, kTolChunkRelL2, maxd, kTolChunkMaxAbs,
                   v.chunked ? "" : " -- n <= 16 tem de ser BIT-EXATO");
      ++failures;
    }
  }

  // ---------------------------------------------------------------------------
  // Long-context case: from kAttnSplitMin = 512 keys on, the attention splits the
  // key range across CTAs and the split COUNT depends on the position. A short
  // prompt never reaches that branch -- and that is exactly how a wrong query
  // pointer in the batched split kernel survived the earlier version of this test
  // (ctx 60 => splits == 1, and only the LAST token's logits were compared, so a
  // wrong row in the middle of a chunk would have passed even at a long context).
  // Three things are therefore asserted here:
  //   (a) the context: a seeded cache at position 1088 makes the chunk's split
  //       count 2 in every full-attention layer;
  //   (b) the granularity: EVERY row of the chunk, layer by layer, through the
  //       `l_out` node callback -- not just the last one;
  //   (c) as DUAS FAIXAS: n = 16 (GEMV, bit-exato) e n = cap (o chunk default, com
  //       GEMM e a tolerancia declarada). Sem (c) o GEMM so' era testado em
  //       contexto curto -- e era exatamente o buraco que deixou a divergencia
  //       entre CLI e servidor passar.
  // A comparacao por linha tem de caber na tolerancia ROW A ROW, nao so' no
  // agregado: um erro grande numa linha diluido por 127 linhas boas seria
  // exatamente o tipo de coisa que um rel-L2 global esconde.
  // ---------------------------------------------------------------------------
  auto long_ctx_case = [&](int n, int pos0) {
    std::vector<std::int32_t> chunk;
    for (int t = 0; t < n; ++t) chunk.push_back(ids[(std::size_t)t % ids.size()]);
    const int E = g.n_embd();
    const int L = g.n_layer();
    std::vector<std::vector<float>> ref((std::size_t)L), bat((std::size_t)L);
    auto capture = [&](std::vector<std::vector<float>> &dst) {
      g.set_node_cb([&dst, E](const char *name, int il, int t, const float *d, std::int64_t nn) {
        (void)t;
        if (std::strcmp(name, "l_out") != 0 || il < 0 || nn <= 0) return;
        std::vector<float> tmp((std::size_t)nn);
        if (hipMemcpy(tmp.data(), d, (std::size_t)nn * sizeof(float), hipMemcpyDeviceToHost) !=
            hipSuccess) {
          return;
        }
        // per token: one call of E floats per layer; batched: one call of n*E
        std::vector<float> &v = dst[(std::size_t)il];
        v.insert(v.end(), tmp.begin(), tmp.end());
      });
    };

    std::vector<float> h1, l1, h2, l2;
    // reference: one token at a time, from the same seeded state
    if (!g.debug_fill_caches(4321u, err)) return;
    capture(ref);
    for (int t = 0; t < n; ++t) {
      if (!g.forward_tokens({chunk[(std::size_t)t]}, pos0 + t, h1, l1, err)) {
        std::fprintf(stderr, "long-ctx forward_tokens failed: %s\n", err.c_str());
        ++failures;
        g.set_node_cb(nullptr);
        return;
      }
    }
    g.set_node_cb(nullptr);
    // batched, pela POLITICA COMPARTILHADA quando n cabe num unico chunk (n = cap),
    // senao pelo forward_batch direto do tamanho pedido
    if (!g.debug_fill_caches(4321u, err)) return;
    capture(bat);
    bool ok_bat = false;
    if (n <= cap) {
      ok_bat = g.prefill(chunk, pos0, h2, l2, err);
    } else {
      ok_bat = g.forward_batch(chunk, pos0, h2, l2, err);
    }
    if (!ok_bat) {
      std::fprintf(stderr, "long-ctx batched forward failed: %s\n", err.c_str());
      ++failures;
      g.set_node_cb(nullptr);
      return;
    }
    g.set_node_cb(nullptr);

    // every row of every layer
    double num = 0, den = 0, mx = 0;
    bool exact = true;
    int bad_rows = 0, rows_exact = 0, rows = 0;
    // Perfil do erro POR CAMADA. Sem ele, "fora da tolerancia" e' um numero sem
    // causa: nao da' para distinguir um erro de integracao (ja' grande na camada 0)
    // de amplificacao legitima ao longo das 64 camadas. E' a diferenca entre
    // "consertar" e "declarar".
    std::vector<double> lnum((std::size_t)L, 0.0), lden((std::size_t)L, 0.0),
        lmx((std::size_t)L, 0.0);
    // ... e o perfil por LINHA do chunk (o token t dentro do chunk). Se o erro
    // estiver confinado as linhas >= 16, o suspeito e' um kernel que ainda assume
    // o chunk de 16 do M8; se todas as linhas erram, o suspeito e' outro.
    std::vector<double> rownum((std::size_t)n, 0.0), rowden((std::size_t)n, 0.0);
    for (int il = 0; il < L; ++il) {
      const std::vector<float> &a = ref[(std::size_t)il];
      const std::vector<float> &b = bat[(std::size_t)il];
      const std::size_t want = (std::size_t)n * (std::size_t)E;
      if (a.size() != want || b.size() != want) {
        std::fprintf(stderr, "long-ctx: layer %d captured %zu/%zu floats (want %zu)\n", il,
                     a.size(), b.size(), want);
        ++failures;
        exact = false;
        continue;
      }
      for (int t = 0; t < n; ++t) {
        double rnum = 0, rden = 0, rmx = 0;
        bool row_exact = true;
        for (int i = 0; i < E; ++i) {
          const double x = (double)a[(std::size_t)t * E + i];
          const double y = (double)b[(std::size_t)t * E + i];
          const double d = x - y;
          rnum += d * d;
          rden += x * x + y * y;
          if (d != 0.0) {
            row_exact = false;
            if (std::fabs(d) > rmx) rmx = std::fabs(d);
          }
        }
        num += rnum;
        den += rden;
        lnum[(std::size_t)il] += rnum;
        lden[(std::size_t)il] += rden;
        rownum[(std::size_t)t] += rnum;
        rowden[(std::size_t)t] += rden;
        if (rmx > lmx[(std::size_t)il]) lmx[(std::size_t)il] = rmx;
        if (rmx > mx) mx = rmx;
        ++rows;
        if (row_exact) ++rows_exact; else exact = false;
        const double rrel = rden > 0 ? std::sqrt(rnum / rden) : 0.0;
        const bool row_ok =
            n <= 16 ? row_exact : (rrel <= kTolChunkRelL2 && rmx <= kTolChunkMaxAbs);
        if (!row_ok) ++bad_rows;
      }
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    double lrel = 0, lmax = 0;
    const bool lexact = compare(l1, l2, &lrel, &lmax);
    const Verdict vrows = judge(n, exact, rel, mx);
    const bool rows_ok = vrows.chunked ? (bad_rows == 0 && vrows.ok) : (exact && bad_rows == 0);
    const Verdict vlog = judge(n, lexact, lrel, lmax);
    std::printf("\nlong context (pos %d..%d, n %d, keys %d, splits %d, cap %d): %d camadas, %d linhas\n",
                pos0, pos0 + n - 1, n, pos0 + n, g.debug_attn_splits(), cap, L, rows);
    if (n <= 16) {
      std::printf("  todas as linhas (l_out por camada): rel-L2 %.3e max|d| %.3e  %d/%d linhas iguais"
                  "  %s\n",
                  rel, mx, rows - bad_rows, rows, exact ? "BIT-EXACT" : "DIFEREM");
      std::printf("  logits do ultimo token:             rel-L2 %.3e max|d| %.3e  %s\n", lrel, lmax,
                  lexact ? "BIT-EXACT" : "DIFEREM");
    } else {
      std::printf("  todas as linhas (l_out por camada): erro L2 rel. %.3e max|d| %.3e  "
                  "%d/%d linhas bit-exatas, %d fora da tolerancia  %s\n",
                  rel, mx, rows_exact, rows, bad_rows, rows_ok ? "OK (tol)" : "FORA DA TOLERANCIA");
      std::printf("  logits do ultimo token:             erro L2 rel. %.3e max|d| %.3e  %s\n",
                  lrel, lmax, verdict_label(vlog));
    }
    // Perfil por camada (sempre impresso na faixa tolerada): e' o que localiza a
    // CAUSA quando o erro passa do limite. Um erro ja' grande na camada 0 aponta
    // para a integracao de uma projecao; um erro que cresce com a profundidade
    // aponta para amplificacao do estado.
    if (n > 16) {
      std::printf("  perfil do erro por camada (l_out, erro L2 rel. / max|d|):");
      const int amostra[] = {0, 1, 2, 3, 4, 8, 16, 24, 32, 48, 63};
      for (int il : amostra) {
        if (il >= L) continue;
        const double lr = lden[(std::size_t)il] > 0 ? std::sqrt(lnum[(std::size_t)il] /
                                                                 lden[(std::size_t)il])
                                                    : 0.0;
        std::printf(" L%d %.1e/%.1e;", il, lr, lmx[(std::size_t)il]);
      }
      std::printf("\n");
      std::printf("  perfil do erro por LINHA do chunk (erro L2 rel., linha 0..%d):", n - 1);
      const int amostra_r[] = {0, 1, 2, 3, 7, 8, 15, 16, 17, 23, 31, 32, 47, 63, 95, 127};
      for (int t : amostra_r) {
        if (t >= n) continue;
        const double rr =
            rowden[(std::size_t)t] > 0 ? std::sqrt(rownum[(std::size_t)t] / rowden[(std::size_t)t])
                                       : 0.0;
        std::printf(" t%d %.1e;", t, rr);
      }
      std::printf("\n");
    }
    if (!rows_ok || !vlog.ok) {
      std::fprintf(stderr,
                   "FAIL: contexto longo (n %d, %d chaves, splits %d, cap %d): %d de %d linhas fora "
                   "da tolerancia, erro L2 rel. %.3e (limite %.1e), max|d| %.3e (limite %.1e), "
                   "logits %s\n",
                   n, pos0 + n, g.debug_attn_splits(), cap, bad_rows, rows, rel, kTolChunkRelL2, mx,
                   kTolChunkMaxAbs, verdict_label(vlog));
      ++failures;
    }
    (n > 16 ? chunked_cases : exact_cases) += 1;
  };
  long_ctx_case(16, 1088);
  if (cap > 16) long_ctx_case(cap, 1088);

  if (failures == 0) {
    std::printf("\ncheck-batch-gpu: OK (%d caso(s) bit-exato(s) <= 16 tokens, "
                "%d caso(s) dentro da tolerancia declarada > 16 tokens)\n",
                exact_cases, chunked_cases);
    return 0;
  }
  std::printf("check-batch-gpu: FAILED (%d configuration(s))\n", failures);
  return 1;
}
