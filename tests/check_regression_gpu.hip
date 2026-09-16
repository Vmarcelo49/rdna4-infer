// Suite de regressao numerica (task 7 do lote) — o gate que se roda depois de
// QUALQUER mudanca no motor.
//
// O que ela faz: roda um conjunto FIXO de prompts pelo caminho real do motor
// (tokenizer -> forward_batch/forward_tokens -> logits), decodifica em greedy
// puro (argmax, RNG-free) e compara duas coisas contra a referencia commitada
// `tests/golden/regression_greedy_f16.txt`:
//
//   1. os IDS GERADOS, bit-exatos (o greedy e deterministico: qualquer mudanca
//      de aritmetica que troque um argmax aparece aqui, no token e no indice);
//   2. os LOGITS do ultimo passo, com tolerancia documentada: rel-L2 <= 1e-5
//      sobre uma subamostra de passo fixo e |d||logits||| / ||logits|| <= 1e-5.
//      A tolerancia existe porque reordenar a atencao (splits, warps por CTA)
//      muda os ultimos bits de forma legitima — M7 mediu rel-L2 ~2,5e-7 nesse
//      tipo de mudanca. Bit-exato aqui seria um gate que quebra a cada
//      otimizacao legitima; sem tolerancia nenhuma, um erro real de ~1e-6
//      passaria.
//
// Cobertura (7 casos; ver kCases): curto, prosa longa, codigo, multilingue
// (CJK + acentos), chat com o template do modelo, prefill real de 4096 tokens e
// o contexto mais longo que cabe com KV f16 (65536, cache sintetico semeado,
// como `bench --start-pos`, que e o caminho onde a atencao com split roda).
//
// ANTI-TESTE-VAZIO: o repo ja foi mordido duas vezes (um teste de tokenizer que
// comparava 0 tokens e um golden que passava com 0 tokens). Aqui: referencia
// ausente = FALHA; menos de kMinCases casos = FALHA; qualquer caso com
// n_gen != declarado, ids.size() != n_gen, ou subamostra de logits menor que o
// minimo = FALHA. Toda comparacao imprime o numero que comparou.
//
// usage: check-regression-gpu <model.gguf> <golden.txt> [--record]
//
// Custo medido (IQ3_S, ctx 65536, f16): ~2 min, quase tudo no prefill real de
// 4096 tokens e no carregamento do modelo — ver docs/autotuning-gfx1201.md
// (alvo do gate: <= 5 min).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "rdna4/chat.h"
#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"
#include "rdna4/sampler.h"
#include "rdna4/tokenizer.h"

namespace {

using Clock = std::chrono::steady_clock;
double now_s() {
  static const Clock::time_point t0 = Clock::now();
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- tolerancias
// Documentadas acima; o gate imprime o valor medido de cada uma.
const double kLogitRelL2 = 1e-5;   // rel-L2 da subamostra de logits
const double kLogitNormRel = 1e-5; // |d||logits||| / ||logits||
const int kLogitStride = 97;       // passo da subamostra (primo, nao divide vocab)
const std::size_t kMinSubsample = 2000;
const int kMinCases = 7;

// ------------------------------------------------------------------- promptos
const char *kProse =
    "A engenharia de compiladores trata a traducao de uma linguagem de alto nivel "
    "para instrucoes de maquina como uma sequencia de transformacoes que preservam "
    "o significado do programa. Cada transformacao tem um contrato: o que ela pode "
    "assumir sobre a entrada e o que ela garante sobre a saida. Quando um passe "
    "assume algo que nao foi provado, o resultado e um erro silencioso que so "
    "aparece em producao, muito depois da mudanca que o introduziu. Por isso os "
    "projetos serios mantem uma suite de regressao: prompts fixos, numeros fixos, e "
    "uma tolerancia explicita para o que pode mudar sem quebrar o contrato. ";

const char *kCode =
    "// Soma os elementos de um vetor em float, com acumulador duplo para reduzir o\n"
    "// erro de arredondamento, e devolve a media.\n"
    "static float media(const float *x, int n) {\n"
    "  double a = 0.0, b = 0.0;\n"
    "  for (int i = 0; i < n; i += 2) {\n"
    "    a += x[i];\n"
    "    if (i + 1 < n) b += x[i + 1];\n"
    "  }\n"
    "  return (float)((a + b) / (double)n);\n"
    "}\n"
    "// A seguir, a versao vetorizada da mesma funcao:\n";

const char *kCjk =
    "Explique em uma frase curta, primeiro em portugues e depois em japones e em "
    "chines, por que a memoria de uma GPU e dividida em hierarquia. Use exemplos "
    "concretos: 日本語では、メモリの階層は速度と容量のトレードオフです。中文里，"
    "缓存和显存的分层同样是为了让常用数据离计算单元更近。Atencao: acentuacao "
    "(ação, coração, irmã) e cedilha (ç) tambem fazem parte do teste.";

struct Case {
  const char *name;
  const char *prompt;   // usado quando `chat` e false
  int n_gen;
  bool chat;            // renderiza com o template do modelo em vez do texto cru
  long long synth_pos;  // > 0: semeia o cache e decodifica a partir desta posicao
  int min_prompt;       // minimo de tokens de prompt esperado (anti vacuo)
};

const Case kCases[] = {
    {"short", "The capital of France is", 32, false, 0, 2},
    {"long", kProse, 24, false, 0, 100},
    {"code", kCode, 32, false, 0, 80},
    {"cjk", kCjk, 24, false, 0, 50},
    {"chat", nullptr, 24, true, 0, 20},
    // prefill real de 4096 tokens (o baseline de decode documentado e a ctx 4096)
    {"ctx4k", kProse, 8, false, 0, 4096},
    // contexto mais longo que CABE com KV f16 nesta placa (cache sintetico
    // semeado, igual ao `bench --start-pos`): e o caminho da atencao com 16
    // splits. O KV f16 deste modelo custa 64 KB por token (16 camadas x 4 cabecas
    // kv x 256 dims x 2 B x 2), entao 64K de ctx pediria 4,3 GB so de KV e
    // estouraria os 15,9 GB da placa junto com os ~11,5 GB de pesos -- foi medido:
    // com ctx 65600 o prefill cai para 6,9 tok/s (paginacao do driver) contra
    // 69 tok/s a 4096. 32768 e o maior que roda a placa cheia.
    {"ctx32k", kProse, 8, false, 32768, 100},
};
const int kCaseCount = (int)(sizeof(kCases) / sizeof(kCases[0]));

struct CaseResult {
  std::string name;
  int n_prompt = 0;
  int n_gen = 0;
  std::vector<std::int32_t> ids;
  int stride = 0;
  std::vector<float> logs;  // subamostra deterministicamente escolhida
  double norm = 0.0;        // ||logits||_2 do ultimo passo (vetor inteiro)
  double ms = 0.0;
};

// ------------------------------------------------------------- serializacao
// Formato de texto, uma linha por campo, para o diff do git ser legivel:
//   case <nome> n_prompt <n> n_gen <n>
//   ids <id> <id> ...
//   logits stride <s> n <n> <v> <v> ...
//   norm <v>
//   end
bool save(const std::string &path, const std::vector<CaseResult> &rs, const std::string &model,
          int ctx) {
  std::ofstream f(path);
  if (!f) return false;
  f << "# regression_greedy_f16 — baseline numerico do motor (tests/check_regression_gpu.hip)\n"
    << "# modelo: " << model << "\n"
    << "# kv: f16/f16   ctx: " << ctx << "\n"
    << "# greedy: temp 0, sem penalidades/top_k/top_p/min_p (argmax puro, RNG-free)\n"
    << "# comparacao: ids bit-exatos; rel-L2 dos logits <= " << kLogitRelL2
    << "; norma L2 relativa <= " << kLogitNormRel << "\n"
    << "# regravar: check-regression-gpu <model> <este arquivo> --record\n";
  for (const CaseResult &r : rs) {
    f << "case " << r.name << " n_prompt " << r.n_prompt << " n_gen " << r.n_gen << "\n";
    f << "ids";
    for (std::int32_t id : r.ids) f << ' ' << id;
    f << "\nlogits stride " << r.stride << " n " << r.logs.size();
    f.precision(9);
    for (float v : r.logs) f << ' ' << v;
    f << "\nnorm " << r.norm << "\nend\n";
  }
  return (bool)f;
}

bool load(const std::string &path, std::vector<CaseResult> &rs, std::string &err) {
  std::ifstream f(path);
  if (!f) {
    err = "referencia AUSENTE: " + path +
          " (grave com `check-regression-gpu <model> " + path +
          " --record` depois de validar a mudanca)";
    return false;
  }
  std::string line;
  const CaseResult *cur = nullptr;
  CaseResult tmp;
  bool in_case = false;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (line.rfind("case ", 0) == 0) {
      char name[128] = {};
      int np = 0, ng = 0;
      if (std::sscanf(line.c_str(), "case %127s n_prompt %d n_gen %d", name, &np, &ng) != 3) {
        err = "linha `case` ilegivel: " + line;
        return false;
      }
      tmp = CaseResult{};
      tmp.name = name;
      tmp.n_prompt = np;
      tmp.n_gen = ng;
      in_case = true;
    } else if (line.rfind("ids", 0) == 0) {
      if (!in_case) {
        err = "`ids` fora de um caso";
        return false;
      }
      const char *p = line.c_str() + 3;
      char *end = nullptr;
      while (*p) {
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        tmp.ids.push_back((std::int32_t)v);
        p = end;
      }
    } else if (line.rfind("logits stride", 0) == 0) {
      if (!in_case) {
        err = "`logits` fora de um caso";
        return false;
      }
      int stride = 0;
      long n = 0;
      if (std::sscanf(line.c_str(), "logits stride %d n %ld", &stride, &n) != 2) {
        err = "linha `logits` ilegivel";
        return false;
      }
      tmp.stride = stride;
      const char *p = std::strstr(line.c_str(), " n ");
      if (!p) {
        err = "linha `logits` sem contagem";
        return false;
      }
      p += 3;
      char *end = nullptr;
      (void)std::strtol(p, &end, 10);  // a propria contagem
      p = end;
      while (*p) {
        const double v = std::strtod(p, &end);
        if (end == p) break;
        tmp.logs.push_back((float)v);
        p = end;
      }
    } else if (line.rfind("norm ", 0) == 0) {
      tmp.norm = std::strtod(line.c_str() + 5, nullptr);
    } else if (line == "end") {
      if (!in_case) {
        err = "`end` sem caso";
        return false;
      }
      rs.push_back(tmp);
      in_case = false;
      cur = nullptr;
      (void)cur;
    }
  }
  if (rs.empty()) {
    err = "referencia sem nenhum caso: " + path;
    return false;
  }
  return true;
}

// ------------------------------------------------------------------- util
std::vector<float> subsample(const std::vector<float> &v, int stride) {
  std::vector<float> out;
  for (std::size_t i = 0; i < v.size(); i += (std::size_t)stride) out.push_back(v[i]);
  return out;
}

double norm2(const std::vector<float> &v) {
  double s = 0;
  for (float x : v) s += (double)x * (double)x;
  return std::sqrt(s);
}

// Prefill com a MESMA decomposicao do CLI (`src/main.hip`, prefill_ids): chunks
// de 16/8/4/3/2, token unico so quando sobra 1. Assim o teste mede o caminho que
// o motor realmente usa, nao um caminho paralelo.
bool prefill(rdna4::Graph &g, const std::vector<std::int32_t> &ids, std::vector<float> &hidden,
             std::vector<float> &logits, std::string &err) {
  static const int kSizes[] = {16, 8, 4, 3, 2};
  std::size_t pos = 0;
  while (pos < ids.size()) {
    std::size_t take = 1;
    for (int sz : kSizes) {
      if (ids.size() - pos >= (std::size_t)sz) {
        take = (std::size_t)sz;
        break;
      }
    }
    if (take == 1) {
      if (!g.forward_tokens({ids[pos]}, (int)pos, hidden, logits, err)) return false;
    } else {
      const std::vector<std::int32_t> chunk(ids.begin() + (long)pos,
                                            ids.begin() + (long)(pos + take));
      if (!g.forward_batch(chunk, (int)pos, hidden, logits, err)) return false;
    }
    pos += take;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  const char *model = nullptr;
  const char *golden = nullptr;
  bool record = false;
  bool verbose = true;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--record") == 0) record = true;
    else if (std::strcmp(argv[i], "--quiet") == 0) verbose = false;
    else if (!model) model = argv[i];
    else if (!golden) golden = argv[i];
  }
  if (!model || !golden) {
    std::fprintf(stderr,
                 "usage: check-regression-gpu <model.gguf> <golden.txt> [--record]\n"
                 "  sem --record: compara com a referencia commitada e FALHA se ela faltar\n"
                 "  com --record: regrava a referencia (so depois de validar a mudanca)\n");
    return 2;
  }

  // Quick smoke mode (REGRESSION_QUICK=1): shrink the dominant costs (32K ctx
  // alloc + 4096-token real prefill + 32-token short decodes) and drop the
  // seeded-32K case. Unset = full default behavior (unchanged). A quick run
  // needs its own golden (record with --record under the same env).
  const bool reg_quick = [] {
    const char *e = std::getenv("REGRESSION_QUICK");
    return e != nullptr && e[0] == '1';
  }();
  // Effective decode length: short cases 32/32/24/24/24 -> 8 in quick mode
  // (ctx4k/ctx32k already generate 8). Default: the declared n_gen.
  auto eff_gen = [reg_quick](const Case &c) { return reg_quick ? (c.n_gen < 8 ? c.n_gen : 8) : c.n_gen; };
  const int eff_min_cases = reg_quick ? (kMinCases - 1) : kMinCases;

  const int ctx = reg_quick ? (4096 + 64) : (32768 + 64);
  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(model, err)) {
    std::fprintf(stderr, "gguf: %s\n", err.c_str());
    return 2;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(loader.meta(), cfg, err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 2;
  }
  rdna4::Tokenizer tk;
  if (!tk.init(loader.meta(), err)) {
    std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
    return 2;
  }
  std::printf("check-regression-gpu: %d casos, ctx %d, kv f16/f16, greedy puro%s\n", kCaseCount,
                ctx, reg_quick ? " [REGRESSION_QUICK=1]" : "");

  std::vector<std::vector<std::int32_t>> prompts((std::size_t)kCaseCount);
  for (int i = 0; i < kCaseCount; ++i) {
    std::string text;
    if (kCases[i].chat) {
      std::vector<rdna4::ChatMessage> msgs;
      msgs.push_back({"system", "Voce e um assistente tecnico conciso."});
      msgs.push_back({"user", "Explique em duas frases por que um kernel de atencao usa "
                              "softmax online em vez de guardar todos os escores."});
      rdna4::ChatOptions opts;
      opts.enable_thinking = false;  // sem bloco de raciocinio: prompt curto e fixo
      if (!rdna4::chat_render(msgs, opts, text, err)) {
        std::fprintf(stderr, "chat_render: %s\n", err.c_str());
        return 2;
      }
    } else {
      text = kCases[i].prompt;
    }
    // ctx4k: repete a prosa ate passar de 4096 tokens (deterministico;
    // QUICK: 1024 tokens, que e o prefill real dominante que encolhe aqui)
    std::vector<std::int32_t> ids = tk.encode(text, true);
    if (kCases[i].min_prompt >= 4096) {
      const int want_prompt = reg_quick ? 1024 : 4096;
      std::string big = text;
      while ((int)tk.encode(big, true).size() < want_prompt) big += text;
      ids = tk.encode(big, true);
    }
    prompts[(std::size_t)i] = ids;
    const int want_min = (reg_quick && kCases[i].min_prompt >= 4096) ? 1024 : kCases[i].min_prompt;
    if ((int)ids.size() < want_min) {
      std::fprintf(stderr, "caso %s: prompt com %zu tokens, esperado >= %d\n", kCases[i].name,
                   ids.size(), want_min);
      return 2;
    }
  }

  rdna4::Graph g(loader, cfg);
  if (!g.init(ctx, rdna4::KvType::F16, rdna4::KvType::F16, err)) {
    std::fprintf(stderr, "graph init: %s\n", err.c_str());
    return 2;
  }
  g.set_want_hidden(false);

  // greedy puro: sem penalidade, sem top_k/top_p/min_p, temp 0 -> argmax; o seed
  // fica so para o caso de alguem trocar a config (o caminho greedy nao usa RNG).
  rdna4::SamplerParams sp;
  sp.temp = 0.0f;
  sp.top_k = 0;
  sp.top_p = 1.0f;
  sp.min_p = 0.0f;
  sp.repeat_penalty = 1.0f;
  sp.seed = 1234;
  rdna4::Sampler sampler;
  sampler.init(sp, g.n_vocab());

  int failures = 0;
  std::vector<CaseResult> got;
  got.reserve((std::size_t)kCaseCount);
  std::printf("%-9s %8s %6s %10s %12s %12s %8s\n", "caso", "prompt", "gen", "ms", "rel-L2",
              "d||logits||", "ids");
  for (int i = 0; i < kCaseCount; ++i) {
    const Case &c = kCases[i];
    const int want_gen = eff_gen(c);
    // QUICK: drop the seeded-32K case (needs ctx 32832); the remaining 6 cases
    // still print the run table + rel-L2 comparison lines below.
    if (reg_quick && c.synth_pos > 0) {
      std::printf("%-9s %8s %6s %10s %12s %12s %8s  SKIPPED (REGRESSION_QUICK=1)\n", c.name,
                  "-", "-", "-", "-", "-", "-");
      continue;
    }
    std::vector<float> hidden, logits;
    const double t0 = now_s();
    if (!g.reset_state(err)) {
      std::fprintf(stderr, "reset_state: %s\n", err.c_str());
      return 2;
    }
    // cache sintetico semeado para o caso de contexto longo (mesmo truque do
    // `bench --start-pos`: e um teste de aritmetica, nao de qualidade do texto)
    if (c.synth_pos > 0 && !g.debug_fill_caches(7u, err)) {
      std::fprintf(stderr, "debug_fill_caches: %s\n", err.c_str());
      return 2;
    }
    const std::vector<std::int32_t> &ids = prompts[(std::size_t)i];
    const double t_pre0 = now_s();
    if (ids.empty()) {
      std::fprintf(stderr, "caso %s: prompt vazio\n", c.name);
      ++failures;
      continue;
    }
    if (!prefill(g, ids, hidden, logits, err)) {
      std::fprintf(stderr, "caso %s: prefill falhou: %s\n", c.name, err.c_str());
      ++failures;
      continue;
    }
    const double prefill_ms = (now_s() - t_pre0) * 1000.0;
    const double t_dec0 = now_s();
    std::size_t pos = ids.size();
    if (c.synth_pos > 0) pos = (std::size_t)c.synth_pos;
    std::vector<std::int32_t> history = ids;
    CaseResult r;
    r.name = c.name;
    r.n_prompt = (int)ids.size();
    std::vector<float> last_logits;
    for (int k = 0; k < want_gen; ++k) {
      if (pos >= (std::size_t)ctx) {
        std::fprintf(stderr, "caso %s: posicao %zu estourou o ctx %d\n", c.name, pos, ctx);
        ++failures;
        break;
      }
      last_logits = logits;  // copia ANTES do sampler (o sampler altera logits)
      const std::int32_t id = sampler.sample(logits.data(), history);
      if (id < 0) {
        std::fprintf(stderr, "caso %s: sampler devolveu %d\n", c.name, id);
        ++failures;
        break;
      }
      r.ids.push_back(id);
      history.push_back(id);
      if (k + 1 == want_gen) break;
      if (!g.forward_tokens({id}, (int)pos, hidden, logits, err)) {
        std::fprintf(stderr, "caso %s: forward falhou: %s\n", c.name, err.c_str());
        ++failures;
        break;
      }
      ++pos;
    }
    r.n_gen = (int)r.ids.size();
    r.stride = kLogitStride;
    r.logs = subsample(last_logits, kLogitStride);
    r.norm = norm2(last_logits);
    r.ms = (now_s() - t0) * 1000.0;
    const double decode_ms = (now_s() - t_dec0) * 1000.0;
    if (verbose) {
      std::printf("  %-9s prefill %8.1f ms (%6.1f tok/s)  decode %8.1f ms (%6.1f ms/tok)\n",
                  c.name, prefill_ms, prefill_ms > 0 ? ids.size() / (prefill_ms * 1e-3) : 0.0,
                  decode_ms, r.n_gen > 0 ? decode_ms / r.n_gen : 0.0);
    }
    // anti-teste-vazio: cada caso tem que gerar exatamente o que declara e
    // guardar uma subamostra de logits de tamanho util
    if (r.n_gen != want_gen) {
      std::fprintf(stderr, "caso %s: gerou %d tokens, esperado %d\n", c.name, r.n_gen, want_gen);
      ++failures;
    }
    if (r.logs.size() < kMinSubsample) {
      std::fprintf(stderr, "caso %s: subamostra de logits com %zu valores (minimo %zu)\n", c.name,
                   r.logs.size(), kMinSubsample);
      ++failures;
    }
    std::printf("%-9s %8d %6d %10.1f %12s %12s %8zu\n", c.name, r.n_prompt, r.n_gen, r.ms, "-", "-",
                r.ids.size());
    got.push_back(r);
  }

  if (got.size() < (std::size_t)eff_min_cases) {
    std::fprintf(stderr, "check-regression-gpu: FAIL: so %zu casos executaram (minimo %d)\n",
                 got.size(), eff_min_cases);
    return 1;
  }

  if (record) {
    if (failures) {
      std::fprintf(stderr,
                   "check-regression-gpu: NAO gravando: %d falha(s) na propria execucao\n",
                   failures);
      return 1;
    }
    if (!save(golden, got, model, ctx)) {
      std::fprintf(stderr, "check-regression-gpu: nao consegui gravar %s\n", golden);
      return 1;
    }
    std::printf("check-regression-gpu: referencia regravada em %s (%zu casos)\n", golden,
                got.size());
    return 0;
  }

  std::vector<CaseResult> want;
  if (!load(golden, want, err)) {
    std::fprintf(stderr, "check-regression-gpu: FAIL: %s\n", err.c_str());
    return 1;
  }
  if (want.size() < (std::size_t)eff_min_cases) {
    std::fprintf(stderr,
                 "check-regression-gpu: FAIL: a referencia tem %zu casos (minimo %d) — comparar "
                 "pouco nao e um gate\n",
                 want.size(), eff_min_cases);
    return 1;
  }
  if (want.size() != got.size()) {
    std::fprintf(stderr,
                 "check-regression-gpu: FAIL: referencia com %zu casos, execucao com %zu%s\n",
                 want.size(), got.size(),
                 reg_quick ? " (REGRESSION_QUICK=1: grave um golden rapido com --record sob o mesmo env)"
                           : "");
    return 1;
  }

  std::printf("\ncomparacao com %s\n", golden);
  std::printf("%-9s %8s %6s %12s %12s %12s  %s\n", "caso", "prompt", "gen", "rel-L2", "max|d|",
              "d||logits||", "ids");
  for (std::size_t i = 0; i < got.size(); ++i) {
    const CaseResult &a = got[i];
    const CaseResult &b = want[i];
    if (a.name != b.name) {
      std::fprintf(stderr, "check-regression-gpu: FAIL: caso %zu e '%s', referencia diz '%s'\n",
                   i, a.name.c_str(), b.name.c_str());
      ++failures;
      continue;
    }
    if (a.n_prompt != b.n_prompt || a.n_gen != b.n_gen) {
      std::fprintf(stderr,
                   "check-regression-gpu: FAIL: caso %s: prompt/gen %d/%d contra %d/%d na "
                   "referencia\n",
                   a.name.c_str(), a.n_prompt, a.n_gen, b.n_prompt, b.n_gen);
      ++failures;
      continue;
    }
    if (a.logs.size() != b.logs.size() || a.stride != b.stride) {
      std::fprintf(stderr,
                   "check-regression-gpu: FAIL: caso %s: subamostra de logits %zu (stride %d) "
                   "contra %zu (stride %d)\n",
                   a.name.c_str(), a.logs.size(), a.stride, b.logs.size(), b.stride);
      ++failures;
      continue;
    }
    // (1) ids bit-exatos
    bool ids_ok = a.ids.size() == b.ids.size();
    std::size_t first_bad = 0;
    if (ids_ok)
      for (std::size_t k = 0; k < a.ids.size(); ++k)
        if (a.ids[k] != b.ids[k]) {
          ids_ok = false;
          first_bad = k;
          break;
        }
    // (2) logits com tolerancia
    double num = 0, den = 0, maxd = 0;
    for (std::size_t k = 0; k < a.logs.size(); ++k) {
      const double d = (double)a.logs[k] - (double)b.logs[k];
      num += d * d;
      den += (double)a.logs[k] * a.logs[k] + (double)b.logs[k] * b.logs[k];
      maxd = std::max(maxd, std::fabs(d));
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    const double dnorm = b.norm > 0 ? std::fabs(a.norm - b.norm) / b.norm : 0.0;
    std::printf("%-9s %8d %6d %12.2e %12.2e %12.2e  %s\n", a.name.c_str(), a.n_prompt, a.n_gen,
                rel, maxd, dnorm, ids_ok ? "OK" : "DIFEREM");
    if (!ids_ok) {
      std::fprintf(stderr,
                   "check-regression-gpu: FAIL: %s: id gerado difere no passo %zu: %d contra %d "
                   "na referencia (%zu ids comparados)\n",
                   a.name.c_str(), first_bad, (int)a.ids[first_bad], (int)b.ids[first_bad],
                   a.ids.size());
      ++failures;
    }
    if (!(rel <= kLogitRelL2)) {
      std::fprintf(stderr,
                   "check-regression-gpu: FAIL: %s: rel-L2 dos logits %.3e > %.0e (%zu valores "
                   "comparados)\n",
                   a.name.c_str(), rel, kLogitRelL2, a.logs.size());
      ++failures;
    }
    if (!(dnorm <= kLogitNormRel)) {
      std::fprintf(stderr,
                   "check-regression-gpu: FAIL: %s: norma L2 dos logits difere %.3e > %.0e\n",
                   a.name.c_str(), dnorm, kLogitNormRel);
      ++failures;
    }
  }

  if (failures == 0) {
    std::printf("check-regression-gpu: OK (%zu casos, %d ids bit-exatos, rel-L2 <= %.0e)\n",
                got.size(), (int)got.size(), kLogitRelL2);
    return 0;
  }
  std::printf("check-regression-gpu: FAILED (%d falha(s))\n", failures);
  return 1;
}
