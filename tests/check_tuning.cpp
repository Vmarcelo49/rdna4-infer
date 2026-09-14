// Gate da tabela de tuning (task 6 do lote de autotuning).
//
// O que ele protege: a configuracao que o motor USA tem que ser a que foi MEDIDA.
// include/rdna4/tuning.h e a unica fonte dos numeros (as tabelas de compilacao de
// matvec.cuh/attn.cuh/graph.cuh leem de la), e este teste imprime a tabela e a
// compara com a referencia commitada tests/golden/ml_tuning.txt. Um refactor
// generico que troque um `rows`, um `ILP`, um `UNROLL`, o numero de warps da
// atencao ou a politica de splits quebra este gate em vez de mudar o desempenho
// em silencio -- que e o que "nao pode apodrecer sozinho" significa aqui.
//
// CPU PURA: sem GPU, sem modelo, sem HIP (tuning.h nao inclui nada de HIP de
// proposito). Roda em milissegundos, entao pode entrar em qualquer gate.
//
// usage: check-tuning [tests/golden/ml_tuning.txt]
//        check-tuning --record tests/golden/ml_tuning.txt   (so depois de medir;
//                     ver docs/autotuning-gfx1201.md, secao "Como re-medir")
//
// Anti-teste-vazio (o repo ja foi mordido duas vezes por isso): o teste exige um
// numero minimo de linhas comparadas E a presenca de todas as chaves obrigatorias;
// referencia faltando, truncada ou com chave a menos e FALHA, nunca PASS.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rdna4/tuning.h"

namespace {

const char *kGoldenDefault = "tests/golden/ml_tuning.txt";
// 14 tipos de matvec + 4 da atencao + 3 do batch + 1 do total = 22 linhas.
const int kMinLines = 21;

// Chaves que TEM que existir na referencia. Uma referencia que perca qualquer uma
// delas e considerada invalida (nao "passa com menos linhas").
const char *kMandatory[] = {
    "matvec.types",     "matvec.q8_0",      "matvec.q2_K",     "matvec.q3_K",
    "matvec.q4_K",      "matvec.q5_K",      "matvec.q6_K",     "matvec.iq2_xxs",
    "matvec.iq2_xs",    "matvec.iq3_xxs",   "matvec.iq1_s",    "matvec.iq4_nl",
    "matvec.iq3_s",     "matvec.iq2_s",     "matvec.iq4_xs",   "attn.warps_per_block",
    "attn.split_wpb_limit", "attn.split_wpb_wide", "attn.split_min", "attn.max_splits",
    "batch.ns",
    "batch.cap",
};

std::string fmt(const char *f, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return std::string(buf);
}

// A tabela, na ordem em que e impressa e comparada. Formato `chave = valor`, uma
// linha por parametro, para um diff de git/olho humano mostrar exatamente o que
// mudou.
std::vector<std::string> report_lines() {
  using namespace rdna4;
  std::vector<std::string> out;
  out.push_back(fmt("matvec.types = %d", tuned::kMtTypes));
  for (int i = 0; i < tuned::kMtTypes; ++i) {
    out.push_back(fmt("matvec.%s = rows %d wpr %d ilp %d unroll %d", tuned::kMtNames[i],
                      tuned::kMtRows[i], tuned::kMtWpr[i], tuned::kMtIlp[i],
                      tuned::kMtUnroll[i]));
  }
  out.push_back(fmt("attn.warps_per_block = %d", tuned::kAttnWarpsPerBlock));
  out.push_back(fmt("attn.split_wpb_limit = %d", tuned::kAttnSplitWpbLimit));
  out.push_back(fmt("attn.split_wpb_wide = %d", tuned::kAttnSplitWpbWide));
  out.push_back(fmt("attn.split_min = %d", tuned::kAttnSplitMin));
  out.push_back(fmt("attn.max_splits = %d", tuned::kAttnMaxSplits));
  std::string ns;
  for (int i = 0; i < tuned::kBatchCount; ++i) {
    if (i) ns += ",";
    ns += std::to_string(tuned::kBatchNs[i]);
  }
  out.push_back("batch.ns = " + ns);
  out.push_back(fmt("batch.cap = %d", tuned::kBatchCap));
  return out;
}

// Le a referencia: comentarios (`#`) e linhas vazias sao ignorados, o resto e
// comparado literalmente (inclusive a ordem).
bool read_reference(const char *path, std::vector<std::string> &lines, std::string &err) {
  std::FILE *f = std::fopen(path, "r");
  if (!f) {
    err = fmt("referencia AUSENTE: %s (rode `check-tuning --record %s` depois de medir, "
              "docs/autotuning-gfx1201.md)",
              path, path);
    return false;
  }
  char buf[512];
  while (std::fgets(buf, sizeof buf, f)) {
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (s.empty() || s[0] == '#') continue;
    lines.push_back(s);
  }
  std::fclose(f);
  return true;
}

bool write_reference(const char *path, const std::vector<std::string> &lines, std::string &err) {
  std::FILE *f = std::fopen(path, "w");
  if (!f) {
    err = fmt("nao consegui escrever %s", path);
    return false;
  }
  std::fprintf(f, "# ml_tuning — configuracao MEDIDA e EMBARCADA deste motor em gfx1201\n");
  std::fprintf(f, "# (AMD RX 9070 XT, ROCm 7.2). Gerado por `check-tuning --record %s`.\n", path);
  std::fprintf(f, "# Comparado por `check-tuning` (gate de pre-merge); numeros medidos,\n");
  std::fprintf(f, "# candidatos rejeitados e piso de ruido: docs/autotuning-gfx1201.md.\n");
  std::fprintf(f, "# Fonte dos valores: include/rdna4/tuning.h (unica).\n");
  for (const std::string &l : lines) std::fprintf(f, "%s\n", l.c_str());
  std::fclose(f);
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  const char *path = kGoldenDefault;
  bool record = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--record") == 0) {
      record = true;
      if (i + 1 < argc) path = argv[++i];
    } else {
      path = argv[i];
    }
  }

  const std::vector<std::string> now = report_lines();
  std::printf("check-tuning: configuracao embarcada (fonte: include/rdna4/tuning.h)\n");
  for (const std::string &l : now) std::printf("  %s\n", l.c_str());
  std::printf("  (%zu linhas; referencia: %s)\n", now.size(), path);

  if (record) {
    std::string err;
    if (!write_reference(path, now, err)) {
      std::fprintf(stderr, "check-tuning: FAIL: %s\n", err.c_str());
      return 1;
    }
    std::printf("check-tuning: referencia regravada em %s (%zu linhas)\n", path, now.size());
    return 0;
  }

  std::vector<std::string> ref;
  std::string err;
  if (!read_reference(path, ref, err)) {
    std::fprintf(stderr, "check-tuning: FAIL: %s\n", err.c_str());
    return 1;
  }
  // anti-teste-vazio (1): numero minimo de linhas comparadas
  if ((int)ref.size() < kMinLines) {
    std::fprintf(stderr,
                 "check-tuning: FAIL: a referencia %s tem so %zu linhas comparaveis (minimo %d) — "
                 "uma comparacao vazia nao e um gate\n",
                 path, ref.size(), kMinLines);
    return 1;
  }
  // anti-teste-vazio (2): toda chave obrigatoria presente
  int missing = 0;
  for (const char *key : kMandatory) {
    bool found = false;
    const std::string prefix = std::string(key) + " =";
    for (const std::string &l : ref)
      if (l.rfind(prefix, 0) == 0) {
        found = true;
        break;
      }
    if (!found) {
      std::fprintf(stderr, "check-tuning: FAIL: chave obrigatoria ausente na referencia: %s\n", key);
      ++missing;
    }
  }
  if (missing) return 1;
  if (ref.size() != now.size()) {
    std::fprintf(stderr,
                 "check-tuning: FAIL: a referencia tem %zu linhas e a tabela embarcada tem %zu — "
                 "a referencia foi gravada de outra configuracao\n",
                 ref.size(), now.size());
    return 1;
  }
  int diffs = 0;
  for (std::size_t i = 0; i < now.size(); ++i) {
    if (now[i] == ref[i]) continue;
    std::fprintf(stderr, "check-tuning: FAIL linha %zu:\n    embarcado: %s\n    referencia: %s\n",
                 i + 1, now[i].c_str(), ref[i].c_str());
    if (++diffs >= 10) {
      std::fprintf(stderr, "    (parando depois de 10 diferencas)\n");
      break;
    }
  }
  if (diffs) {
    std::fprintf(stderr,
                 "check-tuning: FAIL: %d linha(s) diferem da configuracao medida. Se a mudanca foi "
                 "medida de verdade, atualize include/rdna4/tuning.h E a referencia (%s) no mesmo "
                 "commit, com os numeros em docs/autotuning-gfx1201.md.\n",
                 diffs, path);
    return 1;
  }
  std::printf("check-tuning: OK (%zu linhas comparadas, %zu chaves obrigatorias presentes) — a "
              "configuracao embarcada e a medida\n",
              now.size(), sizeof(kMandatory) / sizeof(kMandatory[0]));
  return 0;
}
