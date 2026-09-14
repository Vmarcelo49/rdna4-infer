# Baseline — llama.cpp Vulkan, Qwen3.8-27B-UD-IQ3_S, RX 9070 XT

Data: 2026-09-13. Modelo quente (5 warmups `-p 64 -n 16` antes de medir).

## Setup

- Binary: `/home/marcelo/Projetos/llama.cpp/build/bin/` (`b10902-df03399b8`), backend **Vulkan/RADV** (`GGML_HIP=OFF` neste build — números HIP pendentes, ver fim).
- GPU: AMD Radeon RX 9070 XT (`RADV GFX1201`), 16 GB.
- Modelo: `Qwen3.8-27B-UD-IQ3_S.gguf` (12 GB, `qwen35`, 27.32 B params, 3.4375 bpw).
- Flags comuns: `-ngl 99` (full offload), cache K/V `f16` (default).

## Resultados

### Bench padrão (`llama-bench -p 512 -n 128 -r 3`, ctx default)

| test | t/s |
|---|---|
| pp512 | **1142.60 ± 29.82** |
| tg128 | **38.87 ± 0.26** |

### Contexto 32K (`llama-cli -c 32768 -st`, prompt curto, `-n 128`, 3 reps)

| run | Prompt (prefill) | Generation (decode) |
|---|---|---|
| 1 | 139.7 t/s | 37.3 t/s |
| 2 | 133.2 t/s | 37.1 t/s |
| 3 | 145.6 t/s | 38.0 t/s |

Decode em 32K ≈ **37–38 t/s** — consistente com o tg128 do bench: o decode é bound por banda de VRAM e quase não degrada com o contexto (neste tamanho de prompt).

## Notas

- Prefill degrada com o contexto: ~1140 t/s em ctx curto vs ~140 t/s com KV de 32K alocada.
- O modelo é thinking: o bloco `[Start thinking]` consome parte do `-n 128`, então a resposta final pode truncar — para golden tests usar `-n` maior ou `--no-thinking` (a verificar flag na revisão).
- Armadilha: sem `-st`/`--single-turn`, o `llama-cli` entra em modo interativo e trava esperando stdin (gerou log de 3.7 GB de prompts `>`). Sempre rodar bench não-interativo com redirect.
- RADV avisa que não é implementação Vulkan conformante ("testing use only") — números valem como referência, não como verdade absoluta.
- Próximo: build HIP (`build-hip/` configurado com `-DGGML_HIP=ON -DGPU_TARGETS=gfx1201`, parado em `hip/hip_fp16.h` não encontrado — falta include do ROCm no configure) para repetir estas medidas no backend do projeto.

## Re-medição (2026-09-13, placa ociosa, mesmo binário e mesmas flags)

`llama-bench -m Qwen3.8-27B-UD-IQ3_S.gguf -ngl 99 -p 64 -n 64 -r 3`:

| test | agora | na tabela do M5 | leitura |
|---|---|---|---|
| pp64 | **575,00 ± 64,79** | 440,49 ± 76,70 | **+30 % com o mesmo comando**: `-p 64` é curto demais para medir prefill — o `±` de 65-77 t/s é 11-17 % do valor. Não citar pp64 com três dígitos |
| tg64 | **40,03 ± 0,02** | 39,73 ± 0,45 | estável (0,8 %), o decode é o número confiável desta tabela |

Referência estável de prefill: **pp512 = 1142,60 ± 29,82 t/s** (§1.1), que é 2× o pp64. O
`tg` não depende do tamanho do prompt, o `pp` depende muito: qualquer comparação de prefill
tem de dizer o comprimento do prompt dos dois lados.
