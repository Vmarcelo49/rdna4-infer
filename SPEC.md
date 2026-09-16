# SPEC — rdna4-infer

Engine próprio de inferência para placas RDNA4, com um único alvo: **Qwen3.8 27B denso em GGUF na RX 9070 XT (gfx1201, 16 GB)**.

Decisões travadas: engine próprio em **C++ + HIP** (reusando módulos do llama.cpp sob MIT, sem fork integral) · interface **CLI** · arquivos de teste **UD-IQ3_S (12 GB) e UD-IQ4_XS (14 GB)** com seu union misto de tipos · **sem meta numérica** de performance na v1.

Pesquisas de base em `docs/`: `rdna4-gfx1201-referencias-amd.md`, `referencias-upstream-gfx1201-qwen35.md`, `kernels-ia-gfx1201.md`, `gguf-qwen-quantizacao-llamacpp.md`.

## 1. VAI TER (v1)

### 1.1 CLI
- Binário `rdna4-infer` com subcomando `run`: `-m modelo.gguf -p "prompt" [-n max_tokens] [--temp] [--top-k] [--top-p] [--min-p] [--repeat-penalty] [--seed] [--ctx-size] [--cache-type-k f16|q8_0|q4_0] [--cache-type-v f16|q8_0|q4_0]`.
- Streaming de tokens no stdout; código de saída 0 em sucesso, != 0 com mensagem em stderr em qualquer falha (modelo incompatível, VRAM insuficiente, GPU errada).
`Qwen3.8-27B-UD-IQ3_S.gguf` (12 GB) e `Qwen3.8-27B-UD-IQ4_XS.gguf` (14 GB) — ambos `general.architecture == "qwen35"`, 866 tensores, 65 blocos, ctx 262144, `rope.dimension_sections = [11,11,10,0]`, vocab 248320, chat template embutido (~10 KB), `tokenizer.pre = qwen35`, BOS 248044 / EOS 248046 / PAD 248055. Defaults de sampling nos metadados: top_k 20, top_p 0.95, temp 1.0. Arquivos `mmproj-*` (visão) são ignorados.
- Modo `--chat`: aplica o chat template dos metadados do GGUF antes do prompt (modelo instruct sem template é inútil).

### 1.2 Loader GGUF (só Qwen denso)
- Arquivo único (sem shards) com `general.architecture == "qwen35"`.
- Tensores do layout `QWEN35` (`docs/gguf-qwen-quantizacao-llamacpp.md` §1): embedding/norms/output, attn GQA + QK-norm + post-norm + `attn_qkv`, FFN `gate/down/up`, SSM `a/conv1d/dt/norm/beta/alpha/out`.
- KV obrigatórias validadas no load (fail-fast): `block_count`, `embedding_length`, heads/KV, e **`qwen35.rope.dimension_sections`**.
- Tipos de tensor aceitos: o union misto dos arquivos UD de teste — `F32`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ1_S`, `IQ2_XXS/XS/S`, `IQ3_XXS/S`, `IQ4_NL/XS` (+ `F16`/`BF16` se aparecerem). Qualquer outro tipo no arquivo → erro explícito, não fallback silencioso.
- Layout por camada (verificado nos `.gguf` reais, 65 blocos): **48 camadas lineares GDN** (`attn_qkv` fundido + `attn_gate` + tensores `ssm_*`), **16 camadas full-attention** a cada 4 (`attn_q/k/v` separados + `attn_q_norm/k_norm` + `attn_output`, sem SSM), e o bloco 64 carrega ainda `nextn_*` (draft MTP — **ler e ignorar**). O grafo decide o ramo por camada pela presença dos tensores (ou por `qwen35.full_attention_interval = 4`).
- Peculiaridades: `ssm_a` sem sufixo `.weight`, `ssm_dt.bias` é bias `F32`, `ssm_alpha/beta` em `Q8_0`, norms em `F32`.

### 1.3 Backend HIP, só gfx1201
- Build compila **apenas** o backend HIP com `-DCMAKE_HIP_ARCHITECTURES=gfx1201` (nada de CUDA/Vulkan/CPU multi-backend no binário).
- Em runtime, verifica o arch da GPU e **recusa qualquer coisa != gfx1201** com erro claro.
- Checagem de orçamento de VRAM antes do load (tabela §3); estouro → erro, nunca OOM silencioso.

### 1.4 Inferência (decode + prefill)
- Forward completo do denso Qwen3.8: full attention (GQA) + Gated Delta Net linear, RoPE/MRoPE, FFN SwiGLU, norms.
- KV cache para decode incremental com **quantização configurável por K/V** (`f16` default, `q8_0`, `q4_0` — mesmos tipos do llama.cpp em `common/arg.cpp` L304-314); o kernel de atenção desquantiza on-the-fly. Sem isso o ctx máximo prático seria ~55K (F16); com `q4_0` o teto estimado vai a ~190K no IQ3_S; prefill em batch + decode token-a-token.
- Kernels GEMM/matvec reaproveitados do llama.cpp com tuning RDNA4 (`MMVQ_PARAMETERS_RDNA4`, `mmq-config-rdna4.cuh`) + dequant GPU de Q4_0/IQ3_S.
- Sampler: greedy, temperature, top-k, top-p, min-p, repetition penalty, seed para determinismo.

### 1.5 Qualidade mínima
- Teste de fumaça: prompt fixo com saída esperada (golden) em Q4_0; perplexidade de referência documentada por quant suportado.
- README com build (ROCm no Linux) e exemplos de `run`.

## 2. NÃO VAI TER (v1 — não-goals explícitos)

- **Outros modelos**: `qwen35moe`, Qwen3.5/3.6/4Exp, qualquer outra família (Llama, Mistral, DeepSeek...), multimodais/vision, MTP/draft heads, speculative decoding.
- **Outras interfaces**: servidor HTTP/OpenAI-compatible, biblioteca importável, bindings Python, GUI.
- **Quantizar**: geração dos `.gguf` é com `convert_hf_to_gguf.py` + `llama-quantize` upstream; o engine só lê.
- **Outros quants**: fora do union misto dos arquivos UD (`MXFP4`/`NVFP4`, ternários); `Q4_K_M` puro (~16–17 GB) e `Q8_0`/`F16` full não cabem nos 16 GB.
- **Outro hardware/OS**: NVIDIA, CDNA/Instinct, `gfx1100/gfx1200`, APUs, Windows. Uma GPU, um OS (Linux + ROCm).
- **Shards GGUF**, offload CPU/GPU parcial, multi-GPU, tensor parallel.
- **Treino/finetune**, RLHF, imatrix, avaliação embutida.
- **Gramáticas/constrained decoding** (JSON schema etc.) — sampler cobre só o §1.4.

## 3. Orçamento de VRAM e números medidos (27B, arquivos reais)

Medido no M5 com `rdna4-infer bench`/`info` na 9070 XT (16 GB), tabela completa em
`docs/medicoes-m5.md`. "fill" = cache KV e estado recorrente pré-semeados
(`--fill-cache`), que é como o decode em contexto longo é medido sem esperar por um
prefill real.

| Arquivo | Peso | ctx / KV | VRAM em uso | Decode | Cabe em 16 GB? |
|---|---|---|---|---|---|
| `UD-IQ3_S.gguf` | 11,21 GiB | 4096 / f16 | 11,87 GiB | 27,6 tok/s | sim, folgado |
| `UD-IQ3_S.gguf` | | 32768 / f16 | 13,61 GiB | 23,3 tok/s¹ | sim |
| `UD-IQ3_S.gguf` | | 65536 / f16 | ~15,6 GiB | 19,0 tok/s¹ | sim, apertado |
| `UD-IQ3_S.gguf` | | 65536 / q4_0 | 12,73 GiB | 17,8 tok/s¹ | sim |
| `UD-IQ3_S.gguf` | | 131072 / q4_0 | 13,86 GiB | 13,2 tok/s¹ | sim |
| `UD-IQ4_XS.gguf` | 13,27 GiB | 4096 / f16 | 13,91 GiB | 27,0 tok/s | sim |
| `UD-IQ4_XS.gguf` | | 32768 / f16 | 15,66 GiB | 20,8 tok/s¹ | sim, apertado (0,26 GiB livres) |
| `UD-IQ4_XS.gguf` | | 65536 / q4_0 | 14,79 GiB | 17,1 tok/s¹ | sim |
| `UD-IQ4_XS.gguf` | | 131072 / q4_0 | — | — | **não** (`hipMalloc failed`) |

¹ **Decode no fim do contexto** (`bench --start-pos`), não no início: a atenção cresce
linearmente com a posição e domina acima de ~8K (8,9 ms/token em 4K, 138 ms com f16 em
64K). **Correção do M6:** a primeira versão desta tabela mostrava ~28 tok/s em 64K/131K;
aquilo era decode na posição 5-40 com um cache grande apenas *alocado* (o `--fill-cache`
não movia a posição) — um teste de caber, não de contexto longo. O M7 resolveu isso (`docs/medicoes-m7.md`): cargas vetorizadas, 8 warps por CTA (o texto dizia 32 — o commit `a830570` anunciava 32 mas a linha embarcada sempre foi 8) e a
faixa de chaves dividida entre CTAs com merge online-softmax. O decode agora é quase
plano no contexto (23,7 tok/s em 4K, 24,1 em 16K, 19,0 em 64K, 13,2 em 131K — contra
22,1/13,7/4,2/2,3 antes) e o prefill longo deixou de degradar (27,7 tok/s em 2048
tokens contra 28,8 em 512). Em 64K o f16 passou a ser mais rápido que o q4_0
(19,0 vs 17,8), então `q4_0` é alavanca de VRAM, não de velocidade.

**Correção da estimativa anterior desta seção:** a versão do M0 dizia que o IQ4_XS
estouraria com 32K de contexto f16. Medido: 32K f16 **cabe** (15,66 GiB em uso,
0,26 GiB livres — no limite, mas roda); o que não cabe é **131K** com `q4_0`. O
orçamento do `info` recusa o caso de 32K por ser conservador (assume 1,00 GiB de
overhead; o motor usa ~0,40 GiB) — é um guard pré-voo, não uma medição.

Tempos por arquivo (prompt de 512 tokens, decodificação greedy):

| Arquivo | Prefill (por token) | Decode | Banda efetiva |
|---|---|---|---|
| `UD-IQ3_S.gguf` | 28,8 tok/s (17,75 s) | 27,6 tok/s | 336 GB/s |
| `UD-IQ4_XS.gguf` | 27,4 tok/s (18,67 s) | 27,0 tok/s | 364 GB/s |

Referência na mesma máquina (`llama-bench -ngl 99`, backend Vulkan/RADV, mesmo
arquivo): **decode 39,7 tok/s**, **prefill em batch 440 tok/s**. Ou seja: decode a
~70 % da referência; **prefill é a lacuna grande (16×)**, porque este motor processa
o prompt token a token (ver `docs/medicoes-m5.md` §5).

Qualidade (wikitext-2, 10 chunks de 512 tokens pontuados, comparados **posição a
posição** com o mesmo modelo no llama.cpp): PPL dentro de **0,25 %** no IQ3_S e
**0,15 %** no IQ4_XS.

`Q4_K_M` puro (~16–17 GB) segue fora da v1; `Q8_0` (~29 GB) e `F16` (~54 GB) idem.

## 4. Futuro (fora da SPEC, registrado para não perder)

Servidor OpenAI-compatible · `qwen35moe` · `Q4_K_M` com offload · `MXFP4` nativo · gramáticas · **MTP** (tensores `nextn_*` já presentes no bloco 64 — hoje lidos e ignorados) · **mmproj de visão** (arquivos `mmproj-F16`/`mmproj-q8_0` já existem no diretório de teste) · módulos em Rust com a v1 C++ como referência de comportamento.
