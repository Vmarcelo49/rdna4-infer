# Exploração da arquitetura Qwen3.8 27B — plano vago

Este doc não é compromisso: é o mapa do que precisamos entender do **único modelo que importa**
(`Qwen3.8-27B`, torre de texto `qwen3_5`, 65 blocos no GGUF) antes e durante M1–M3.
Perguntas respondidas aqui viram regras no loader/grafo; o resto continua em aberto.

Fontes: `config.json` + KVs/tensores dos `.gguf` UD, `docs/gguf-qwen-quantizacao-llamacpp.md` §1,
`docs/referencias-upstream-gfx1201-qwen35.md` (SGLang `qwen3_5.py`, hipfire `forward.rs`).

## 1. O que já sabemos (fatos medidos)

- Dims: hidden 5120, FFN 17408 (SwiGLU), 24 heads Q / 4 KV, head_dim 256, rms_eps 1e-6, vocab 248320.
- 64 camadas de texto (índices 0–63) + bloco 64 do MTP: 48 GDN lineares, 16 full-attention (a cada 4, começando na 3).
- GDN linear: key 128×16 heads, value 128×48 heads, conv kernel 4, `mamba_ssm_dtype float32`, `attn_output_gate` (swish).
- RoPE: theta 1e7, `partial_rotary_factor` 0.25, MRoPE intercalado `[11,11,10]` (+0 no GGUF).
- UD files: mistura de 15 tipos, norms/bias em F32, `ssm_alpha/beta` em Q8_0, `output` Q5_K, `token_embd` Q3_K.
- Bloco 64 carrega `nextn_*` (draft de 1 camada, `mtp_num_hidden_layers: 1`) — preservado, ignorado.

## 2. Perguntas abertas (por prioridade)

1. **Mapeamento tensor→op nas 48 lineares**: o que exatamente consomem `attn_qkv` fundido e `attn_gate` no ramo GDN? (candidatos: `build_qkvz`/`build_norm_gated` em `qwen35moe.cpp` L251-278, `Qwen3_5GatedDeltaNet` em SGLang L322-929).
2. **Estado recorrente do GDN por token**: qual o tamanho real do estado (ssm_state 128 × groups 16 × ...)? Define o custo de decode das lineares e o formato do nosso "KV" nelas.
3. **`ssm_a`/`ssm_dt` em F32**: foi escolha de estabilidade do Unsloth ou requisito do arch? Testar `q8_0` nesses tensores seriaqtis economia relevante?
4. **Divergência de EOS**: `config.json` diz `eos_token_id 248044` (= BOS), o GGUF diz 248046. Qual termina geração de verdade? (testar no llama-cli e fixar no golden test).
5. **Sensibilidade por tensor**: os 496 `imatrix` entries dizem quais tensores doem mais ao quantizar? Cruzar com o mix UD para priorizar `--tensor-type` de alta precisão.
6. **KV `q4_0` nas full layers**: qual a perda real vs F16 em 64K+? (só dá para medir em M5; até lá, tratar `q8_0` como default seguro e `q4_0` como experimental).
7. **Bloco 64 como draft futuro**: o que o MTP de 1 camada precisa além dos `nextn_*` (embed próprio? não — `mtp_use_dedicated_embeddings: false`, reusa o da torre)? Só mapear, não implementar.

## 3. Frentes de exploração (sem ordem prometida)

- **E1 — Auditoria tensor→op**: para cada um dos 14 tensores do bloco linear e 11 do full, anotar qual op o consome (tabela). Ferramentas: `scripts/inspect_gguf.py`, `qwen35.cpp`, SGLang `qwen3_5.py`.
- **E2 — Sensibilidade a quant**: rodar perplexidade variando 1 tensor por vez (via `llama-quantize --tensor-type`) nos candidatos caros (`ssm_out`, `attn_qkv`, `ffn_down`). Cara, fazer por amostragem.
- **E3 — Estado GDN no decode**: instrumentar tamanhos e banda do estado recorrente; decide se prefill das lineares pode ser chunkado agressivo (hipfire usa batch 384 em gfx1201 — ver se o número faz sentido aqui).
- **E4 — Posições longas**: validar MRoPE além de 32K (o baseline mediu 32K; 64K/128K ficam para M5 com KV quantizado).
- **E5 — Mapear o draft**: documentar o formato `nextn_*` do bloco 64 para o futuro MTP (SPEC §4), sem escrever código.

## 4. Regra de saída

Nada deste doc vira código sem antes virar linha em `SPEC.md` (regra) ou `PLAN.md` (passo com aceite).
Achados vão para `docs/` como notas curtas, não neste arquivo.
