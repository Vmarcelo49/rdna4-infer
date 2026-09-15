# Pendente: adaptar as ideias do NInfer — fila restante

Estado em 14/09, tarde. Companheiro de `docs/plano-ninfer.md` (a decisão de *o que* adaptar, com
os três filtros) e de `docs/plano-prefill.md` (o plano de prefill). Este arquivo é só a **fila
restante**, com o que já caiu, o que falta, e como cada item entra.

## 0. Onde estamos (para o leitor não refazer caminho)

**Já em produção, medido e com gate:**

| o que | ganho medido | gate |
|---|---|---|
| GEMM tilejado int8 (`gemm.cuh`) cobrindo **iq3_s + iq3_xxs + iq4_xs = 68,0 %** dos bytes | 13,5-17,4 T-MAC/s contra 5-6 do GEMV em lote | **bit-exato contra o `vec_dot` do motor** (0/4096, max ulp 0) |
| **chunk 128 default** + `Graph::prefill()` único para CLI e servidor | prefill 512: **104,86 → 231,83 tok/s (2,21×)**; o servidor, que prefillava token a token, ganhou o lote | `check_server` 118/0; `check_batch_gpu` com **tolerância declarada** (`kTolChunkRelL2 1e-6`) só acima de 16 tokens, bit-exato até 16 |
| política de splits por **tipo de KV** (N-POL) | atencao f16: **1,047× a 4K, 1,089× a 8K, 1,030× a 16K, 1,023× a 32K, 1,020× a 64K** (A/B 15/15); E2E −2,13 % a 512 keys, −1,56 % a 16K, −1,10 % a 32K | `check_attn_split` PPL <0,03 % (limite 0,5 %); `kAttnSplitCtasDense` protegido pelo `check-tuning` |
| KV `q5_0`/`q4_1` consertado (stride de V usava o de K) | par do alvo deixou de dar page fault; `q8_0/q4_1` **corrompia memória em silêncio** | `scripts/check_kvbatch.sh`, 5 pares, todos bit-exatos |

**Fechados por medição — não reabrir sem um número novo:**

| ideia | por que morreu |
|---|---|
| decodificar no laço de consumo, nunca LDS→LDS (padrão central do ninfer) | 1,7-4,8× **mais lento**; o nosso staging é limitado por **tráfego de DRAM** (572 GB/s = 90 % do pico), não por instrução |
| GQA fundido no decode **a 4K** | com CTA igualada, 1,78-2,03× mais lento (80-100× o piso) |
| drafter em bloco (DFlash2) | nosso draft custa 1/12-1/14 de um forward; já estamos no regime barato; upside 5,4 % |
| ReplaySSM (registrar entradas em vez de snapshot) | 86× em **bytes**, <1 % em **tempo** (a cópia é ~1 % do round) |
| PDL | **não existe** no ROCm 7.2.4 (sonda de compilação) e atacaria a rampa de GPU, não o nosso enfileiramento de CPU |
| fusão de kernels em massa | prefill 1,1 %, decode ~1-2 % |
| sampler do host como resíduo do MTP | 3,2 %, não 12-20 %; a causa do meu número errado era bug de unidade (corrigido) |

---

## 0.1 FALSO BUG (resolvido 14/09, noite): o fallback de sub-lote está CERTO

**Hipótese refutada por medição.** O "bug de stride" não existe: o quantizador
empacota denso e os deslocamentos estão certos. Provas, todas com GPU-lock:

- fallback 2×16 vs 32×16×1: **bit-exato** nos 6 tipos testados
  (iq4_xs, iq1_s, iq2_xs, q4_K, iq3_xxs, q5_K);
- com o GEMM **desligado**, n = 32 dá **0.0e+00 em tudo** (curto, prompt cheio,
  contexto longo com splits) — o caminho exato ponta a ponta existe;
- kernels compartilhados (rms_norm, quantize, rope, deinterleave) bit-exatos
  em n = 32; `RD_PREFILL_BATCH=0` dá números **idênticos** (o meio está livre).

O resíduo é **arredondamento do GEMM amplificado**: 2e-07 (IQ) a 9e-07 (q6_K)
por projeção → 1e-02..6e-02 ponta a ponta, com picos determinísticos até
**2.64e-01** (IQ4_XS, prompt curto, n = 128, com KQ; 4.7e-02 sem KQ). Os
tensores suspeitos estão limpos (staging 0/2048, oráculo de CPU bit-exato) e o
perfil por camada cresce suave (~9 %/camada) — caos, não bug. **Neutra em
qualidade**: PPL chunk-128 == chunk-16 dígito a dígito (pior desvio 0,354 % vs
limite 0,5 %) e golden inalterado.

**Consequência**: default de volta a **`RD_PREFILL_CHUNK=128`**, com a
tolerância HONESTA declarada em `tests/check_batch_gpu.hip`
(`kTolChunkRelL2 = 5e-01`, `max|d|` como guarda catastrófica) + checks por tipo
(`kTolPerType`, 8 solvers tensor a tensor) + pino de VRAM a 128 no
`check_kvtype` (269.822.436 B; 131K q5_0/q4_1 continua cabendo, 14,19 GiB).

---

## 1. KQ — FECHADO 14/09 noite (medido, tolerado, reabilitado)

**Estado**: os cinco solvers entraram no despacho. A tabela de velocidade do
cabeçalho de `gemm.cuh` foi **re-medida de forma independente**
(`bench-gemm-engine-gpu --kq`, min de 5, piso de ruído ≤1,006x) e **confere
dentro de 1-3 %** — incluindo staging (0/2048), oráculo de CPU bit-exato e as
divergências vs `vec_dot` (q2 4,51e-07, q3 7,39e-07, q4 8,44e-07, q5 4,69e-07,
q6 9,23e-07, todas confirmadas dígito a dígito).

Decisão por tipo (regra: ≥1,5× entra; M=16 nunca perde — o pior, q3_K a 1,09×,
continua sendo ganho, e n ≤ 16 nem usa GEMM):

| tipo | M=16 | M=64 | M=128 | M=512 | veredito |
|---|---|---|---|---|---|
| q2_K | 2,11× (6,42 T) | 3,30× (9,09 T) | 3,50× (9,45 T) | 3,86× (10,27 T) | ENTRA |
| q3_K | 1,09× (3,33 T) | 2,54× (6,95 T) | 3,42× (9,54 T) | 4,44× (12,17 T) | ENTRA |
| q4_K | 1,99× (4,26 T) | 2,87× (7,42 T) | 4,17× (10,61 T) | 6,40× (13,12 T) | ENTRA |
| q5_K | 2,71× (6,86 T) | 3,67× (11,37 T) | 3,70× (11,42 T) | 3,75× (11,51 T) | ENTRA |
| q6_K | 1,49× (3,62 T) | 3,32× (7,44 T) | 4,58× (10,21 T) | 5,62× (12,46 T) | ENTRA |

(T-MAC/s e razões vs GEMV em lote; 1,49× do q6_K em M=16 está no piso de ruído
da barra de 1,5× e nunca é perda.)

Cobertura 68,0 % → **89,9 %** dos bytes; prefill curto a 128: 7,17× → **8,33×**
sobre o per-token no IQ3_S (10,53× no IQ4_XS). Tolerância por tipo declarada e
**cobrada** em `check_batch_gpu.hip` (`kTolPerType` + `check_per_type()`, M=32
tensor a tensor); o `check-batch-gpu` passa nos dois arquivos UD. O que falta
deste item: `iq2_s/iq2_xs/iq2_xxs` (9,4 %, depois dos precedentes).

---

## 2. GDN — o port em blocos (autorizado, sequenciado)

**Estado**: especificado e **provado exato** por `docs/estudo-ninfer-gdn.md` (698 linhas): float64
**1,0e-15…1,7e-15** contra o nosso sequencial para B ∈ {8,16,32,64} e T até 8192; em fp32
5,0e-7…1,0e-6, a mesma ordem do erro que o nosso sequencial já tem; cond(I−A) ≤ 6,2 (a inversa não
amplifica); o layout do estado persistente é **idêntico** ao nosso (`[valor][chave]` fp32, sem
conversão). Autorizado pelo dono. Alvo: `gdn_delta` **0,625 → ≤0,30 ms/token**.

**Por que não é o próximo**: vale ~5 % do prefill hoje. Depois de KQ, o matvec cai tanto que ele
passa a ser a maior fração do que sobra — é aí que entra.

**Cuidados obrigatórios, todos medidos pela pesquisa**:
- **não copiar os dtypes deles**: `W`, `U`, `v_new`, `h_chunk` são BF16 no ninfer; emular só isso dá
  3,0e-3…4,9e-3 — **5000× pior** que o nosso gate possível (fp32, rel-L2 ≤ 1e-5).
- três armadilhas de convenção, cada uma vira asserção de teste: cumsum tem de ser **inclusivo**
  (exclusivo dá 0,275 de desvio), a diagonal da saída tem de **incluir** `s=r` (estrita dá 0,824),
  e `β` entra pela **linha** (coluna dá 1,1e-2…2,6e-2).
- a cauda: ninfer recusa T não múltiplo de 64; **usar preenchimento inerte** (o truque do llama.cpp),
  o que de quebra deixa a aritmética depender **só da posição, não do corte** — a mesma propriedade
  que protege CLI/servidor/MTP de divergirem.
- infraestrutura: workspace novo (789.504 B com n=16/B=16) ⇒ `device.h` + o pino de
  `check_kvtype` andam junto, sempre no mesmo commit. ATUALIZADO 14/09 noite: a
  base agora é 269.822.436 B (cap 128), então o pino com GDN passa a ser
  269.822.436 + ws_bytes (não mais 172.258.340 + 789.504).

## 3. Atenção — E1/E2 (a parte do GQA que o E0 não fechou)

O E0 fechou a 4K e **reabriu a 16K**: com grade larga, HG=3 mede **0,84-0,91×** do kernel que
embarca, 8/8 rodadas, e em grade fixa a fusão sozinha vale **1,29×**. A N-POL já capturou a maior
parte disso **sem kernel novo** (política por tipo de KV, commitada). O que resta é o desenho que o
ninfer usa e o nosso protótipo C não tem:

- **E1 (protótipo D, "tile de linhas")**: grade `(n_head_kv, S)`, 6 warps (uma por cabeça), K/V
  decodificado uma vez na LDS, acumulador **particionado** entre warps em vez de replicado. É a
  diferença estrutural que a frente N1 isolou: nós dividimos **chaves** (cada warp carrega o estado
  de todas as cabeças → 18× de replicação → o derrame de 400 B/lane que matou o HG=6); eles dividem
  **linhas** (16 f32/lane). ~200 linhas, só em `tests/bench_attn_gpu.hip`.
  **Alvos escritos antes de rodar**: ≥1,02× ⇒ 4K ≤0,0709, 16K ≤0,2381, 64K q4_0 ≤1,0897.
- **E2** só se E1 ganhar: eixo de T>1 (verify/MTP), onde o tile de linhas paga de verdade.
- **Limitação honesta do caminho**: a 4K o ganho já está morto; a 64K/131K **não há como medir E2E
  hoje** porque f16 a 64K não cabe (4 GiB de KV) e KV quantizado é justamente onde a política
  antiga (melhor) se mantém. Ou seja, E1 vale **a 16K/32K em f16** e mais nada, até haver um
  caminho de KV quantizado que aceite a regra de 96 CTAs.

## 4. MTP — o item certo agora

Re-ranking medido do round (D=3, 67,29 ms): **verify 43,2 (64 %) + replay do rollback 12,1
(18 %) = 82 % é forward do tronco**; drafts/KV-rebuild 8,6 (13 %); sampler 2,2 (3,2 %); resíduo
1,1-1,7 (1,7-2,5 %). O replay dispara em 10 de 28 rounds e é **5,6× o sampler**. Então:

1. replay do rollback (18 % do round) — atacar com o que o KQ/GEMM já entrega e, se não bastar, com
   estado por linha em vez de reexecutar;
2. **pequeno-almoço grátis**: FECHADO 14/09 noite — `bench --mtp/--draft*` agora
   responde que é run-only (antes: "unknown argument" mudo) e o `--mtp` do
   `run` diz "(greedy only)" no usage; `--mtp --temp 0.8` já errava por
   construção;
3. D=1/2/3 já empatam em ~51 tok/s ⇒ o eixo do número de drafts **saturou**; não investir aí.

## 5. Ordem e custo

| # | item | custo | ganho esperado | risco |
|---|---|---|---|---|
| 0 | **fallback**: era falso bug — fallback bit-exato, tolerância honesta + PPL | FECHADO 14/09 noite | desbloqueia o 128 | — |
| 1 | **KQ**: medir, declarar tolerância, reabilitar | FECHADO 14/09 noite (8,33× a 128 no lote) | **~1,4× no prefill** (a medir E2E) | numérico, gated |
| 2 | **GDN em blocos** | 1 sessão + workspace | ~5 % hoje, maior depois do KQ | quebra bit-exatidão ⇒ gated por PPL |
| 3 | **E1 atenção** (tile de linhas) | ~200 linhas no bench | 1,29× na fusão, só a 16K/32K f16 | numérico, gated por PPL |
| 4 | **MTP rollback replay** | a medir primeiro (já orçamentado) | até 18 % do round | contrato do md5 |
| — | E2, f32 KV, `iq2_*` | depois dos seus precedentes | — | — |

**Nota de processo, porque foi assim que o KQ quase pegou fogo**: agente nenhum pode deixar
trabalho na árvore sem relatório; e eu não faço mais `git add -A` em workspace com frentes
concorrentes sem antes rodar `git status` e olhar o que está entrando. Os dois estão no passo 1
desta fila de propósito.
