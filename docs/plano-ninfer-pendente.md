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

## 0.1 BUG ABERTO (é o passo zero de tudo): o fallback de sub-lote está errado

Descoberto ao desligar os k-quants: **com eles ligados o fallback nunca era exercitado**, e os
gates passavam por isso. Com ele exercitado, `check-batch-gpu` falha:

| caso | erro L2 rel. | max\|d\| | veredicto |
|---|---|---|---|
| n = 2..16 (GEMV em lote) | **0,0** | 0,0 | BIT-EXATO ✓ |
| prompt de 48 tokens ⇒ chunk de **32** | **8,7e-03** | 1,7e-01 | FORA DA TOLERÂNCIA (limite 1e-06) |
| contexto longo, n = **128** | **6,1e-02** (8142 de 8192 linhas) | 4,5e+01 | FORA DA TOLERÂNCIA |
| contexto longo, n = **16** | 0,0 | 0,0 | BIT-EXATO ✓ |

É **duas ordens de magnitude** acima do que erro de ordem de soma produziria (o GEMM dá 2,4e-07),
logo é índice/stride errado, não numérica. O suspeito é o deslocamento por token no laço de
sub-lotes de `proj_batch` (`d_aqb_ + t0*act_stride` / `d_y + t0*nrows`), com `act_stride` vindo de
`ncols/QK8_1` enquanto o scratch `d_aqb_` é dimensionado por `aq_blocks_per_row_` (o **maior** K do
modelo, 544 blocos, não os 160 de um K=5120) — se o quantizador escreve com o stride do scratch e
não com o do tensor, o primeiro sub-lote acerta e os seguintes leem a linha errada, o que bate com
o erro crescer com n.

**Consequência**: o **default voltou para `RD_PREFILL_CHUNK=16`** (estado verificado: tudo
bit-exato). O chunk 128 com o GEMM **está correto e é real (2,21× medido)** e volta assim que o
fallback estiver certo — ou for substituído pelo caminho por token para os tipos não cobertos, que
é correto por construção e custa pouco (a cobertura IQ já dá quase todo o ganho).

**Como atacar (30 min)**: um caso mínimo, `check-batch-gpu` já imprime o perfil por linha do chunk
(t0..t127) — imprimir o erro por *tensor* de um tipo não coberto (q3_K) responde em uma execução se
é o stride do scratch. O teste que não existe hoje e deveria: **chunk grande com um tensor de cada
tipo não coberto**, porque é só ali que o fallback aparece.

---

## 1. KQ — terminar o que já está escrito (maior valor, menor esforço)

**Estado**: os cinco solvers k-quant (`q2_K, q3_K, q4_K, q5_K, q6_K`) estão **implementados** em
`gemm.cuh` e **desligados do despacho** por um motivo legítimo: entraram na árvore varridos por um
`git add -A`, sem relatório de velocidade (o agente falhou antes de reportar). O que se sabe
deles, medido no bench `--kq`:

- staging **bit-exato** (0 de 256 bits diferentes do dequant);
- GEMM **diverge do `vec_dot` do motor** — rel-L2 **4,51e-07** (q2_K) e **7,39e-07** (q3_K),
  `max|d|` 5,7e-06 / 1,3e-05. Plausivelmente só ordem de soma: os k-quants têm **duas cadeias fp32
  por saída** (`sumf_d`/`sumf_m`) e escala por grupo dentro do bloco, então reproduzir bit a bit
  custaria 2× acumuladores;
- **velocidade: nunca medida**.

**Por que desligar e não deixar**: autorização para quebrar o contrato não é autorização para
embarcar ganho não medido. O caminho de volta é de graça — `gemm_launch` devolve `false` e o
`proj_batch` cai no GEMV em sub-lote, que é o estado verificado.

**O que fazer** (meia sessão):
1. Rodar `build/bench-gemm-engine-gpu <model> --kq --reps 5` e registrar T-MAC/s por tipo contra o
   GEMV em sub-lote, nos tensores reais que o bench já seleciona (q2_K `blk.3.attn_q`, q3_K
   `blk.7.ffn_down`, q4_K `blk.63.ffn_down`, q5_K `output.weight`, q6_K `blk.64.ffn_down`).
2. Decidir por tipo, com número: se ≥1,5× do GEMV, entra; se não, o solver fica no arquivo e o
   despacho continua desligado, com a razão escrita.
3. **Tolerância por tipo declarada** no `check_batch_gpu.hip` (não uma só para todos) + linha no
   doc dizendo que a rota k-quant é numericamente gated e por quê.
4. Reabilitar as cinco linhas, rodar os gates completos.

**Ganho esperado, com aritmética**: k-quants = **21,9 %** dos bytes. Cobertura vai de 68,0 % a
89,9 %, e o matvec passa de ~2,0× a ~3,1× o pré-D2 (tempo relativo `0.899/3.4 + 0.101/1 = 0.365`).
Matvec é 83,7 % do prefill ⇒ **préfill ~1,4× sobre os 231 tok/s atuais** (ordem de ~320-330 tok/s
a 512). Depois, `iq2_s/iq2_xs/iq2_xxs` (9,4 %) levariam a ~99 % de cobertura e ~3,9× no matvec.

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
  `check_kvtype` (172.258.340 → 173.047.844 B) andam junto, sempre no mesmo commit.

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
2. **pequeno-almoço grátis**: `bench` não aceita `--mtp` (só `run`), e `--mtp --temp 0.8` sai com
   erro — MTP é ganancioso por construção; documentar isso evita a próxima pessoa tentar o gate
   errado;
3. D=1/2/3 já empatam em ~51 tok/s ⇒ o eixo do número de drafts **saturou**; não investir aí.

## 5. Ordem e custo

| # | item | custo | ganho esperado | risco |
|---|---|---|---|---|
| 1 | **KQ**: medir, declarar tolerância, reabilitar | meia sessão (código já escrito) | **~1,4× no prefill** | numérico, gated |
| 2 | **GDN em blocos** | 1 sessão + workspace | ~5 % hoje, maior depois do KQ | quebra bit-exatidão ⇒ gated por PPL |
| 3 | **E1 atenção** (tile de linhas) | ~200 linhas no bench | 1,29× na fusão, só a 16K/32K f16 | numérico, gated por PPL |
| 4 | **MTP rollback replay** | a medir primeiro (já orçamentado) | até 18 % do round | contrato do md5 |
| — | E2, f32 KV, `iq2_*` | depois dos seus precedentes | — | — |

**Nota de processo, porque foi assim que o KQ quase pegou fogo**: agente nenhum pode deixar
trabalho na árvore sem relatório; e eu não faço mais `git add -A` em workspace com frentes
concorrentes sem antes rodar `git status` e olhar o que está entrando. Os dois estão no passo 1
desta fila de propósito.
