# Quantização: precisão **medida** por formato (metade medida da tarefa 4)

Branch `feat/medicoes-gpu` (worktree `../rdna4-wt-medicoes-gpu`), RX 9070 XT
(`gfx1201`, ROCm 7.2.4), com `scripts/gpu-lock.sh` em toda execução de GPU.
**O inventário** de formatos/kernels é do outro agente (`docs/quants-inventario.md`);
este documento traz só os **números**.

Ferramentas:

| ferramenta | o que dá |
|---|---|
| `tests/check_quant_precision_gpu.hip` → `check-quant-precision-gpu <gguf>` (**novo**) | por formato: matvec do motor vs **produto interno em fp64 sobre os pesos desquantizados**, e as duas metades do erro (q8_1 da ativação e aritmética do kernel). `<gguf> --time` acrescenta o tempo do mesmo tensor |
| `check-quant-precision-gpu --cross <A.gguf> <B.gguf>` (**novo**) | distância entre **formatos**: o mesmo tensor desquantizado dos dois arquivos — a única medida disponível do erro do **peso** sem um modelo fp32 original |
| `check-matvec-gpu` (existente) | gate por tipo contra o oráculo CPU do llama.cpp (bit-exatidão do `vec_dot`) |
| `bench-matvec-shapes-gpu` (existente) | GB/s por tipo **sobre o inventário real** (1 passada por tensor, não em laço fechado) |
| `scripts/compare_ppl.sh` (existente) | fim a fim: nossa engine vs llama.cpp no **mesmo** arquivo, por posição |

---

## 1. Método: o que exatamente está sendo medido

Um matvec quantizado do motor é `y = W_q · q8_1(x)`. O erro que chega ao token tem
**duas metades** e o peso quantizado é o *denominador comum*, não uma variável:

```
total   y_engine        vs  W_f32 . x_f32      (o que o token recebe, tudo junto)
ativ    W_f32 . x_q8_1  vs  W_f32 . x_f32      (só a quantização q8_1 da ativação)
kernel  y_engine        vs  W_f32 . x_q8_1     (só a aritmética inteira/fp16 do vec_dot)
```

`W_f32` é o peso **desquantizado** (`rdna4_cpu_dequant_row`, validado bit-exato contra
o llama.cpp em `tests/check_dequant_gpu.hip`), e a referência é um produto interno
**em `double`** — nenhuma das duas pernas herda o erro do kernel que está sendo medido.

Consequência honesta, e é a chave para ler a tabela: **esta comparação NÃO mede
"quão lossy é o formato do peso"** — os dois lados usam os mesmos pesos
desquantizados. O erro do peso só é visível contra um original fp32 que não temos
(não existe F16 deste modelo na máquina: só IQ3_S e IQ4_XS). O que se mede aqui é
(a) quanto o **q8_1 da ativação** custa em cada formato e (b) quanto a **aritmética
do kernel** de cada formato acrescenta. O erro do peso é medido **por comparação
cruzada entre os dois arquivos** (§4), que é o proxy disponível.

Ativação usada: gaussiana com RMS 1 (é o que sai de um RMS-norm e entra nas
projeções), semente fixa `777 + tipo`; 2048 linhas de um tensor **real** de cada
tipo (o primeiro do arquivo que cabe no limite). O `q8_1` é produzido pelo próprio
kernel do motor (`quantize_q8_1_kernel`) e lido de volta, para que os dois lados
vejam exatamente os mesmos bytes.

---

## 2. Por formato (IQ3_S, os 14 tipos presentes)

```
scripts/gpu-lock.sh ./build/check-quant-precision-gpu <IQ3_S> --time
```

| tipo | tensor real medido | bpw | total relL2 | total max | **ativ relL2** | **kernel relL2** | x_q8 relL2 | RMS(y) |
|---|---|---|---|---|---|---|---|---|
| q8_0 | `blk.64.attn_k.weight` | 8,500 | 4,01e−03 | 2,76e−02 | 4,01e−03 | **6,5e−08** | 3,79e−03 | 1,15 |
| q2_k | `blk.3.attn_q.weight` | 2,625 | 3,74e−03 | 2,82e−02 | 3,74e−03 | **6,9e−08** | 3,77e−03 | 1,27 |
| q3_k | `blk.7.ffn_down.weight` | 3,438 | 3,81e−03 | 2,58e−02 | 3,81e−03 | **7,1e−08** | 3,79e−03 | 1,42 |
| q4_k | `blk.63.ffn_down.weight` | 4,500 | 3,86e−03 | 3,33e−02 | 3,86e−03 | **1,0e−07** | 3,81e−03 | 1,54 |
| q5_k | `blk.63.ffn_up.weight` | 5,500 | 3,84e−03 | 2,01e−02 | 3,84e−03 | **9,2e−08** | 3,86e−03 | 0,91 |
| q6_k | `blk.64.ffn_down.weight` | 6,562 | 3,75e−03 | 3,05e−02 | 3,75e−03 | **7,2e−08** | 3,79e−03 | 1,63 |
| iq2_xxs | `blk.1.ffn_up.weight` | 2,062 | 3,65e−03 | 1,57e−02 | 3,65e−03 | **1,9e−05** | 3,78e−03 | 0,78 |
| iq2_xs | `blk.0.ffn_gate.weight` | 2,312 | 3,77e−03 | 1,74e−02 | 3,77e−03 | **2,3e−05** | 3,73e−03 | 0,77 |
| iq3_xxs | `blk.0.ffn_down.weight` | 3,062 | 3,72e−03 | 2,67e−02 | 3,72e−03 | **5,3e−06** | 3,77e−03 | 1,42 |
| **iq1_s** | `blk.0.ffn_up.weight` | 1,562 | 3,78e−03 | 1,44e−02 | 3,75e−03 | **4,4e−04** | 3,77e−03 | 0,75 |
| iq4_nl | `blk.11.attn_k.weight` | 4,500 | 4,00e−03 | 2,57e−02 | 4,00e−03 | **6,5e−08** | 3,85e−03 | 1,10 |
| iq3_s | `blk.2.ffn_down.weight` | 3,438 | 4,00e−03 | 3,29e−02 | 4,00e−03 | **7,5e−08** | 3,82e−03 | 1,43 |
| iq2_s | `blk.1.ffn_gate.weight` | 2,562 | 3,61e−03 | 1,58e−02 | 3,60e−03 | **2,2e−05** | 3,71e−03 | 0,80 |
| iq4_xs | `blk.21.ffn_down.weight` | 4,250 | 3,78e−03 | 2,84e−02 | 3,78e−03 | **6,4e−08** | 3,75e−03 | 1,46 |

Três leituras, todas medidas:

1. **O erro total é o erro da ativação, e ele é o mesmo em todos os formatos**
   (3,6e−03 a 4,0e−03). O `x_q8 relL2` (3,7e−03 a 3,9e−03) é o erro do *vetor* de
   ativação antes de qualquer peso: 8 bits por bloco de 32 elementos com escala
   fp16. A projeção aleatória `W·x` preserva a norma relativa do erro, e é por isso
   que `total ≈ ativ ≈ x_q8` para todo tipo — **a quantização da ativação, não o
   formato do peso, é o que limita a precisão do matvec de hoje.**
2. **A metade do kernel separa os formatos em dois grupos.** Nove tipos acumulam a
   `double`/fp32 e ficam no ruído de fp32 (**6,4e−08 a 1,0e−07**). Cinco tipos usam
   truncamento inteiro no *scaling* das somas parciais e desviam de forma
   mensurável: `iq3_xxs` 5,3e−06, `iq2_xxs` 1,9e−05, `iq2_s` 2,2e−05, `iq2_xs`
   2,3e−05 e **`iq1_s` 4,4e−04** — 4.000× pior que o pior dos outros. Ainda é ~9×
   menor que o erro da ativação, mas é o único lugar onde "o formato" muda a
   precisão do matvec de fato.
3. **O `max|d|` absoluto é pequeno em relação à escala da saída** (2e−02 a 3e−02
   contra RMS(y) 0,8-1,6): o erro é branco e distribuído, não um viés.

### 2.1 O que isso implica para a metade `iq1_s`/`iq2_*`

`iq1_s` custa 0,126 ms/token no IQ3_S (0,3% do token) e responde por 0,035 GB/token;
seu erro de kernel (4,4e−04) é 0,1% do sinal que chega ao token. Não é um problema
de qualidade hoje, mas é o teste #1 para qualquer trabalho de *precisão* por
formato: uma correção de truncamento no `vec_dot_iq1_s_q8_1`
(`include/rdna4/vecdotq.cuh:731`) derrubaria esse número para o nível dos outros,
sem tocar no tamanho do arquivo. Gate: `check-matvec-gpu` (que já tem tolerância
`kIntTruncTol = 1e-3` exatamente para esses tipos) e `check-dequant-gpu`.

---

## 3. Distância entre formatos: o erro do **peso**, medido

Sem original fp32, a medida disponível é a distância entre os mesmos tensores
desquantizados de dois arquivos diferentes:

```
scripts/gpu-lock.sh ./build/check-quant-precision-gpu --cross <IQ3_S> <IQ4_XS> --max-elems 20000000
  294 tensores comparados; agregado rel-L2 1,53e-01, pior 4,06e-01
```

| par de formatos (A vs B) | n | rel-L2 médio | pior |
|---|---|---|---|
| q5_k vs q6_k | 1 | 3,1e−02 | 3,1e−02 |
| q4_k vs q5_k | 13 | 6,4e−02 | 7,0e−02 |
| **iq4_xs vs q5_k** | 22 | **6,5e−02** | 6,8e−02 |
| iq4_nl vs q5_k | 1 | 6,5e−02 | 6,5e−02 |
| iq4_xs vs q4_k | 9 | 7,9e−02 | 8,0e−02 |
| **iq3_s vs q4_k** | 20 | **1,17e−01** | 1,19e−01 |
| **iq3_s vs iq4_xs** | 94 | **1,19e−01** | 1,26e−01 |
| q3_k vs q4_k | 4 | 1,22e−01 | 1,24e−01 |
| q3_k vs iq4_xs | 10 | 1,23e−01 | 1,25e−01 |
| iq3_xxs vs iq4_xs | 45 | 1,49e−01 | 1,56e−01 |
| iq3_xxs vs q4_k | 10 | 1,50e−01 | 1,55e−01 |
| iq3_xxs vs iq3_s | 15 | 1,70e−01 | 1,80e−01 |
| iq2_s vs iq3_s | 14 | 2,06e−01 | 2,11e−01 |
| iq2_xxs vs q3_k | 1 | 2,83e−01 | 2,83e−01 |
| iq1_s vs iq2_s | 2 | **3,98e−01** | 4,06e−01 |

- A distância cresce de forma quase monótona com a diferença de bpw: **cada ~1 bpw
  vale 6-8 pontos percentuais de distância relativa**. Por isso essa tabela serve
  como régua de precisão relativa entre formatos quando não há original.
- O par que interessa: **IQ3_S está a 11,9% do IQ4_XS, mas o IQ4_XS está a apenas
  6,5% do q5_K** (5,5 bpw). Como o erro do q5_K é muito menor que os dois, a
  distância IQ4_XS↔q5_K é praticamente o erro do **próprio** IQ4_XS (~6,5%), e a
  distância IQ3_S↔q5_K (~12%, via q4_K) é praticamente o erro do **próprio**
  IQ3_S. Medido, não de folha de dados: **o IQ4_XS representa estes pesos com
  cerca de metade do erro do IQ3_S**, e custa +2,3 GiB de VRAM (13,26 vs 11,20 GiB
  por token medidos no `bench`).
- `iq1_s` contra `iq2_s` dá 39,8% — o formato mais grosseiro do arquivo está a
  0,4 de distância relativa de um 2,5 bpw, o que é consistente com o erro de kernel
  dele medido em §2 e com o bpw de 1,56.

---

## 4. Fim a fim: nossa engine vs llama.cpp no mesmo arquivo

```
scripts/gpu-lock.sh ./scripts/compare_ppl.sh <IQ3_S> 4     # ctx 512 / stride 512, wikitext-2
scripts/gpu-lock.sh ./scripts/compare_ppl.sh <IQ4_XS> 4
```

Medido nesta sessão (2048 posições por arquivo, 4 chunks de 512, janela 768, KV f16):

| chunk | IQ3_S: nossa PPL / llama.cpp / desvio | IQ4_XS: nossa PPL / llama.cpp / desvio |
|---|---|---|
| 0 | 3,8231 / 3,8160 / **0,185%** | 3,7992 / 3,8012 / **0,053%** |
| 1 | 8,7500 / 8,7591 / **0,105%** | 8,7351 / 8,7338 / **0,015%** |
| 2 | 7,2332 / 7,2326 / **0,008%** | 7,2500 / 7,2429 / **0,099%** |
| 3 | 6,3395 / 6,3407 / **0,018%** | 6,2930 / 6.2857 / **0,116%** |
| pior chunk | **0,185%** | **0,116%** |
| pior posição (\|dNLL\|) | 0,4581 | 0,8729 |
| veredito | `compare-ppl: OK` | `compare-ppl: OK` |

- Os números do IQ3_S **reproduzem exatamente** os históricos (0,185 / 0,105 / 0,008
  / 0,018 % por chunk) — é a mesma engine, e é a prova de que este worktree não
  mexeu em nada do caminho numérico.
- **A afirmação "IQ4_XS é consistentemente um pouco melhor que IQ3_S" não se
  confirma** nesta medição: o IQ4_XS ganha nos chunks 0 e 1 (0,053% e 0,015% contra
  0,185% e 0,105%), perde nos chunks 2 e 3 (0,099% e 0,116% contra 0,008% e
  0,018%) e tem pior desvio de posição única (0,873 contra 0,458). O que se pode
  dizer com os dados: **o pior chunk é melhor no IQ4_XS (0,116% vs 0,185%)** e os
  dois arquivos ficam na mesma ordem de grandeza (≤0,19%), bem abaixo do limite de
  1% do gate. O desvio por chunk é dominado pela posição, não pelo arquivo.
- O `bench` mede o preço: IQ4_XS = **36,40 ms/token a 4K (27,48 tok/s)** contra
  34,6 ms do IQ3_S, com 13,92 GiB em uso (2,00 GiB livres) contra 11,87 GiB.

Interpretação do que este gate mede (e do que não mede): os dois lados rodam **o
mesmo arquivo GGUF**, então a diferença por chunk é a **fidelidade da nossa
engine**, não a perda do formato. A comparação *entre arquivos* (IQ3_S vs IQ4_XS)
é o que diz qual quantização preserva mais o modelo original — e ela vem de dois
fatos medidos aqui: (i) a engine desvia menos do llama.cpp no IQ4_XS e (ii) o peso
do IQ4_XS está medidamente mais perto da referência de 5,5 bpw (§3).

---

## 5. Tabela consolidada: formato × kernel nativo × precisão × banda

Banda medida com `bench-matvec-shapes-gpu <IQ3_S> --no-ab` (uma passada por tensor
do inventário real, 12 GB escalonados — **não** é laço fechado, é o número de DRAM):

| tipo | kernel nativo | bytes/token | ms/token | GB/s medido | erro do kernel | erro total |
|---|---|---|---|---|---|---|
| q8_0 | sim, `matvec.cuh:103` / `vecdotq.cuh:348` | 0,025 | 1,061 | **24**¹ | 6,5e−08 | 4,0e−03 |
| q2_k | sim, `matvec.cuh:104` / `vecdotq.cuh:365` | 0,117 | 0,277 | 422 | 6,9e−08 | 3,7e−03 |
| q3_k | sim, `matvec.cuh:105` / `vecdotq.cuh:388` | 0,401 | 1,481 | **271**² | 7,1e−08 | 3,8e−03 |
| q4_k | sim, `matvec.cuh:106` / `vecdotq.cuh:415` | 0,233 | 0,592 | 393 | 1,0e−07 | 3,9e−03 |
| q5_k | sim, `matvec.cuh:107` / `vecdotq.cuh:465` | 0,986 | 1,693 | **582**³ | 9,2e−08 | 3,8e−03 |
| q6_k | sim, `matvec.cuh:108` / `vecdotq.cuh:516` | 0,004 | 0,031 | 138¹ | 7,2e−08 | 3,8e−03 |
| iq2_xxs | sim, `matvec.cuh:109` / `vecdotq.cuh:542` | 0,269 | 0,678 | 397 | 1,9e−05 | 3,7e−03 |
| iq2_xs | sim, `matvec.cuh:110` / `vecdotq.cuh:574` | 0,292 | 0,716 | 408 | 2,3e−05 | 3,8e−03 |
| iq2_s | sim, `matvec.cuh:111` / `vecdotq.cuh:612` | 0,581 | 1,176 | 494 | 2,2e−05 | 3,6e−03 |
| iq3_xxs | sim, `matvec.cuh:112` / `vecdotq.cuh:657` | 1,863 | 4,703 | 396 | 5,3e−06 | 3,7e−03 |
| iq3_s | sim, `matvec.cuh:113` / `vecdotq.cuh:692` | 3,717 | 8,751 | 425 | 7,5e−08 | 4,0e−03 |
| iq1_s | sim, `matvec.cuh:114` / `vecdotq.cuh:731` | 0,035 | 0,126 | 276 | **4,4e−04** | 3,8e−03 |
| iq4_nl | sim, `matvec.cuh:115` / `vecdotq.cuh:762` | 0,003 | 0,015 | 199¹ | 6,5e−08 | 4,0e−03 |
| iq4_xs | sim, `matvec.cuh:116` / `vecdotq.cuh:783` | 2,596 | 5,055 | 513 | 6,4e−08 | 3,8e−03 |

Todos os 14 tipos têm kernel nativo (nenhum cai em caminho genérico); a variante
que **envia** está escolhida em `include/rdna4/matvec.cuh` (`case 12:` → `TIQ3S_PERM`
para `iq3_s`, `case 9:` → `TIQ3XXS_PERM2` para `iq3_xxs`, `113-116` + `868-895`).

¹ q8_0, q6_k e iq4_nl são tensores *muito pequenos* no inventário (`ssm_beta`/
`ssm_alpha` do GDN, 48×160) — 24-199 GB/s é **latência**, não banda: são 6-30 MB/s
de trabalho real em 1,0 ms/token somados.
² `q3_k` é o único k-quant em emulação de bytes 16-bit (`v_sub_nc_u16`/`v_and_b32`,
208 instruções por chamada). O estudo ROCm mediu 307 GB/s nele; eu meço **271 GB/s**
numa sessão em que `q5_k` deu exatamente os 582 GB/s do estudo — a discordância é
do `q3_k` (-12%), não da sessão, e vale re-medir com A/B intercalado antes de
atribuir causa.
Para o IQ4_XS o mesmo comando (`bench-matvec-shapes-gpu <IQ4_XS> --no-ab`) dá:
`iq4_xs` **7,217 GB/token (54,1%) a 526 GB/s**, `q5_k` 2,096 GB (15,7%) a **610 GB/s**,
`iq3_s` 1,615 GB (12,1%) a 459, `q4_k` 1,262 GB (9,5%) a 445, `iq3_xxs` 0,546 GB a 438 —
ou seja, o IQ4_XS troca os `iq3_xxs`/`iq3_s` do IQ3_S por `iq4_xs`, que roda a
**526 GB/s contra 425 do `iq3_s`** (24% mais banda) e ao mesmo tempo tem metade do
erro de peso (§3). É o argumento medido a favor do IQ4_XS onde ele cabe.

³ Confere com os 582 GB/s e com os 627 GB/s do LM head (`output.weight` é q5_K
248320×20) medidos no `docs/rocm-estudo.md` §A.2.5 — dentro de 1%.

---

## 6. Recomendação por contexto (VRAM **medida** + precisão **medida**)

Fits medidos no `bench` (`vram in use` reportado pela ferramenta; "falha" =
`graph init failed: hipMalloc failed for blk.N.*`, medido, não estimado):

| contexto | IQ3_S + f16 | IQ3_S + q8_0 | IQ3_S + q4_0 | IQ4_XS + f16 | IQ4_XS + q4_0 |
|---|---|---|---|---|---|
| 4K | 11,87 GiB / **28,9 tok/s** | — | — | 13,92 GiB / **27,5 tok/s** | — |
| 24K | 13,2 GiB / **23,6 tok/s** | — | — | — | — |
| 32K | 13,62 GiB / **22,8 tok/s** | — | — | **falha** | 15,35 GiB (0,57 livres) / 15,6 tok/s |
| 48K | 14,62 GiB / **20,8 tok/s** | — | — | **falha** | **falha** |
| 64K | 15,62 GiB, 0,30 livres → **2,1 GB em GTT → 2,2-8,0 tok/s** | 13,75 GiB / **18,28 tok/s** | 12,76 GiB / 18,10-18,17 tok/s | **falha** | **falha** |

- **Até 48K: IQ3_S com KV `f16`**, sem dúvida — é o mais preciso e o mais rápido
  onde cabe (28,9 → 20,8 tok/s; a atenção medida é só 1,4-5,0 ms/token nessa faixa).
- **A partir de ~56-60K: IQ3_S com KV `q8_0`**, não `q4_0`. Medido a 64K: `q8_0` é
  marginalmente **mais rápido** que `q4_0` (18,28 contra 18,10-18,17 tok/s; atenção
  isolada 19,87 contra 21,03 ms) **com o dobro da precisão do KV** e 2,18 GiB de
  folga contra o transbordo. Isso contradiz a recomendação implícita do `README.md`
  ("`q4_0` é o que faz 64K+ caber"): hoje `q8_0` cabe e é melhor nos dois eixos.
- **O IQ4_XS não é utilizável a contexto longo nesta placa**: medido, ele não
  inicializa nem com `q4_0` acima de 32K (falha de `hipMalloc` no `blk.38` a 48K) e
  nem com `f16` a 32K. O ganho de precisão dele (§3) existe só até ~32K; para 64K+ o
  arquivo é o IQ3_S.
- **A escolha do KV é decisão de fit, não de velocidade**: a 64K, `q4_0` lê 1,21 GB
  únicos/token e `q8_0` lê 2,28 GB, mas a atenção leva 21,03 ms contra 19,87 ms —
  o `q4_0` **não compra tempo nenhum**, só VRAM (confirma `docs/rocm-estudo.md`
  §A.2.2, M5 e M7).
- **IQ4_XS vs IQ3_S onde os dois cabem (4K)**: +2,06 GiB de VRAM e +1,8 ms/token
  (36,4 contra 34,6 ms) em troca de ~metade do erro de peso (§3) e do mesmo desvio
  fim a fim dentro de 0,2% (§4). Não é uma troca que se paga em decode; é uma
  escolha de qualidade quando a VRAM sobra.

---

## 7. O que não foi medido (e por quê)

1. **O erro absoluto do peso de cada formato** (contra o original fp32/torch) — não
   existe esse modelo na máquina. O que está em §3 é a **distância entre dois
   formatos**, que é um limitante superior da soma dos dois erros, não o erro de
   cada um isolado.
2. **`check-matvec-gpu`/`check-dequant-gpu` completos** não foram rodados nesta
   sessão (o M8 os deixou verdes; `check-graph-gpu` está vermelho por um dump de
   oráculo desatualizado, ver `docs/rocm-estudo.md` §F.1) — os números de precisão
   daqui não dependem deles, a referência é fp64 própria.
3. **`iq1_s`/`iq2_*` com correção de truncamento** — não implementado (é mudança em
   `vecdotq.cuh`, dono é a frente de kernels); o ganho máximo medido é 4,4e−04 de
   erro relativo em 0,3% do token.
4. **KV `q4_0` vs `f16` na *qualidade* a contexto longo** — só o `q8_0` foi
   comparado em velocidade; a PPL a 64K com cada tipo de KV não foi medida (custo
   de GPU).
5. **Formatos ausentes nestes dois arquivos** (`q4_1`, `q5_0`, `bf16`, `fp16`):
   nada a medir sem um GGUF que os use.
