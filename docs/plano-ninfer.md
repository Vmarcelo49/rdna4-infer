# Plano: o que pegar do NInfer e como adaptar aqui

Fonte: `docs/estudo-ninfer.md` (958 linhas, 124 citações `file:line`, clone em `/tmp/ninfer`
HEAD `d492968`). Este plano **não** copia a lista de lá: cada item passa por três filtros antes
de virar trabalho — (1) o gargalo dele é o NOSSO gargalo? (2) ele já foi testado aqui? (3) qual
é a medição que decide, e quanto custa?

Regra da casa que vale para tudo: **gpu-lock com `timeout` por dentro**, uma GPU compartilhada,
gate antes de cronometrar, e número medido no lugar de adjetivo.

## 0. O que a lista do ninfer vira depois dos filtros

| item do ninfer | veredito aqui | por quê |
|---|---|---|
| **1. GQA fundido no decode** (1 CTA por cabeça de KV, 6 cabeças do grupo dentro) | **JÁ TESTADO E REFUTADO na nossa forma** | protótipo C: com a mesma grade `(4,S)`, cortar o tráfego de K/V em 6× deixou o kernel **1,25× (4K) e 1,82× (16K) MAIS LENTO** — 68 VGPR + 400 B de derrame no `HG=6` contra 56 VGPR/0 B no `HG=1` (`docs/journal-longctx.md` §4.1). A releitura 6× é absorvida pelo cache; o custo de servir 6 cabeças por CTA é maior que os bytes economizados. **Mas a estrutura do ninfer é outra** (SIMT, dot-product-major, T pequeno, 6 cabeças em registrador) → vira **pesquisa**, não implementação, e só depois um experimento novo se a estrutura diferir onde importa (§1 N1). |
| **2. Dequant LDS→registrador dentro do laço de consumo, nunca LDS→LDS** | **VIVO** — é o item de maior valor imediato | nosso `gemm_i8_kernel` faz global→reg→decodifica→**LDS int8**→reg (`include/rdna4/gemm.cuh:307-332` e `:384`): duas voltas pela LDS e a decodificação *antes* da barreira. O ninfer estagia os bytes crus com `cp_async<16>` e decodifica no laço de FMA (`q4_rowsplit_gemm_simt.cuh:77-110` stage, `:112-159` consume). Staging é 18-61 % do nosso kernel dependendo da variante (frentes F/G/H). |
| **3. `ColsPerTile ≤ 8` com `static_assert`, paralelismo em warps** | **VIVO e alinhado** | é exatamente a regra que faltou ao nosso D1: subir N levou de 62 para 133 VGPR e custou +71 % (`docs/plano-prefill.md` §0.1). O limiar deles (`T ≤ 16 → SIMT, T > 16 → tensor core`, `q4_dispatch.cpp:14-17`) é o mesmo do nosso despacho. |
| **4. GDN em blocos de 64 com três estágios** | **VIVO, e é o segundo maior** | ataca 0,852 ms/token de recorrência (`docs/estudo-prefill-c-nosso.md` §5) — o maior custo de andaime que sobra depois do matvec. A matemática (block-WY) **não foi validada termo a termo** pela frente do ninfer → vira pesquisa antes de virar código (§1 N2). |
| **5. Launcher por T exato com K dividido entre warps** | **VIVO, e é o item do MTP** | ataca o coeficiente `6,04 ms/token` de `pass_ms ≈ 13,9 + 6,04·N` e o custo marginal da verificação. |
| **honra: ReplaySSM** (registrar entradas e reexecutar em vez de snapshot+restore) | **VIVO, barato de medir** | contra os nossos **0,54 ms** por par snapshot+restore; a razão de tráfego de estado deles é 86,1×. |
| **honra: PDL** (28 sítios) | **ESTUDO primeiro** | contra os nossos ~1.940 lançamentos/token (~4 ms/token de despacho). **Não há equivalente HIP verificado** — é a primeira coisa a checar. |
| **honra: política de split em degraus, teto 85** | **VIVO, mas subordinado** | a nossa atenção a 64K custa 19,0 ms/token e o teto é 16 splits; a frente J mediu que a atenção é ≤2 % do prefill, então isto é **contexto longo**, não prefill. |
| **§3.13 drafter em bloco (DFlash2)** | **enquadramento, não port** | bloqueio de **artefato** (pesos que o GGUF não tem). O que passa a valer é a medição F0: *ms por draft proposto hoje* contra `1/K` de um forward. |

**O que o ninfer sugere e nós já refutamos por medição** (não repetir): fusão de kernels em massa
para o prefill (1,1 %), duplo buffer de ativação (−17 %), aumentar o tile em M para matar a cauda
de onda (−13 %), staging LDS da ativação (−5 %). Nada no ninfer contradiz: eles **não** fundem em
massa (usam PDL) e o duplo buffer deles é **só do peso** — igual ao nosso V6.

## 1. As frentes, com o que cada uma entrega

Ordem por (ganho esperado × risco) e com dependências explícitas. **Pesquisa e implementação
podem rodar em paralelo** desde que os arquivos não colidam.

### N1 — Pesquisa: a estrutura do GQA do ninfer difere da nossa onde importa? (sem GPU)
Ler `small_t.cu` / `small_t_bf16.cuh` / `small_t_k8v4.cuh` (grid `(KVHeads, splits, batch)`,
`kv_head = blockIdx.x`, `q_head = kv_head*GroupSize + local_q`, teto
`static_assert(TokenTile*GroupSize <= 48)`), comparar **estrutura por estrutura** com o nosso
protótipo C (`tests/bench_attn_gpu.hip:229` `attn_gqa_split_kernel`, e
`docs/journal-longctx.md` §4.1) e responder: a refutação daqui cobre a forma deles?
- Se cobrir: registrar o porquê com o número e **fechar o item** (economiza uma semana).
- Se não cobrir: entregar o desenho de um experimento novo (kernel, shapes, comando) com o
  número que ele moveria e o **gate numérico** (GQA muda a ordem das somas → não é bit-exato).
Entregável: `docs/estudo-ninfer-gqa.md`.

### N2 — Pesquisa: a matemática do GDN em blocos (sem GPU)
Ler `gated_delta_net/chunked/{prepare_wy_wu,state_passing,output}` do ninfer, escrever a
decomposição termo a termo (block-WY / chunked delta rule), comparar com o nosso
`delta_rule` (`include/rdna4/gdn.cuh:57-108`) e entregar uma **especificação de port**:
estágios, shapes, workspace, onde entra o estado, e o que muda na ordem das somas.
- Ganho alvo: os **0,852 ms/token** de recorrência (48 camadas), que no degrau D4 viram ~52 %
  do que sobra.
- Gate obrigatório: não é bit-exato (outra ordem) → `scripts/compare_ppl.sh` com tolerância
  declarada + `check-regression-gpu`, e o caminho antigo atrás de `RD_GDN_CHUNKED=0`.
Entregável: `docs/estudo-ninfer-gdn.md`.

### N3 — Implementação: tirar a volta dupla pela LDS do `gemm_i8_kernel` (GPU)
Reescrever o staging/consumo de `include/rdna4/gemm.cuh` no padrão do ninfer: estagiar os
**bytes crus** (não o int8 já decodificado) e mover a decodificação para dentro do laço de
consumo; a escala continua dobrada sobre o tile de colunas.
- Medir: T-MAC/s em M=64/128/512 contra o `gemm_i8_kernel` de hoje (**13,49 / 15,69 / 17,42**),
  mais a fração de staging (a frente G mediu 18,3 % no V6, a H 50-61 % no WMMA).
- **Gate: tem de continuar bit-exato** contra o `vec_dot` (o D2 é 0/4096, max ulp 0) — se a
  decodificação mudar de lugar e o resultado mudar, o ganho não vale a perda do contrato.
- Entregável: `gemm.cuh` editado + números; o relatório vai para mim, o doc sou eu que escrevo.

### N4 — Medição: o custo por draft do MTP e o custo do snapshot (GPU)
Duas medições baratas que decidem dois itens grandes:
1. **ms por draft proposto** hoje (o nosso MTP sorteia autoregressivamente como o deles) contra
   `1/K` de um forward — decide se o §3.13 (drafter em bloco) é a explicação do 0,96×, ou se o
   gargalo é só o matvec (e então N5/N6 bastam).
2. **tráfego do snapshot+restore** por par (0,54 ms) contra o que um "registrar entradas e
   reexecutar" moveria (a razão do ninfer é 86,1×) — dimensiona o ReplaySSM antes de escrevê-lo.
Entregável: números para mim; o doc sou eu que escrevo.

### N5 — Implementação: launcher small-T com K dividido entre warps (depois de N4)
Padrão do ninfer `q4_small_t_mma`: `kKWarps=8`, `kTileKPerWarp=64`, `kRowsPerCta=16`, redução na
LDS, instanciado para T=2/3 (a nossa janela de verificação do MTP). Alvo: o coeficiente
6,04 ms/token na parte que é do T pequeno.

### N6 — Estudo: existe equivalente HIP do PDL? (sem GPU, meia hora)
Checar `/opt/rocm/include` e a documentação do ROCm por algo como
`cudaGridDependencySynchronize`/`griddepcontrol` (PTX `griddepcontrol.wait/launch_dependents`).
Se existir, dimensionar o ganho contra ~4 ms/token de despacho; se não existir, **fechar** o item
com a evidência (e registrar como "não portável", não como "a fazer").

### N7 — Implementação: política de split em degraus (contexto longo, depois de N4)
`attn_splits_for` (`include/rdna4/graph.cuh:432-445`) hoje é `keys/kAttnSplitMin` com teto 16;
o ninfer usa 4 faixas (64/128/256/480 chaves por split) com teto 85. Ganho possível é de contexto
longo (a atenção é 19,0 ms/token a 64K), não de prefill. Gate: `scripts/check_attn_split.sh`
(escada de desvio já medida: 5,1989 → 5,2114, limite 0,5 %).

## 2. Ordem e paralelismo

```
agora      N1 (pesquisa, sem GPU)  ─┬─ decide se o GQA volta a ser trabalho
           N2 (pesquisa, sem GPU)  ─┼─ especifica o GDN em blocos
           N3 (impl, GPU)          ─┼─ maior ganho imediato no gemm
           N4 (medição, GPU)       ─┴─ decide o MTP e o ReplaySSM
depois     N5 (impl small-T) ← depende de N4
           N7 (impl splits)  ← independente, contexto longo
meia hora  N6 (estudo PDL)  ← decide se vale um protótipo
```

N1/N2/N6 não tocam a GPU nem o código (só criam os docs deles) → rodam livres. N3/N4 disputam a
GPU pelo lock (serializam, cada um com `timeout` dentro).

## 3. O que este plano se recusa a fazer

- **Copiar o item 1 sem passar pelo filtro**: a nossa própria medição já mostrou que compartilhar
  K/V naquela forma perde; repetir isso seria gastar GPU para reconfirmar um número que temos.
- **Começar pelo drafter em bloco** (§3.13): é bloqueio de artefato, não de kernel — e antes dele
  a medição F0 diz se ele é sequer a explicação.
- **Aceitar "86,1× menos tráfego de estado" como ganho nosso**: o número é deles, noutra máquina;
  o nosso é 0,54 ms por par e é isso que N4 mede.
- **Trocar bit-exatidão por velocidade sem declarar**: N3 mantém o gate bit-exato; N2 e N7 não são
  bit-exatos e por isso entram com PPL/regressão e com o caminho antigo atrás de `RD_*`.
