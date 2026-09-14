# A atenção de decode com GQA fundido do ninfer é a mesma coisa que nós já refutamos?

Leitura de código, **sem GPU** (a placa está compartilhada; nada aqui foi executado). Fonte:
clone raso de `https://github.com/Neroued/ninfer` em `/tmp/ninfer`, HEAD `d492968`. Todas as
referências ao ninfer foram lidas nesta sessão. Marcação: **[D]** = documentado por leitura de
código, **[M]** = medido (digo por quem, o comando está no doc citado), **DERIVADO** = aritmética
minha sobre medições nossas, **INFERIDO** = não verificado, com o que confirmaria.

Convenção de citação (os dois repos têm `README.md` e `docs/`): `nosso <path>:<linha>` = este
repo; `ninfer <path>:<linha>` = `/tmp/ninfer`. É a mesma convenção de `docs/estudo-ninfer.md`;
aqui os dois prefixos aparecem sempre, para não haver ambiguidade.

O que este documento liga: a leitura estrutural do `docs/estudo-ninfer.md` §3.5 (o que o kernel
deles faz) com a nossa medição de `docs/journal-longctx.md` §4 (o que a fusão de GQA vale aqui).
As duas coisas foram escritas em frentes diferentes e nunca foram confrontadas.

## TL;DR

1. **Não cobre o kernel deles.** O nosso parâmetro `HG` varia *quantas cabeças a CTA serve*;
   nunca varia **qual eixo as warps dividem**. O ninfer divide as **linhas** (T·GroupSize ≤ 48,
   `ninfer src/ops/softmax_attention/dense/causal_cache/small_t_k8v4.cuh:33,52`) com o K/V
   estagiado **uma vez** na LDS (`.../small_t_k8v4.cuh:294-341`) e o estado das linhas
   **particionado** pelos warps (o Q é relido da LDS a cada bloco de chaves, `:359-364`); o
   protótipo C mantém `qv[6][8]+acc[6][8]+m[6]+l[6] = 108 f32` **por lane, replicado em cada uma
   das 8 warps**, com um laço serial de 6 cabeças dentro do laço de chaves
   (`nosso tests/bench_attn_gpu.hip:249-285`). Os 68 VGPR + **400 B/lane de derrame** do HG=6 são
   da nossa organização, não da fusão — e o default deles (bf16-K/f16-V, **1024 B/chave**, §5a) é
   a mesma linha em que a fusão nos custou 1,25-1,82×, não uma linha estreita.
2. **Mas a parte da refutação que decide o valor já está medida**: no mesmo grid e **sem
   derrame** (HG=3 = 2 passadas, 125 VGPR, 0 B), cortar as leituras 3× vale **+5,0 % a 4K**
   (0,1358 → 0,1293 ms) e **0 % a 16K** (0,2652 → 0,2657 ms) — `nosso docs/journal-longctx.md`
   §4.1/§4.2. Ou seja: o que se ganha removendo a releitura é pequeno; o que se perde servindo
   mais cabeças por CTA é que é grande.
3. **A diferença que decide é a grade, não a fusão.** A 4K o kernel que embarca ganha 1,9× do
   HG=1 **com o mesmo tráfego** (0,0715 vs 0,1358 ms, §4.1) porque tem 192 CTAs contra 32
   (`nosso include/rdna4/attn.cuh:946-948`, `graph.cuh:444`). O ninfer compensa a grade estreita
   (4 CTAs por split) com **64-85 splits** → 256-340 CTAs (`ninfer .../small_t.cu:38-41`,
   `small_t.cuh:122-130`). **Essa célula — protótipo C com S=48/64 a 4K/16K — nunca foi medida.**
4. **O que a fusão ainda poderia valer, e não está refutado**: a 16K f16 o motor gasta
   **4,49 ms/token** de atenção = **12,1 % do passo** emitindo **1430 GB/s**; a 64K f16,
   **16,5 ms = 33,6 %** emitindo **1561 GB/s** — em cima do pico de Infinity Cache medido
   independentemente (**1499,6 GB/s**, `nosso docs/medicoes-banda-e-gargalos.md:272-273,278`).
   As duas medições que variam o tráfego estão uma confundida (HG=1 ↔ HG=6, com o derrame de
   400 B/lane) e a outra fora do regime (HG=1 ↔ HG=3: +5 % a 4K e 0 % a 16K, medida com **32
   CTAs**, contra os 192-384 CTAs do kernel que embarca).
5. **Veredito**: a refutação fecha **a forma do protótipo C**, não a do ninfer. Fechar o item com
   razão custa **uma célula** (E0, binário que já existe, sem linha nova); se ela perder, o que
   resta do ninfer é outro projeto — atenção *small-T* de tile de linhas para verify/MTP — e não
   "fusão de GQA".

---

## 0. O que o ninfer faz, em números [D]

| | ninfer (decode/verify `small_t`) | nosso decode que embarca |
|---|---|---|
| grade | `(KVHeads=4, splits, batch)`, `kv_head = blockIdx.x` (`ninfer src/ops/softmax_attention/dense/causal_cache/small_t.cu:102,136`) | `(n_head=24, splits)` (`nosso include/rdna4/attn.cuh:193,652`) |
| splits | degraus de 64/128/256/480 chaves; teto **85** (`ninfer .../small_t.cuh:81-94`, `geometry.cuh:11-12`) | `chaves/512`, teto **16** (`nosso graph.cuh:443-445`) |
| CTAs a 4K / 16K / 64K | **256 / 256 / 340** (4×64, 4×64, 4×85; `small_t.cu:38-41`, `small_t.cuh:122-130`) | **192 / 384 / 384** (24×8 com 8 warps; 24×16 com 16 warps; `attn.cuh:946-948`) |
| linhas de consulta por CTA | `TokenTile·GroupSize` = **6 … 48** (`small_t_k8v4.cuh:33,52`) | **1** (`attn.cuh:100`) |
| threads/CTA, LDS/CTA | 256, 28 KiB dinâmico (`small_t_k8v4.cu:21,24,51`) + ~7 KiB estático DERIVADO de `small_t_k8v4.cuh:58-72` | 256-512, 8,2 KiB (`nosso attn.cuh:192`) |
| unidade de cálculo | `mma` m16n8k16 / m16n8k32 (`ninfer src/ops/common/mma.cuh:36,45,53,62`) | `fma` fp32 + **5 `shfl_xor` por (chave, cabeça)** (`attn.cuh:131-133`) |
| estado por lane no T=1 | **particionado**: `acc[4][4]` = 16 f32 no k8v4 (`small_t_k8v4.cuh:283-288`) ou 128+64+16 f32 no bf16, por warp e por tile de linhas (`small_t_bf16.cuh:196,199-214,265`) | **replicado**: `qv[8]+acc[8]+m+l` = 18 f32 **por cabeça, em cada warp** (`attn.cuh:113-119`); HG=6 = 108 f32/lane → spill medido |
| tráfego de K/V | **1× os bytes únicos**, por construção (`ninfer small_t_k8v4.cuh:3-4,294-341`) | **6×** (uma vez por cabeça de consulta, `nosso README.md:510-511`) |
| bytes por chave (K+V) | ~**402 B** no k8v4 (fp8 1 B/elem + escala por linha D256; V nvfp4, grupo 16 → 0,5 B/elem + 16 escalas; `ninfer kv_cache/fp8_e4m3_row_codec.cuh:3,45-58`, `nvfp4_group16_codec.cuh:17-18`), mas **1024 B no default bf16** (§5a) — DERIVADO das constantes | f16 **1024 B**; q4_0 **288 B** (`nosso kv.h:144`: 18/32 B por elemento × 2 linhas) |

`GroupSize` é 6 na geometria que interessa (`CausalD256H24Kv4`, `ninfer geometry.cuh:15` +
`head_mapping.cuh:12-14`) — **a nossa mesma razão 24/4**. O `GroupSize` é 8 na outra geometria que
existe (`CausalD256H16Kv2`, `geometry.cuh:16`): **não existe no repo deles nenhuma geometria de
GroupSize 1**, ou seja, a fusão de GQA não é um parâmetro que eles tenham A/B — é a forma do
kernel [D].

Uma nota sobre a métrica deles: o bench de atenção do ninfer conta os bytes de cache como
`causal_key_sum(...) × geometry.kv_heads × (key+value)` (`ninfer bench/ops/causal_softmax_attention_bench.cu:645-646`),
isto é **bytes únicos**, não emitidos. A "banda atingida" que o bench deles reporta é, por
construção, o ideal de 1× — ela não pode falsear a afirmação do comentário de projeto.

---

## 1. Onde as 6 cabeças moram

**(a) ninfer — as 6 cabeças são *linhas* de um tile.** `RowCount = TokenTile·GroupSize`
(`ninfer .../small_t_k8v4.cuh:33`) e `RowTiles = ceil(RowCount/16)` (`:34`), com o teto
`static_assert(TokenTile·GroupSize <= 48)` (`:52`). No T=1 isso dá `RowCount = 6`, `RowTiles = 1`,
`Br = 16` (`:35`), `Warps = 8` (`small_t_k8v4.cu:21`). O mapeamento linha→(cabeça, token) é
`q_head = kv_head·GroupSize + local_q` (`ninfer .../small_t.cuh:157-163`).

**(b) ninfer — o estado das linhas é *particionado* pelos warps, não replicado.** No k8v4 do T=1,
`RowTiles = 1`, `Wc = 8` e `ConsumerWarpsPerTile = Wc/RowTiles = 8`, então o PV é dividido por
**d**: `PVNtPerWarp = D/(8·8) = 4` e cada warp guarda só `acc[4][4]` = **16 f32**
(`ninfer .../small_t_k8v4.cuh:43-44,283-288`; consumo em `:495-527`). O **Q nem fica residente**: o
fragmento `af[4]` é relido da LDS por bloco de chaves, dentro do laço (`:359-364`), e o K idem
(`:366-373`). O estado da linha (m, l) são 4 escalares por warp (`:289-292`), não 6 vetores.

No caminho **bf16** (o default do CLI, ver §5a) o estado é maior, mas continua **particionado por
linha**: cada warp é dono de 16 linhas (`warp_row0 = warp*16`, `small_t_bf16.cuh:196`) e guarda
`acc[PVNt][4]` com `PVNt = D/8 = 32` → 128 f32, mais `af_q[QKKs][4]` = 64 f32 de Q residente
(`:36-37,199-214`) e `score[4][4]` = 16 f32 (`:265`). Nada disso é replicado entre warps.

**(c) nosso protótipo C — as 6 cabeças são estado por lane, dentro do laço de chaves.**
`float qv[HG][8]; float acc[HG][8]; float m[HG], l[HG];` carregados **antes** do laço
(`nosso tests/bench_attn_gpu.hip:252-261`) e o laço de chaves (`:263`) tem, **dentro**, um laço
serial sobre as cabeças (`:269-285`): por chave, 6 × (8 fma + 5 `shfl_xor` + `expf` + 8 fma). O
comentário do próprio protótipo diz de onde vem o custo: *"Registers are the reason HG is a knob:
q and acc are HG·8 floats per lane each (head_dim 256 -> dpw 8). HG=6 costs ~110 registers per
lane"* (`tests/bench_attn_gpu.hip:220-221`) — e foi o que a medição achou: **68 VGPR + 400 B de
derrame no HG=6 contra 56 VGPR e 0 B no HG=1** (`nosso docs/journal-longctx.md:379-382`).

**(d) A diferença, então, não é "6 cabeças por CTA" (os dois têm): é *partição contra
replicação*.** Na forma deles as 6 linhas ocupam **um** tile de 16 linhas, com o acumulador
particionado pelas 8 warps em d e o Q relido da LDS a cada bloco (16 f32/lane no total,
`:283-288`); na nossa, **cada uma das 8 warps carrega o estado das 6 cabeças**, porque cada warp
anda por uma fatia de chaves diferente e precisa da linha inteira para si: `108 f32/lane × 8 warps
× 32 lanes ≈ 27,6 mil f32` de estado de linha por CTA para 6 linhas de 256 d (1 536 f32 de carga
útil) — **18× de replicação** — e o resultado medido foi 68 VGPR + 400 B de derrame
(`nosso docs/journal-longctx.md:379-382`). O derrame que matou o HG=6 **não tem onde ocorrer** na
forma deles. [D] quanto à estrutura; o número de VGPR deles é INFERIDO (ver §7).

**Consequência para a refutação: o item 1 dela ("servir 6 cabeças por CTA custa mais que os
bytes economizados") é um resultado sobre a *nossa* forma. Ele não se transporta.**

## 2. O que paraleliza o passo sobre as chaves

**(a) ninfer — as chaves não são divididas entre as warps da CTA.** Dentro do CTA, o laço de
blocos de chaves é **serial** (`small_t_k8v4.cuh:348`, `for (int kb = 0; kb < key_blocks; ++kb)`) e
a warp produtora é **uma só** no T=1 (`if (warp < RowTiles)` com `RowTiles = 1`, `:350`); as outras
7 warps fazem a dequantização de V para a LDS (`:465-483`) e voltam para o PV (`:495-527`). O
paralelismo sobre chaves vem de dois outros lugares: (i) o eixo `splits` da grade — 64 a 85 fatias
por cabeça de KV (`small_t.cu:38-41`) e (ii) dentro do bloco de 32 chaves, que é a dimensão **N**
do mma, espalhada pelas lanes. O comentário do launcher assume isso: o alvo de splits é
"manter a grade completa em uma ou duas ondas de 170 SMs" (`small_t.cu:229-235`).

**(b) nosso — as chaves são o eixo que as warps dividem.** `for (int j = w; j <= t; j += WPB)`
(`nosso include/rdna4/attn.cuh:121`; idem no batelado `:348` e no com split `:521`): 8 (ou 16)
warps por CTA, cada uma com a sua fatia de chaves, uma cabeça de consulta por CTA. O paralelismo
sobre chaves vem de `n_head × WPB × splits`.

**(c) Por que isso importa para a pergunta.** As duas organizações pagam por coisas diferentes:
quem divide **chaves** precisa carregar, em cada lane, o estado de **todas** as cabeças que a CTA
serve (é por isso que `HG` custa registrador); quem divide **linhas** paga em linhas de tile e em
leitura de LDS (o Q e o K são relidos da LDS a cada bloco — barato, mas é instrução). O `HG` do
protótipo C mexe no *quanto* do primeiro eixo, nunca no eixo. **É exatamente o eixo que decide, e
é o que não foi testado.**

**(d) O efeito colateral que o ninfer resolve e nós não**: com 4 CTAs por split, a grade fica
estreita e ele compensa com 64-85 splits; a nossa política de splits é `chaves/512` com teto 16
(`nosso graph.cuh:443-445`) e roda a 8-16 splits **porque** tem 24 CTAs por split. A 4K: 192 CTAs
nossos (8 splits × 24 cabeças), 256 deles (64 splits × 4 KV). A 16K: 384 nossos, 256 deles. As
contagens de CTA são *da mesma ordem* — o que não é da mesma ordem é a contagem de CTA **do
protótipo C** (32 a 4K, 64 a 16K com S=8/16, `journal-longctx.md:366,418-423`).

**Ressalva de porte, DERIVADO**: a política de splits deles é dimensionada para *uma onda* de uma
placa de **170 SMs** com 2 CTAs por SM: o comentário do launcher diz "manter a grade completa em
uma ou duas ondas de 170 SMs" (`ninfer small_t.cu:229-235`) e o kernel bf16 declara
`__launch_bounds__(128, 2)` (`small_t_bf16.cuh:21`), isto é 160-320 CTAs alvo. A nossa placa tem
**32 WGP = 64 CU** (`nosso docs/rdna4-gfx1201-hardware-brief.md:9`, medido em
`journal-longctx.md:372-373`). "Mesma contagem de CTA" entre os dois desenhos **não é mesma
ocupação**; é mais um motivo para a célula de E0 ser medida, e não deduzida.

## 3. O regime de T (e o teto de 48 linhas)

**(a) A fusão de GQA e o tile de tokens competem pelo mesmo orçamento de linhas.** O
`static_assert(TokenTile·GroupSize <= 48)` (`ninfer small_t_k8v4.cuh:52`, idem `small_t_bf16.cuh:27`)
é o teto dos 3 tiles de 16 linhas do mma. Com `GroupSize = 6`, **T ≤ 8**. Ou seja: no desenho
deles, fundir GQA *tira* capacidade de T. Se não fundissem (uma cabeça por CTA, como no prefill
deles), o mesmo teto de linhas permitiria T ≤ 48.

**(b) No T=1 a fusão deles é quase de graça — e o motivo é o tile de 16 linhas.** Com
`RowCount = 6` e a linha do mma com 16, **10 das 16 linhas do tile são preenchimento**: 62 % de
trabalho de mma desperdiçado. Sem fusão (1 cabeça por CTA) seriam 15 de 16, 94 % desperdiçado.
A fusão de GQA, ali, não é um trade-off: ela **enche linhas de um tile que já estava pago**, e o
K/V já está na LDS de qualquer jeito. [D] quanto à estrutura; a leitura "é de graça" é INFERIDO —
confirmaria com um ablation de `GroupSize ∈ {1,6}` na mesma geometria deles (não existe no repo).

**(c) E no T>1 a amortização vem de T, não de GQA.** Com T=4 e `GroupSize = 6`, `RowCount = 24`
(2 tiles de 16); com T=8, `RowCount = 48` (3 tiles). O mesmo tile de K/V na LDS passa a servir 24
ou 48 linhas em vez de 6 — **4× a 8× menos bytes de K/V por linha de consulta**, que é mais do que
a fusão de GQA dá (6× no total, dos quais o T já é dono). O launcher tem `TokenTile` exato de 1 a
8 (`small_t.cu:296-329` para o bf16, `small_t_k8v4.cu:121-154` para o k8v4), com `Warps = 12`
quando `RowTiles = 3` e `KeyBlock = 64` para T ≥ 2 (`small_t_k8v4.cu:21-22`).

**(d) O nosso lado.** O decode é T=1 (`nosso attn.cuh:95-100`, uma cabeça por CTA, um token); o
verify do MTP é T=2..4 e usa `attn_batch_kernel` com **grade `(n_head, n_tok)`**
(`nosso attn.cuh:416`), isto é, **uma CTA por (cabeça, token) — zero compartilhamento de linha de
K/V**. A 16K isso é `24 × 4 = 96` CTAs relendo o mesmo KV 96 vezes por camada. [D]

**(e) Um detalhe que confirma a leitura de que o default deles mira T ≥ 2.** No caminho bf16 o
`WarpsPerCta` do T=1 é **2** (`ninfer small_t.cu:298`), o que dá `Br = 2·16 = 32` linhas para
`row_count = 6` (`small_t_bf16.cuh:31,65,109-110`): a warp 1 (linhas 16-31) é **inteiramente
preenchimento** e as linhas 6-15 da warp 0 também. O caminho otimizado para T=1 é o k8v4, cujo
`RowTiles` acompanha `RowCount` (`small_t_k8v4.cuh:34,350`, um tile só, 6 linhas reais de 16).
[D]

**Consequência: parte do que o `docs/estudo-ninfer.md` §3.5 atribui à fusão de GQA é, no repo
deles, amortização por T — e é a parte que o nosso motor poderia usar hoje (MTP roda com 4 linhas,
`nosso docs/mtp.md:3-6`), sem fusão nenhuma.**

## 4. "Dot-product major" vs flash-softmax: a premissa do briefing está errada

**(a) O `small_t` deles *é* online softmax com rescala**, não um kernel de "linha de score em
registrador e depois softmax": há `m`/`l` correntes, `alpha = exp2((m_old − m_new)·log2e)` e rescala
do acumulador a cada bloco de chaves (`ninfer small_t_k8v4.cuh:289-292`, `:423-428`, `:502-507`;
idem no bf16 `ninfer small_t_bf16.cuh:215,318-321,350-360`). A diferença não é online-vs-Não-online.

**(b) A diferença real é a *redução da linha*.** No mma m16n8k16/k32 a soma sobre d acontece
dentro da unidade de matriz (`ninfer mma.cuh:36,45,53,62`) e a linha é reduzida em **4 lanes**
(`gid = lane >> 2`, `warp_max<4>` em `warp.cuh:31-38`, usado em `small_t_k8v4.cuh:423-424`): 2
`shfl` por par de linhas por bloco de 32 chaves. No nosso kernel a redução é **de warp inteiro, 5
`shfl_xor`, por (chave, cabeça)** (`nosso attn.cuh:131-133`; protótipo C `:271-274` dentro do laço
de cabeças). Por chave e por cabeça servida, a nossa forma paga a redução 32-lanes inteira;
a deles paga 2 `shfl` por 32 chaves. É o segundo custo que a fusão de GQA multiplica por 6 na
nossa forma e não na deles.

**(c) O que isso muda no custo da fusão**: nada no *tráfego*; muito no custo *por cabeça*. Como o
nosso laço de cabeças é serial e a redução é de warp, servir 6 cabeças por CTA na nossa forma
custa 6 × (dot+redução+`expf`) **na mesma lane**, enquanto na deles as 6 cabeças são 6 das 16
linhas do mma, resolvidas por lanes diferentes. O "custo de servir 6 cabeças" que
`journal-longctx.md` §4.1 mediu **não é uma constante física da fusão**; é uma propriedade de
quem reduz com `shfl` de 32 lanes e guarda o estado na lane.

## 5. Precisão e formato

**(a) O default deles tem a *nossa* linha, não uma linha estreita.** O CLI deles usa
`KvCacheStorage::BFloat16` por default (`ninfer apps/cli/options.h:27`; as opções são
bf16/int8/fp8/nvfp4/k8v4, `apps/cli/options.cpp:56-60`) e esse caminho guarda **K em bf16 e V em
f16** (`ninfer small_t_bf16.cuh:3`; "K is copied exactly and V is rounded once to FP16", `:150,164-166`):
512 B + 512 B = **1024 B por chave** com head_dim 256 — **exatamente o nosso f16**
(`nosso kv.h:142`). O caminho fp8-K/NVFP4-V (`k8v4`) é *opção*, e aí sim a linha é estreita:
fp8-E4M3 (1 B/elemento + escala por linha de 256) para K e NVFP4 grupo-16 (0,5 B/elemento + 16
escalas) para V (`ninfer kv_cache/fp8_e4m3_row_codec.cuh:3,45-58`,
`ninfer kv_cache/nvfp4_group16_codec.cuh:17-18`) → ~402 B por chave (K+V), DERIVADO das
constantes.

**(b) A consequência para a pergunta é o contrário do que se poderia esperar.** A fusão não é um
truque que só paga com KV quantizado: o caminho default deles funde GQA **com a mesma linha de
1024 B que nós medimos como o caso onde a fusão perde 1,25-1,82×** (`nosso docs/journal-longctx.md`
§4.1: 0,1358 → 0,1694 a 4K e 0,2652 → 0,4834 a 16K). A nossa medição com q4_0 (144 B de K por
chave, ganho de 9,5-11,8 % a 64K/131K, `journal-longctx.md:484-491`) mostra *outro* regime, com
linha estreita — não é o regime do default deles. Ou seja: **o formato não explica a diferença de
desenho entre os dois kernels; a organização explica.**

**(c) Precisão do score.** Os scores deles nascem de um mma bf16/fp8 com correção de escala em
fp32 (`ninfer small_t_k8v4.cuh:377-388`); os nossos são dot fp32 SIMT com `expf` fp32. Um port
fiel não pode trocar a nossa aritmética de score sem trocar o gate numérico: a tolerância do
`check_attn_split.sh` é de 0,5 % de PPL em texto real (`nosso scripts/check_attn_split.sh`,
escada medida 5,1989 → 5,2054 → 5,2114, `nosso README.md:516-518`).

## 6. O contraste do prefill deles (evidência sobre quando a fusão paga)

**[D]** O prefill do ninfer **não funde nada**: `q_block = blockIdx.x`, `q_head = blockIdx.y`,
`kv_head = q_head / Geometry::GroupSize` (`ninfer .../prompt_bf16.cuh:97-103`) e a grade é
`(ceil(tokens/64), QHeads, 1)` (`ninfer .../prompt.cu:49-51`) — **uma cabeça de consulta por CTA**,
com `Br = 64` linhas de consulta e `Bc = 64` chaves por tile (`prompt_bf16.cuh:6`). Cada CTA de
prefill relê o mesmo K/V 6× (uma vez por cabeça do grupo), como nós.

**Por que faz sentido**: com 64 linhas de consulta por tile, o K/V já é amortizado 64×
*independentemente de GQA*, e o grupo de 6 não muda a ordem de grandeza. Onde o tile de linhas é
grande, a fusão de GQA deles é dispensável; onde ele é pequeno (T=1 → 6 linhas), ela é a única
forma de encher o tile. **O mesmo repo, com a mesma cabeça de atenção, escolhe fundir só no
small-T** [D]. É a evidência mais forte de que o que o `small_t` vende é o *tile de linhas*, e não
a fusão de GQA em si.

---

## Veredito

**Não — a nossa refutação não cobre a forma do ninfer.** Ela cobre, isso sim, e com número, o
prêmio que a fusão tem para dar aqui. Separando as duas coisas:

1. **Como kernel, não cobre.** O `HG` do protótipo C varia o número de cabeças *servidas* por CTA,
   mantendo fixo o eixo que as warps dividem (chaves) — e é esse eixo que decide onde o estado das
   6 cabeças mora. Na forma deles as 6 cabeças são linhas de um mma de 16 linhas, com Q e K
   relidos da LDS por bloco (`ninfer small_t_k8v4.cuh:359-373`) e só 16 f32/lane residentes
   (`:283-288`); na nossa, 108 f32/lane e 400 B de derrame. **O derrame que a refutação mediu é da
   nossa organização.**

2. **O item "o custo de servir 6 cabeças por CTA é maior que os bytes economizados"
   (`journal-longctx.md:395-400`) fica de pé como afirmação sobre a forma C, e não como lei.** A
   medição limpa (mesmo grid, sem derrame: HG=3 contra HG=1) dá **+5 % a 4K e 0 % a 16K** para 3×
   menos leituras. Isso diz que o ganho do lado das leituras é pequeno *naquele regime*; não diz
   que seja zero no regime em que o kernel que embarca roda (192-384 CTAs, 1430-1561 GB/s
   emitidos, isto é em cima do teto de cache medido em 1499,6 GB/s,
   `nosso docs/medicoes-banda-e-gargalos.md:272-273`).

3. **A conclusão de §4.2 ("não implementar para 4K-16K, perde 1,1-1,8×") tem um confundidor que a
   própria §4.2 nomeia e não mede:** a 4K/16K o protótipo C rodou com **S=8** → 32 CTAs, contra
   192-384 do kernel que embarca. O que está em jogo nessa comparação é CTA, e o kernel que
   embarca ganha 1,9× do HG=1 **com o mesmo tráfego** (`§4.1`: 0,0715 vs 0,1358). O ninfer
   resolve exatamente isso com 64-85 splits — a célula equivalente no nosso instrumento
   (protótipo C com S=48/64 a 4K/16K) **não existe em nenhuma tabela nossa**.

4. **O que fecharia o item com razão** (uma das duas):
   - **E0 (§ abaixo) mostrar que o kernel que embarca continua ganhando a 4K/16K com a contagem de
     CTA igualada.** Aí a fusão, *como fusão*, está refutada para onde o motor vive: com 4 CTAs por
     split o máximo que ela tem para vender já é a grade, e nós já temos a grade mais larga.
   - **E1 mostrar que o tile de linhas (sem derrame) ganha.** Aí não é a fusão do protótipo C que
     ressuscita: é outro kernel, e o item a fechar é outro (atenção small-T com tile de linhas,
     útil para o verify/MTP com T = 2..4, `nosso docs/mtp.md`).

5. **O que teria de ser verdade para a refutação ficar *completa* sem E0/E1**: que o custo por
   (chave, cabeça) fosse dominado pelos bytes. Não é o que os nossos números mostram — no **mesmo
   kernel e no mesmo contexto**, com 3,6× menos bytes emitidos (f16 → q4_0 a 64K) a atenção fica
   **1,29× mais lenta** (`journal-longctx.md:475-482`), e no mesmo grid, com 6× mais bytes (HG=1
   contra HG=6), fica **1,25× mais rápida**. Os dois apontam para o mesmo lado: o que domina é o
   trabalho por visita (dot + redução + `expf`), não os bytes. **Isso corta contra a fusão** — mas
   é também o motivo pelo qual a fusão continua **não medida** no regime certo, porque essa
   conclusão vale para um kernel cujo gargalo é outro.

Fechar hoje seria fechar com a razão errada ("os bytes não importam") quando a razão certa
disponível é mais estreita ("na forma C, servir mais cabeças por CTA custa mais do que as leituras
economizadas, e o ganho do lado das leituras é de 5 % a 4K e 0 % a 16K **nessa forma**").

---

## Experimento proposto

Três etapas, em ordem de custo crescente. A primeira usa binário que já está em `build/`.

### E0 — a célula que falta (0 linha de código, 1 janela de GPU, ~15 min)

Iguala a contagem de CTA entre o protótipo C e o kernel que embarca a 4K/16K, que é o confundidor
de §4.2. `--smax` levanta o teto de splits do buffer de parciais
(`nosso tests/bench_attn_gpu.hip:385,406-407,530`) e `--gqa-splits` permite forçar o S do
protótipo independente do kernel que embarca (`:424,538`).

```
timeout 900 ./scripts/gpu-lock.sh ./build/bench-attn-gpu f16 4096 16384 \
  --splits 8,16,24,48,64 --smax 64 --wpb 8 --gqa-hg 1,3,6
# e a célula reversa (protótipo com S=48/64, embarcado no S da política):
timeout 900 ./scripts/gpu-lock.sh ./build/bench-attn-gpu f16 4096 16384 \
  --splits 8,16 --smax 64 --gqa-splits 48 --wpb 8 --gqa-hg 3
```

- **Número a mover**: `ms` por camada do protótipo C a 4K/16K nas linhas S=48 (→192 CTAs) e S=64
  (→256 CTAs). Regra de decisão, escrita antes de rodar: **o item fecha** se, com CTA igualada, o
  melhor HG continuar acima do kernel que embarca (0,0715 a 4K; 0,2429 a 16K) pelo piso de ruído
  (1,000×/0,995×, `journal-longctx.md:369-370`); **o item reabre** se o HG=3 com S=48/64 cair
  abaixo disso — porque aí passam a existir dois efeitos somados (grade e fusão) que só o
  protótipo D (E1) separa.
- **Custo**: uma janela, sem escrever código. É a medida mais informativa por minuto de GPU que
  está pendente nesta frente.

### E1 — protótipo D: tile de linhas (o eixo do ninfer, na nossa aritmética) — ~200 linhas em `tests/bench_attn_gpu.hip`

É a versão mínima que testa **o único eixo que o `HG` nunca variou**. Esboço:

- **Grade** `(n_head_kv, S)`, igual ao protótipo C (`tests/bench_attn_gpu.hip:338`); **bloco**
  `WPB·32` com **WPB = 6** (T=1): **uma warp por cabeça do grupo** — nenhuma warp serve duas
  cabeças, nenhuma cabeça é dividida entre warps.
- **LDS**: `K_tile[Bc][head_dim]` + `V_tile[Bc][head_dim]` no formato já decodificado (f16):
  `Bc = 32` → 2 × 16 KiB = 32 KiB; duplo buffer = 64 KiB = exatamente o teto por workgroup desta
  placa (`nosso docs/rdna4-gfx1201-hardware-brief.md:10`, `docs/rocm-estudo.md:213`). Variante
  `Bc = 64` em buffer único (64 KiB).
- **Estágio**: as 6 warps copiam o tile da global cooperativamente (`kv_load8`/`kv_load`, `kv.h`),
  `__syncthreads()` por tile. O tile é buscado **uma vez**; as 6 warps o leem da LDS.
- **Consumidor**: warp `h` percorre as `Bc` chaves do tile na LDS com **exatamente o miolo do
  kernel que embarca** (`attn.cuh:121-134`: 8 fma + 5 `shfl_xor` + online softmax + 8 fma sobre V),
  mantendo `qv[8]+acc[8]+m+l = 18 f32/lane` → **56 VGPR e 0 B de derrame por construção**, igual ao
  HG=1 (`journal-longctx.md:382`).
- **Saída**: cada warp escreve o parcial da **sua própria cabeça** no layout `[h][S][2+head_dim]`
  que já existe (`bench_attn_gpu.hip:309-325`), então o `attn_merge_kernel` que embarca
  (`nosso attn.cuh:613`) combina sem mudança — e a **fusão intra-CTA desaparece**: não há redução
  entre warps, nem as `__syncthreads` de merge por cabeça do protótipo C (`:290-308`).
- **Variantes de 20 linhas cada, se a primeira célula empatar**: (a) *dequant-on-read* para
  q4_0/q5_0/q8_0 — guardar a linha **quantizada** na LDS e decodificar por warp na leitura, o que
  corta o tráfego de LDS de 6× para 6×·(bytes da linha quantizada); (b) 2 cabeças por warp (3
  warps) — metade do tráfego de LDS; (c) `Bc ∈ {16,32,64}`.
- **Formas**: `T = 1` (decode, este experimento); contextos **4 096 / 16 384 / 65 536**; KV
  **f16, q5_0, q4_1, q8_0** (os dois default e a referência); `S ∈ {8,16,24,48,64}` — o S tem de ser
  varrido porque o CTA do protótipo D é mais magro que o que embarca.
- **Comando**:
  ```
  timeout 900 ./scripts/gpu-lock.sh ./build/bench-attn-gpu f16 4096 16384 65536 \
    --splits 8,16,24,48,64 --smax 64 --gqa-row 1 --gqa-bc 16,32,64
  timeout 900 ./scripts/gpu-lock.sh ./build/bench-attn-gpu q5_0 16384 65536 \
    --splits 16,24,48 --smax 64 --gqa-row 1 --gqa-bc 32 --gqa-dequant-on-read
  ```
  (as flags `--gqa-row/--gqa-bc/--gqa-dequant-on-read` são novas; o resto do CLI não muda,
  `bench_attn_gpu.hip:17-18`).
- **Números a mover**, com o alvo escrito antes de rodar. Critério: **≥1,02×** sobre o baseline
  (acima de qualquer piso de ruído medido nesta frente, 0,995-1,000×, `journal-longctx.md:369-370`):

  | contexto / KV | baseline que embarca (ms/camada) | ms/token (×16) | alvo de E1 (≥1,02×) |
  |---|---|---|---|
  | 4 096 f16 | 0,0723 (`§3.1`) | 1,16 | **≤ 0,0709** |
  | 16 384 f16 | 0,2429 (`§3.1`) | 3,89 | **≤ 0,2381** |
  | 65 536 f16 | 0,9671 (`§3.1`) | 15,47 | ≤ 0,9405 (contra a melhor célula medida, 0,9593) |
  | 65 536 q4_0 | 1,2428 (`§4.3`) | 2,10 | ≤ 1,0897 (contra a célula do HG=3 = 1,1115, que **já** ganha 11,8 % sem kernel novo) |

  Banda efetiva a reportar nos dois sentidos: **emitida** (bytes emitidos ÷ tempo) e **única**
  (bytes únicos ÷ tempo). Hoje, a 16K f16, o motor gasta 4,487 ms/token de atenção emitindo
  **1430 GB/s** (`journal-longctx.md` §3.2) para 1,07 GB/token de bytes únicos = **~240 GB/s**
  (DERIVADO); a assinatura de que a fusão funcionou é a banda emitida cair para perto da única.
  VGPR/derrame de cada instanciação saem do `hipFuncGetAttributes` que o bench já
  imprime (`bench_attn_gpu.hip:461-464`) — a célula que derramar explica a própria lentidão.
- **Gate numérico**: o protótipo D **não é bit-exato** contra o caminho que embarca — a partição
  (cabeça, fatia de chaves) muda (uma warp por cabeça, chaves sequenciais), então a ordem de soma
  muda. Gates, na ordem em que valem:
  1. **`rel-L2` contra o kernel sem split, dentro do bench** (`bench_attn_gpu.hip:809-811`), com o
     teto sendo o desvio que o **próprio caminho com splits que embarca** já apresenta naquele
     contexto: **3,24e-07 a 4K e 6,45e-07 a 16K** (`journal-longctx.md:390-394`). Não é para ser
     melhor que isso, é para não ser pior.
  2. **`RD_ATTN_SPLITS=4 ./scripts/check_attn_split.sh`** — PPL em texto real, limite **0,5 %**
     (`nosso scripts/check_attn_split.sh`); é o gate que decide se o caminho pode embarcar
     (`nosso README.md:516-518`).
  3. **`check-kvctx-gpu`** para cobertura de formato/geometria — **não** como gate numérico: o
     cabeçalho do próprio script explica que com chave sintética o resultado é uma média quase
     cancelante e uma diferença de 1e-7 vira nível de porcento
     (`nosso scripts/check_attn_split.sh:5-12`).
  4. Se o caminho de verify (T>1) for tocado: `check_batch_gpu` é **bit-exato hoje** e deixaria de
     ser; a exigência do MTP é saída byte-idêntica em ganancioso (`nosso docs/mtp.md`), o que
     **não** é compatível com mudar a partição de chaves. Ou seja: E1 no decode não contamina o
     verify, mas E2 tem esse custo explícito.
- **Custo**: um protótipo no bench (não no motor), ~200 linhas, mais uma janela de GPU de
  45-60 min (as células são poucas e curtas; a parte longa é a varredura de S × Bc × KV). Não
  toca `include/rdna4/`, `CMakeLists.txt` nem a política — nada disso se justifica antes de E1
  ganhar. **Ressalva de expectativa, com número nosso**: o custo do *estágio* é o risco desta
  forma — num minigemm WMMA medido aqui, **58-94 % do tempo foi staging, não unidade de matriz**
  (`nosso docs/estudo-prefill-d-wmma.md:10-20`, veredito em `:320-332`). É por isso que a variante
  (b) (2 cabeças por warp, metade do tráfego de LDS) e a (c) (`Bc` maior, menos barreiras por
  chave) estão no desenho desde o começo: se E1 perder, o número que explica será o custo de
  estágio por chave, e ele tem de sair medido e não deduzido.

### E2 — T > 1 (verify/MTP), só se E1 ganhar

A amortização por linhas é maior aqui do que na fusão de GQA: com `T = 4` (o teto do MTP hoje:
`--draft 3` → 4 linhas, `nosso docs/mtp.md`) são **24 linhas por tile** contra 6, ou seja 4× menos
bytes de K/V por linha de consulta; com T=8, 48 linhas (o teto deles,
`ninfer small_t_k8v4.cuh:52`). O instrumento não existe: `bench-attn-gpu` não tem eixo de token
(`bench_attn_gpu.hip:17`) e o nosso verify é `attn_batch_kernel` com grade `(n_head, n_tok)`
(`nosso attn.cuh:416`) — sem compartilhamento nenhum. O caminho é (i) um eixo `--tq N` no bench
para o protótipo D (linhas = `tq × 6`), e (ii) no motor, `bench-phases-gpu --level 2` para isolar
a atenção e `bench --mtp --draft 3` para o passo. Número a mover: 29,33 tok/s ganancioso vs
51,47 com `--mtp --draft 3` a 4K (`nosso docs/mtp.md:3-6`) e a atenção de 12,1 % do passo a
16K (`journal-longctx.md` §3.2); o multiplicador do MTP é gated no custo **marginal por token** do
caminho batelado (`nosso README.md:485-501`). Custo: ~80 linhas no bench + 1 janela.

---

## 7. O que não deu para determinar

- **VGPR/derrame do kernel deles é INFERIDO.** Não há `nvcc` nesta máquina (`which nvcc` → nada) e
  não se usa GPU nesta tarefa, então os "16 f32 de acumulador por lane" são aritmética sobre
  `acc[PVNtPerWarp][4]` com `PVNtPerWarp = D/(ConsumerWarpsPerTile·8)` (`small_t_k8v4.cuh:43-44`)
  — não um número de compilador. Confirmaria com `nvcc -arch=sm_120a -Xptxas -v` no
  `small_t_k8v4.cu` ou `cuobjdump -res-usage` num binário deles.
- **Não existe no repo deles nenhum kernel de decode *não* fundido**, então a fusão de GQA nunca
  foi A/B no próprio repo: não há, do lado deles, um número que isole "fusão" de "tile de linhas"
  e de "formato de KV" e de "T". As geometrias com `GroupSize` 6 e 8
  (`ninfer geometry.cuh:15-16`) são as duas que existem.
- **O que exatamente domina o custo por visita no nosso kernel não está medido** — o candidato é
  latência de L2/IC com pouca paralelidade por warp (95 VGPR no kernel que embarca
  `journal-longctx.md:377`), mas nenhuma medição nossa separa "latência por visita" de
  "banda por byte" nem conta requisições de L2 por chave. E1 mede o efeito agregado; um contador de
  requisições (`--mem`/roofline do `bench-phases-gpu`) mediria a causa.
- **O custo real do derrame de 400 B/lane do HG=6** não foi quantificado em nenhum doc nosso: sem
  ele, não se sabe quanto de `+25 % a 4K / +82 % a 16K` é derrame e quanto é o laço serial de 6
  cabeças. É o número que daria o teto do que a forma do protótipo C poderia render se o
  registrador fosse grátis — e portanto o teto do que E1 pode ganhar.
- **Não li o resto do caminho de atenção deles** (i8, fp8, nvfp4, e o `context_softmax_attention`)
  além do que o §0 usa; o k8v4 e o bf16 são os dois que importam para T=1 e T=2..8. Se algum deles
  tiver um desenho de tile diferente, o §1-§3 pode não valer para esse formato.
