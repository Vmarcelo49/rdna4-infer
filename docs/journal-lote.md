# O matvec em lote (prefill e MTP): o que limita, medido

Continuação do diário da noite (`docs/journal-noite.md`), escrita depois do fechamento,
porque o item ficou aberto com um número e não com uma causa. Janela: 14/09, 07:45-08:40.
Todas as corridas sob `scripts/gpu-lock.sh` com `timeout` dentro do lock.

## Por que esta medição existe

Duas das três pontas do alvo passam por este kernel:

- **prefill**: 123,68 tok/s a 512 tokens, e o `matvec` em lote é **83,7 %** desse tempo
  (C14 do diário da noite). Se ele melhorasse, o prefill melhorava quase na mesma fração.
- **MTP**: o multiplicador medido (1,22× no ramo, 0,96× na árvore final, 2,03× em código)
  depende de quanto custa **o token marginal** dentro do lote de verificação, não do custo
  de uma passagem.

O diário tinha o número (109 GB/s por token contra 446 GB/s do caminho por token) e uma
hipótese explícita deixada para medir (§10 de `docs/journal-kernels.md`): *"a hipótese a medir
é o tráfego de re-leitura da ativação por linha"*, com duas alavancas candidatas (mais cargas
em voo por thread; staging de ativação em LDS). Esta é a medição dessa hipótese — e ela
**refuta** a hipótese e as duas alavancas.

## Ferramenta nova: `--batch` no `bench-matvec-shapes-gpu`

O bench já media o caminho por token sobre o inventário **real** de 497 tensores (11,122 GB
lidos por token). Faltava o caminho em lote. Foram acrescentados:

- `--batch 1,2,4,8,16`: passa o inventário inteiro com N tokens compartilhando uma leitura
  dos pesos, com **uma** dupla de eventos por passagem (fila cheia, como o grafo chama), mais
  um aquecimento descartado por célula (o *ramp* de DPM que este bench existe por causa).
  `N=1` na lista roda o caminho por token, para as linhas da tabela serem comparáveis entre si.
- `--batch-cap`: a mesma passagem em **duas streams**, para separar latência de throughput.
- `--batch-unroll 1,2,4`: o knob de MLP bit-exato forçado dentro do kernel em lote.
- Impressão de `hipFuncGetAttributes` por (tipo, N): registradores, scratch, LDS.

    ./scripts/gpu-lock.sh timeout 900 ./build/bench-matvec-shapes-gpu \
      /mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf \
      --batch 2,4,8,16 --batch-unroll 1,2,4 --reps 3

## Medida 1 — a forma do custo: `pass_ms ≈ 13,9 + 6,04·N`

| N | pass_ms | ms/token | GB/s (passagem) | GB/s por token-equivalente | reuso (act_stride=0) |
|---|---|---|---|---|---|
| 1 (GEMV, referência) | 21,32 | 21,32 | **521,8** | 521,8 | — |
| 2 | 23,96 | 11,98 | 464,2 | 928,4 | 100,0 % |
| 4 | 37,90 | 9,48 | 293,5 | 1173,8 | 99,4 % |
| 8 | 62,84 | 7,86 | 177,0 | 1416,0 | 95,8 % |
| 16 | 110,70 | 6,92 | 100,5 | 1607,5 | 95,0 % |

Ajuste em N=2..16: `pass_ms ≈ 13,9 + 6,04·N` (o N=1 mede 21,3 porque é outro kernel: o GEMV
satura a DRAM em ~500 GB/s). Leitura:

1. **O lado do peso é uma constante de ~13,9 ms por passagem**, e ela **não cresce com N**:
   11,122 GB em 13,9 ms = 800 GB/s efetivos, acima do roofline de 633 GB/s medido — o que diz
   que essa constante **não** é uma passagem completa pelas DRAM (parte é servida pela Infinity
   Cache: a passagem é sequencial por tensor e alguns tensores ficam residentes). O que importa
   aqui é a forma: é o ganho que o lote existe para capturar, e ele **não** melhora com N.
2. **Cada token adicional custa 6,04 ms e esse custo NÃO se amortiza.** É ele que decide o
   prefill (512 tokens = 32 sub-lotes de 16) e é ele que decide o MTP.
3. O ganho por token satura: 3,20× em N=16 contra 1,70× em N=2. Dobrar de 8 para 16 tokens
   rende 12 % por token, não 100 %.

### O que isso prevê para o MTP (e a previsão bate com o medido)

| | ms |
|---|---|
| um passo de decode (1 token, GEMV) | 21,32 |
| uma passagem de verificação com 2 tokens | 23,96 → **1,12×** um passo |
| aceitação necessária só para empatar (D=1) | 56 % |
| aceitação medida em prosa (ramo) | 68 % → 1,22× ✅ |
| aceitação medida em código | 95,2 % → 2,03× ✅ |

Ou seja: o multiplicador do MTP **não é um mistério nem um bug de kernel** — ele é
`2·aceitação / (1,12)` com o custo do token marginal a 6,04 ms. O que separa "MTP paga" de
"MTP não paga" é a taxa de aceitação contra o custo de verificar 2 tokens, e este último é
exatamente o número que a Medida 1 dá. Isso fecha qualitativamente o C14: o MTP está correto
e exato; quem decide o ganho é a aceitação em prosa (68 %) contra um custo de verificação
que o matvec em lote impõe.

## Medida 2 — o controle que refuta a hipótese da ativação (95,0 %)

`act_stride = 0` faz **todos os N tokens lerem a MESMA linha de ativação**. Nada mais muda:
mesmo kernel, mesmas instruções, mesmos dp4a, mesmo tráfego de peso. Só o *working set* da
ativação muda (5 KB em vez de N × 5 KB — 82 KB em N=16, três vezes a L1).

**Resultado: 95,0 % em N=16 (110,70 → 105,21 ms), 95,8 % em N=8, 99,4 % em N=4.** O ganho de
tornar a ativação residente na L1 é de **5 %**, não de metade. A hipótese do §10 do
`journal-kernels.md` está **REFUTADA por medição**: o custo por token não é tráfego de
re-leitura da ativação, nem do L2, nem da Infinity Cache. (Faz sentido olhando o padrão de
acesso: a caminhada em k é sequencial, então a janela viva da ativação por CTA é de poucos
blocos por token, não as 82 KB inteiras.)

**Consequência prática**: o staging de ativação em LDS — que era a alavanca nº 2 da lista de
próximos passos — **não tem alvo**. Fica documentado como refutado, com o número que o refuta.

## Medida 3 — o UNROLL (o MLP bit-exato do caminho por token) não transfere: REFUTADO

O caminho por token embarca `kMtUnroll = 2` justamente nos tipos em que `kMtIlp == 1`
(`iq3_s`, `iq3_xxs`, `iq2_s`, `iq4_nl` — **59 % dos bytes deste modelo**), e o kernel em lote
**não tinha o knob**. Era a diferença estrutural mais óbvia entre os dois caminhos, então foi
implementada: `matvec_kernel_batch` ganhou um parâmetro `UNROLL` com a **mesma semântica** do
`matvec_kernel_gen` (UNROLL blocos por iteração no mesmo acumulador, na mesma ordem — bit-exato
por construção) e um lançador bench-only `matvec_launch_batch_unroll`.

| N | shipping (62,84 / 110,70) | unr1 | unr2 | unr4 |
|---|---|---|---|---|
| 8 | 62,84 ms | 62,61 (−0,4 %) | 62,96 (+0,2 %) | 63,18 (+0,5 %) |
| 16 | 110,70 ms | 110,17 (−0,5 %) | **116,87 (+5,6 %)** | 111,44 (+0,7 %) |

**Refutado**: o knob vale ~0 no lote (dentro do piso de ruído de ±1 % que o próprio bench
mede com a variante repetida) e é **5,6 % pior** em N=16. O `kUnr` do despacho de produção
foi **revertido para 1**; o parâmetro de template, o lançador bench-only e este parágrafo
ficam como registro. Bit-exatidão conferida de qualquer forma:
`check-matmul-gpu` → "OK (bit-exact in every configuration tested)".

Interpretação: no lote o paralelismo de memória já vem do eixo dos tokens (N=16 cadeias
independentes); o UNROLL adiciona pressão de registrador sem adicionar paralelismo novo.

## Medida 4 — não é transbordo de registrador (de graça, sem GPU)

`hipFuncGetAttributes` em 14 tipos × N ∈ {2,4,8,16}: **`localSizeBytes = 0` em todos**.
`numRegs` varia de 40 (`iq4_nl`, N=2) a 212 (`q6_K`, N=16), `sharedSizeBytes` de 0 a 512 B
(as tabelas de redução com WPR>1). **REFUTADO**: não há spill para memória local, e o
`iq3_s`/N=16 usa 63 registradores — longe do teto.

## O que sobra, e o que fica sem explicação

Somando as medidas, o custo por token (6,04 ms) **não é**: tráfego de ativação (Medida 2),
falta de MLP (Medida 3), transbordo de registrador (Medida 4), nem banda de DRAM (o lado do
peso já roda a ~500-520 GB/s no caminho por token, 79-82 % do roofline de 633 GB/s medido).

O que resta é o próprio fluxo de instruções por token — dp4a + cargas de ativação + as
esperas que o compilador insere entre elas — e a contabilidade disso **não fecha** com o
modelo de issue deste GPU:

- por token, `Σ nrows×ncols / 4 ≈ 7,55e9` dp4a (30,2e9 pesos no inventário);
- o §10 do `journal-kernels.md` contou ~675e6 instruções de warp por token no lote;
- em 6,04 ms isso dá ~112e9 instruções de warp/s contra ~640e9 slots de issue/s do cartão
  (64 CU × 4 SIMD × 2,5 GHz) = **~17 % do pico de issue**.

Ou seja: **não é banda, não é tráfego, não é registrador, e também não é saturação de issue**
— sobra latência não coberta ou um recurso que a conta não está vendo. A sondagem de
capacidade (abaixo) descarta uma das duas leituras e a hipótese da L1 (também abaixo) é a que
fica no lugar.

### Registro da sondagem de capacidade (`--batch-cap`)

O inventário inteiro duas vezes, em duas streams, contra uma passagem:

| N | uma passagem | duas passagens em duas streams | razão |
|---|---|---|---|
| 2 | 26,19 ms | 49,58 ms | 1,89× |
| 4 | 33,28 ms | 67,65 ms | 2,03× |
| 8 | 62,02 ms | 120,38 ms | 1,94× |
| 16 | 109,73 ms | 211,80 ms | 1,93× |

**Leitura honesta, e é menos do que eu queria**: as duas passagens praticamente não se
sobrepõem (1,93× de 2,00× em N=16, ~5 % de sobreposição com N=2). Isso mostra que **não há
slot de CTA livre** — cada lançamento já enche as 64 CUs — mas **não** separa "saturado por
throughput" de "cheio de warps parados em latência": se as CUs estão ocupadas por warps que
esperam, uma segunda stream também não acha lugar. Ou seja, a sondagem **descarta a hipótese
"faltam CTAs"** mas deixa as duas leituras de pé (throughput de algum recurso × latência com
ocupação máxima). Quem separa as duas é medir dentro do kernel (mais trabalho por thread, que
é exatamente a alavanca `RM` abaixo) ou um profiler — e **não há profiler instalado** neste
sistema (`rocprof`/`omniperf` ausentes; fica registrado como o que faltou para fechar isto).

### A hipótese que fica no lugar (a medir, não medida)

A contabilidade de recursos **não fecha** com nenhum recurso saturado:

- issue: ~17 % dos slots (112e9 de ~640e9 instruções de warp/s);
- dp4a: ~24 % de um pipe de quarto de taxa (39e9 de ~160e9 por segundo);
- DRAM: o lado do peso roda a 500-522 GB/s no caminho por token (79-82 % do roofline).

O que **não** está na conta e é do tamanho certo: por token, cada linha de peso relê a linha
de ativação inteira, e isso dá `Σ nrows×ncols ≈ 30 GB **por token**` de cargas de ativação.
Em 6,04 ms isso é ~5 TB/s. Se essas cargas **acertassem a L1**, seria 25 % de uma L1 de
20 TB/s (64 CU × 128 B/ciclo × 2,5 GHz) — mas o fluxo de peso (11 GB por passagem) passa pela
mesma L1 e a evictiona continuamente, então é plausível que **toda** carga de ativação vá para
o L2/Infinity Cache, e aí 5 TB/s contra ~1,5 TB/s de IC é saturação por 3×.

Isso também explica por que o controle de reuso **não** discrimina como eu esperava: com
`act_stride = 0` o número de cargas é o mesmo (30 GB/token), só o endereço muda — se o fluxo
de peso expulsa a linha dos dois jeitos, os dois casos são L2-bound e o controle mede ~0.

O experimento que separa: **custo marginal por token por byte, por tipo de quantização**.
`iq1_s` tem 1,5 bit/peso (5,3 elementos/byte) e `iq3_s` 3,4 bit/peso (2,33 elementos/byte):
se o custo marginal for proporcional ao número de *elementos* (volume de ativação), `iq1_s`
custa ~2,3× mais por byte; se for proporcional aos *bytes*, custa igual. Isso é uma filtragem
por tipo no bench (`--only-dt`), ~20 linhas e uma corrida por tipo — **fica como o próximo
experimento, nomeado, com o número que o justifica**.

## Consequências para o backlog (com o número que as prioriza)

1. **`matvec_kernel_batch` continua sendo o item de maior valor do projeto**, agora com o
   alvo estreitado: o que vale atacar é o **custo por token (6,04 ms)**, não o lado do peso
   (13,9 ms, que já roda perto do roofline e satura).
2. **As duas alavancas que estavam na lista estão mortas**: staging de ativação em LDS
   (refutado pela Medida 2) e o UNROLL bit-exato no lote (refutado pela Medida 3). Não
   voltar a elas sem uma medida nova que as reabilite.
3. **O caminho que sobra é mudar a forma aritmética, não afinar esta**: o custo por token é
   proporcional ao número de *elementos de saída* × K, porque cada elemento de saída refaz
   as suas próprias cargas de ativação e o seu próprio dp4a. As duas formas que dividem isso:
   - **tile de registrador em M (`RM` linhas por thread)**: a MESMA carga de ativação serve
     `RM` linhas. É bit-exato pela mesma razão que `ROWS` é livre (muda só qual thread calcula
     qual linha, e a ordem das somas por elemento é preservada), logo tem gate barato.
     `RM=2` corta as cargas de ativação por elemento pela metade; o dp4a por elemento **não**
     muda (é irredutível).
   - **WMMA `i32_16x16x16_iu8`** (existe no gfx1201; MFMA não): 4096 MACs por instrução
     contra 4 do `v_dot4_i32_iu8`. Exige dequantizar os pesos para int8 em LDS e tratar a
     escala por bloco de 32 como correção — reescrita de dias, não de horas, e a primeira
     coisa a medir é isolada (um microbench de WMMA i8 neste cartão), não no motor.
4. **MTP**: o número de aceitação em prosa (68 %) contra o custo de verificação
   (`1,12×` um passo, Medida 1) explica o 1,22×/0,96× sem invocar bug. Melhorar o MTP em
   prosa passa por **subir a aceitação** (draft, temperatura, D), não pelo kernel — mas o
   teto do ganho continua sendo o custo do token marginal, então RM também mexe aqui: se
   2 tokens passassem a custar ~1,05× um passo, o break-even de aceitação cairia para ~52 %.
