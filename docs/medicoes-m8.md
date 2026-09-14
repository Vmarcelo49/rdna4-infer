# M8 — prefill em batch (e por que a decodificação especulativa do MTP ainda não paga)

## O que foi implementado

`Graph::forward_batch(tokens, start_pos, ...)` em `include/rdna4/graph.cuh`: processa
2..16 tokens em **uma passada layer-major**. O que é batch:

- as projeções — uma leitura de cada peso serve os N tokens, pelo
  `matvec_kernel_batch` já validado bit-exato no M6 (`docs/medicoes-m6.md`);
- a quantização de ativação (`quantize_q8_1_batch_kernel`, um warp por bloco de 32,
  mesma aritmética do kernel por linha);
- as norms (`rms_norm_launch` já aceita N linhas) e as operações elementwise;
- o upload das posições (1 `hipMemcpy` no lugar de N cópias síncronas de 4 bytes);
- e, quando o par gate/up do FFN lê a mesma ativação, a quantização é feita **uma vez**
  (`act_ready`) — o item que o estudo de ROCm apontou como redundância (192 das 305
  quantizações por token eram repetidas).

O que **continua sequencial**, porque é inerentemente sequencial: as escritas no KV
cache, a atenção token a token (já com split-KV) e a recorrência do GDN. Todas as
chamadas mantêm a aritmética do caminho por token, então o resultado é
**bit-idêntico** — que é o que torna a mudança segura sem revalidar o modelo.

## Gate: bit-exatidão (`tests/check_batch_gpu.hip`)

```
    N per-token ms   batched ms      speedup     rel-L2     max|d|
    2       56.102       32.530        1.72x    0.0e+00    0.0e+00  BIT-EXACT
    3       49.776       27.860        1.79x    0.0e+00    0.0e+00  BIT-EXACT
    4       44.879       19.683        2.28x    0.0e+00    0.0e+00  BIT-EXACT
    8       37.639       15.386        2.45x    0.0e+00    0.0e+00  BIT-EXACT
   16       36.645       13.974        2.62x    0.0e+00    0.0e+00  BIT-EXACT
full prompt (48 tokens, mixed 16+tail chunks): rel-L2 0.000e+00 max|d| 0.000e+00  BIT-EXACT
check-batch-gpu: OK (batched prefill is bit-identical to per-token)
```

O último caso é o que o CLI realmente faz (chunks de 16 com cauda por token), e o
teste também roda o prompt inteiro dos dois jeitos comparando os logits finais — que
dependem de todas as linhas de KV e de todo o estado recorrente.

## Ganho medido (`bench --prefill`, IQ3_S, mesmo prompt repetido)

| prompt | antes (por token) | agora (batch) | ganho |
|---|---|---|---|
| 512 tokens | 18,09 s (28,30 tok/s) | **7,37 s (69,47 tok/s)** | 2,45× |
| 2048 tokens | 71,56 s (28,62 tok/s) | **29,18 s (70,19 tok/s)** | 2,45× |

O ganho cresce com o tamanho do prompt (até o teto de N=16 por chunk) e é plano daí
em diante: 70 tok/s contra os 440 tok/s da referência (llama.cpp, Vulkan, em batch) —
a distância que resta é o laço interno do `vec_dot`, não o batching.

## Reuso da quantização de ativação no caminho de decode (+1,3%)

O estudo de ROCm mediu que **192 das 305 quantizações de ativação por token eram
redundantes** (a mesma `d_xn_` quantizada 3× na atenção, 2× no FFN, 4× no GDN; nada
escreve nela entre as projeções). `proj_qq()` reusa os blocos `q8_1` já calculados
pelo `proj()` anterior, com o contrato documentado no header. A/B na mesma máquina
(3 repetições de 32 tokens, ctx 4096, aquecimento de 8 fora da medida):

| | tok/s |
|---|---|
| re-quantizando (antes) | 28,89 |
| reusando (agora) | **29,26** (+1,3 %) |

São 384 lançamentos a menos por token (~2200 → ~1800), e o ganho está na borda do
ruído de execução (1-2 %), mas é **bit-exato** — `check-graph-gpu` (oráculo por nó)
PASS e `check_golden_run.sh` OK provam que os valores não mudaram.

## Decodificação especulativa com a cabeça MTP: medida, e **não** implementada

Com `forward_batch` e a aceitação medida da cabeça MTP (86,7% em texto natural,
`docs/mtp.md`), é possível projetar o ganho da decodificação especulativa usando só
números medidos — custo do passo de rascunho 2,24 ms, passo do tronco 36 ms, verificação
em batch de 14,0-32,5 ms/token conforme N:

| D | tokens esperados/rodada | verificação | rodada | ms/token | ganho | só-aceitação-total |
|---|---|---|---|---|---|---|
| 2 | 2,62 | 65 ms | 72 ms | 27,5 | 1,31× | 1,14× |
| 3 | 3,27 | 84 ms | 94 ms | 28,6 | 1,26× | 1,03× |
| 4 | 3,84 | 79 ms | 91 ms | 23,8 | **1,51×** | 1,13× |
| 8 | 5,44 | 123 ms | 146 ms | 26,8 | 1,34× | 0,77× |

**Conclusão: 1,3-1,5× no melhor caso, e isso já assumindo a parte difícil pronta.** A
parte difícil é o *rollback* do estado recorrente: o modelo é híbrido (48 camadas GDN
com estado), então aceitar um prefixo parcial de rascunhos deixa o estado avançado por
tokens rejeitados. O llama.cpp resolve isso com checkpoints de estado
(`common/speculative.cpp`, "single-position checkpoints … on restore"); aqui seria:
snapshot de `d_state_`/`d_convst_` (157 MB, ~0,5 ms) + captura dos tensores de entrada
do GDN por token/camada (qkv/beta/alpha, ~8 MB por rodada) + replay da recorrência
apenas para o prefixo aceito (~0,8 ms/token). A variante simples (só aceitar a rodada
inteira) mede **1,0-1,1×**, ou seja, não paga.

Ou seja: a especulação fica 1,3-1,5× por um trabalho considerável e com risco de
correção num modelo recorrente — e o teto é baixo **pela mesma razão que o prefill não
chega aos 440 tok/s**: o `vec_dot` por byte de peso é caro (issue-bound, medido no M6),
então verificar 4 rascunhos custa quase o mesmo que 4 passos normais (79 ms contra
144 ms). O caminho que destrava os dois é o mesmo: um laço interno mais barato (kernel
tiled estilo MMQ). Está registrado aqui para quem retomar.
