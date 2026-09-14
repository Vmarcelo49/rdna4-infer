# Inventário de quantizações do motor (tarefa 4)

Metade de **leitura de código** da tarefa 4. Nenhuma medição de GPU foi feita aqui: os
números marcados **[gerado]** saíram de um utilitário CPU-only que só lê o cabeçalho GGUF
(`load_tensor()` nunca é chamado, nenhum byte de tensor é tocado, nenhum `hipMalloc`), e os
demais vêm de documento deste repo com `arquivo:linha`. Onde eu faço aritmética, está
marcado **[minha conta]**.

**Correção de método**: o briefing sugeria `./build/rdna4-infer --list-tensors`, mas (a) este
worktree não tem `build/`, (b) `--list-tensors` (`src/main.hip:890-900`) imprime
nome/tipo/dims e **não** bytes, e (c) não existe captura dessa saída em lugar nenhum do repo
(verificado: os únicos `.txt` versionados são fixtures de teste). Em vez de rodar o binário de
outro checkout, compilei um dumper em `/tmp` contra `include/` + `src/backend/{gguf,loader,model}.cpp`
**deste worktree** — só leitura de header, `g++`, sem HIP. Ele reusa exatamente
`GgufLoader::open()` (a mesma whitelist e a mesma aritmética de bytes do motor) e
`parse_qwen35_config()` + `validate_qwen35_layout()`. Nos dois arquivos:
`TENSORS 866`, `LAYOUT_VALID 1`.

---

## 1. Todo formato que o motor pode encontrar

A whitelist é a de `include/rdna4/dtype.h:20-36` (15 `DType`, `dtype_from_ggml` em
`:41-60`). Geometria de bloco em `dtype.h:69-97`; kernels em `matvec.cuh`, `dequant_row.cuh`.

- **GEMV** = `T::dot` via `matvec_kernel_gen` despachado por `matvec_launch`
  (`matvec.cuh:516-541`), forma por tipo em `MtShape<N>` (`matvec.cuh:444-458`) e ILP em
  `MtIlp<N>` (`matvec.cuh:463-477`).
- **Batched** = `matvec_kernel_batch` por `matvec_launch_batch_n<N>`
  (`matvec.cuh:551-582`), N ∈ {2,3,4,8,16} (`matvec.cuh:589-596`).
- **Dequant** = `dequant_row_launch` (`dequant_row.cuh:65-104`); o *unit* não é o bloco de
  armazenamento e sim 256 elementos, exceto q8_0 (32) — `dequant_row.cuh:54-64`.
- **Par q8_1** = a ativação é sempre `block_q8_1` de **32** elementos (`quants.h:22,78-81`),
  produzida por `quantize_q8_1_block` (`matvec.cuh:28-54`). Um tensor de tipo `T` com
  `qk != 256` consome `256/32 = 8` blocos q8_1 por bloco de peso; com `qk = 32` consome 1.

| tipo | ord | elems/bloco (`qk`) | bytes/bloco | bpw | GEMV | batched | dequant | par q8_1 |
|---|---|---|---|---|---|---|---|---|
| `F32` | 0 | 1 | 4 | 32.0 | **não** | **não** | `hipMemcpyAsync` (`dequant_row.cuh:67-70`) | n/a (não é peso de matvec) |
| `Q8_0` | 1 | **32** | 34 | 8.5 | sim | sim | sim (unit 32) | **1** bloco q8_1 |
| `Q2_K` | 2 | 256 | 84 | 2.625 | sim | sim | sim | 8 |
| `Q3_K` | 3 | 256 | 110 | 3.4375 | sim | sim | sim | 8 |
| `Q4_K` | 4 | 256 | 144 | 4.5 | sim | sim | sim | 8 |
| `Q5_K` | 5 | 256 | 176 | 5.5 | sim | sim | sim | 8 |
| `Q6_K` | 6 | 256 | 210 | 6.5625 | sim | sim | sim | 8 |
| `IQ2_XXS` | 7 | 256 | 66 | 2.0625 | sim | sim | sim | 8 |
| `IQ2_XS` | 8 | 256 | 74 | 2.3125 | sim | sim | sim | 8 |
| `IQ3_XXS` | 9 | 256 | 98 | 3.0625 | sim | sim | sim | 8 |
| `IQ1_S` | 10 | 256 | 50 | 1.5625 | sim | sim | sim | 8 |
| `IQ4_NL` | 11 | **32** | 18 | 4.5 | sim | sim | sim (unit 256) | **1** bloco q8_1 |
| `IQ3_S` | 12 | 256 | 110 | 3.4375 | sim | sim | sim | 8 |
| `IQ2_S` | 13 | 256 | 82 | 2.5625 | sim | sim | sim | 8 |
| `IQ4_XS` | 14 | 256 | 136 | 4.25 | sim | sim | sim | 8 |

**Resumo (o que o briefing pede):**

- Formatos com kernel GEMV **e** batched **e** dequant: **14** (todos os quantizados). `F32`
  é o único sem nenhum dos três (o dequant dele é `memcpy`; ele não passa por matvec).
- **Tipos com `qk != 256`: exatamente 2 — `Q8_0` (32) e `IQ4_NL` (32)**. `F32` tem bloco de
  1 elemento, mas não é peso de matvec. Todos os k-quants e os demais i-quants são 256.
  Consequência para o par q8_1: só esses dois consomem 1 bloco de ativação por bloco de peso;
  os outros 12 consomem 8.
- **Assimetria de layout a registrar**: a tabela `dtype_block_elems` (`dtype.h:69-76`) devolve
  o bloco de **armazenamento** (32 para `IQ4_NL`), mas o kernel de dequant usa **256**
  (`dequant_row.cuh:71`, com o `x + ibs*(QK_K/QK4_NL)` em `dequant.cuh:288`). Contar `IQ4_NL`
  em blocos de 32 lançava 8× unidades demais e escrevia fora do destino — está documentado
  como CRITICAL em `dequant_row.cuh:54-64`. É a única armadilha de geometria do repo.
- **Variantes A/B de `iq3_s`/`iq2_*`** não são formatos: são corpos alternativos de `vec_dot`
  (`TIQ3S_S`, `TIQ2XXS_S`, … em `matvec.cuh:126-140`) usados só no envio/bench, bit-idênticos
  aos vendados.

---

## 2. Existe algum caminho dequantiza-e-faz-matmul-genérico?

**Não para peso quente.** Os três caminhos que desquantizam antes de multiplicar são:

| caminho | onde | o que faz | custo |
|---|---|---|---|
| linha do embedding | `graph.cuh:1050` (`forward_batch`) e `graph.cuh:1146` (`forward_run`), via `dequant_row_launch(tok_embd_.dt, …)`; e `mtp.cuh:450` | desquantiza **uma linha** de `token_embd.weight` para f32 e depois usa `proj` com peso f32 | 1 lançamento de `dequant_kernel_256` (q3_K, 5120 elementos = 20 blocos) ≈ 4 µs = **0,01 %** do token (`docs/rocm-estudo.md:111-116`) |
| cache KV dentro da atenção | `kv_load<CT>` / `kv_load8<CT>` (`kv.h:80-107`, `:125-169`), usados em `attn.cuh:123-125,145-147` e `:320-322,341-343` | desquantiza elemento a elemento **na leitura**, dentro do laço de chaves | dentro da atenção; não removível (o cache *é* armazenado quantizado por decisão de VRAM — `docs/rocm-estudo.md:98-109`) |
| pesos F32 | não passam por matvec: `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, normas são lidos direto como f32 (`graph.cuh:338`, `upn_f32` em `:455-460`) | — | desprezível |

O matvec de peso **consome o peso quantizado no lugar**: `T::dot` é
`matvec.cuh:97-100` chamando `vec_dot_*_q8_1` (`vecdotq.cuh:348-798`), e o kernel recebe
`const void *__restrict__ vx` apontando para os bytes do GGUF copiados crus
(`matvec.cuh:190-191,236`). **Nenhuma GEMM quente passa por buffer desquantizado** — não
existe GEMM nenhuma no motor (ver `docs/qwen-kernels.md` §3).

**A conversão no load é evitada — verificado.** `Graph::up()` (`graph.cuh:350-365`) faz
exatamente:

```
ld_.load_tensor(i, bytes, err);            // bytes CRUS do GGUF (loader.cpp:13-14: "Tensor DATA stays on disk")
hipMalloc(&t.ptr, bytes.size());
hipMemcpy(t.ptr, bytes.data(), bytes.size(), hipMemcpyHostToDevice);
```

Não há requantização, nem dequantização, nem reempacotamento no host. Os bytes que o kernel
lê são byte-a-byte os bytes do arquivo — é isso que permite ao oráculo de CPU
(`tests/dequant_cpu_oracle.cpp`) comparar o `vec_dot` com `dequantize_row_*` do llama.cpp
sobre o mesmo buffer. `load_tensor` vive em `include/rdna4/loader.h:47` ("Copies the raw
quantized bytes").

---

## 3. A união realmente presente nos dois arquivos UD

**[gerado]** com o dumper descrito no cabeçalho. `ALL` = o arquivo inteiro (é o que
`loader.total_bytes()` soma); `TOK` = o que o decode lê de fato por token, ou seja
**excluindo** `token_embd.weight`, todo o bloco `blk.64.*` (MTP) e todo tensor F32 / sem
forma de matvec. Essa é a mesma regra de exclusão do
`tests/bench_matvec_shapes_gpu.hip:199-204` e do `docs/rocm-estudo.md:135-140`.

### 3.1 `Qwen3.8-27B-UD-IQ3_S.gguf`

`TENSORS 866` · `TOTAL_BYTES 12.029.886.464` · `PER_TOKEN_SKIPPED 907.894.784` →
**tráfego por token = 11.121.991.680 B (11,122 GB)**.

| tipo | ALL nº | ALL bytes | TOK nº | TOK bytes | **share do tráfego/token** | share (M6, errado) |
|---|---|---|---|---|---|---|
| `iq3_s` | 127 | 3.717.120.000 | 127 | 3.717.120.000 | **33,42 %** | 30,9 % |
| `iq4_xs` | 88 | 2.595.880.960 | 88 | 2.595.880.960 | **23,34 %** | 21,6 % |
| `iq3_xxs` | 77 | 1.862.533.120 | 77 | 1.862.533.120 | **16,75 %** | 15,5 % |
| `q5_k` | 16 | 985.825.280 | 16 | 985.825.280 | **8,86 %** | 8,2 % |
| `iq2_s` | 21 | 581.058.560 | 21 | 581.058.560 | **5,22 %** | 4,8 % |
| `q3_k` | 16 | 947.302.400 | **15** | **400.998.400** | **3,61 %** | 7,9 % |
| `iq2_xs` | 12 | 292.495.360 | 12 | 292.495.360 | **2,63 %** | 2,4 % |
| `iq2_xxs` | 12 | 268.984.320 | 12 | 268.984.320 | **2,42 %** | 2,2 % |
| `q4_k` | 23 | 232.980.480 | 23 | 232.980.480 | **2,09 %** | 1,9 % |
| `q2_k` | 6 | 116.981.760 | 6 | 116.981.760 | **1,05 %** | 1,0 % |
| `iq1_s` | 2 | 34.816.000 | 2 | 34.816.000 | **0,31 %** | 0,3 % |
| `q8_0` | 98 | 36.208.640 | **96** | **25.067.520** | **0,23 %** | 0,3 % |
| `q6_k` | 7 | 344.064.000 | **1** | **4.300.800** | **0,04 %** | 2,9 % |
| `iq4_nl` | 1 | 2.949.120 | 1 | 2.949.120 | **0,03 %** | 0,0 % |

### 3.2 `Qwen3.8-27B-UD-IQ4_XS.gguf`

`TENSORS 866` · `TOTAL_BYTES 14.241.849.344` · `PER_TOKEN_SKIPPED 907.894.784` →
**tráfego por token = 13.333.954.560 B (13,334 GB)**.

| tipo | ALL nº | ALL bytes | TOK nº | TOK bytes | **share do tráfego/token** |
|---|---|---|---|---|---|
| `iq4_xs` | 211 | 7.216.660.480 | 211 | 7.216.660.480 | **54,12 %** |
| `q5_k` | 51 | 2.096.005.120 | 51 | 2.096.005.120 | **15,72 %** |
| `iq3_s` | 46 | 1.615.257.600 | 46 | 1.615.257.600 | **12,11 %** |
| `q4_k` | 55 | 1.262.223.360 | 55 | 1.262.223.360 | **9,47 %** |
| `iq3_xxs` | 17 | 545.914.880 | 17 | 545.914.880 | **4,09 %** |
| `q3_k` | 9 | 803.123.200 | **8** | **256.819.200** | **1,93 %** |
| `iq2_s` | 7 | 199.843.840 | 7 | 199.843.840 | **1,50 %** |
| `iq2_xs` | 2 | 51.527.680 | 2 | 51.527.680 | **0,39 %** |
| `iq4_nl` | 1 | 35.389.440 | 1 | 35.389.440 | **0,27 %** |
| `q8_0` | 98 | 36.208.640 | **96** | **25.067.520** | **0,19 %** |
| `q2_k` | 1 | 20.643.840 | 1 | 20.643.840 | **0,15 %** |
| `q6_k` | 8 | 348.364.800 | **2** | **8.601.600** | **0,06 %** |
| `iq2_xxs` | **0** | 0 | 0 | 0 | — |
| `iq1_s` | **0** | 0 | 0 | 0 | — |

**A união difere entre os arquivos**: `IQ4_XS` **não contém** `IQ2_XXS` nem `IQ1_S`
(confirma `PLAN.md:23`, `PLAN.md:56`); 13 `DType` presentes (12 quantizados + `F32`) contra
15 no `IQ3_S`. Um kernel instanciado para os 14 tipos continua correto: as duas instâncias
não usadas simplesmente nunca são despachadas.

### 3.3 O que o M6 errou, e a prova aritmética de que errou

O briefing diz que as shares do `docs/medicoes-m6.md:35-50` foram calculadas sobre
`loader.total_bytes()`. **Confirmado por aritmética, não por leitura de intenção**
[gerado + minha conta]: dividindo cada `ALL bytes` da minha tabela 3.1 pelo
`total_bytes()` = 12.029.886.464, todas as 14 linhas reproduzem a coluna "share of model"
do M6 com 1 casa decimal:

```
iq3_s   3.717.120.000 / 12.029.886.464 = 30,90 %   (M6: 30,9 %)   ✓
iq4_xs  2.595.880.960 / 12.029.886.464 = 21,58 %   (M6: 21,6 %)   ✓
iq3_xxs 1.862.533.120 / 12.029.886.464 = 15,48 %   (M6: 15,5 %)   ✓
q5_k      985.825.280 / 12.029.886.464 =  8,19 %   (M6:  8,2 %)   ✓
q3_k      947.302.400 / 12.029.886.464 =  7,87 %   (M6:  7,9 %)   ✓
iq2_s     581.058.560 / 12.029.886.464 =  4,83 %   (M6:  4,8 %)   ✓
q6_k      344.064.000 / 12.029.886.464 =  2,86 %   (M6:  2,9 %)   ✓
iq2_xs    292.495.360 / 12.029.886.464 =  2,43 %   (M6:  2,4 %)   ✓
iq2_xxs   268.984.320 / 12.029.886.464 =  2,24 %   (M6:  2,2 %)   ✓
q4_k      232.980.480 / 12.029.886.464 =  1,94 %   (M6:  1,9 %)   ✓
q2_k      116.981.760 / 12.029.886.464 =  0,97 %   (M6:  1,0 %)   ✓
q8_0       36.208.640 / 12.029.886.464 =  0,30 %   (M6:  0,3 %)   ✓
iq1_s      34.816.000 / 12.029.886.464 =  0,29 %   (M6:  0,3 %)   ✓
iq4_nl      2.949.120 / 12.029.886.464 =  0,025 %  (M6:  0,0 %)   ✓
```

14/14 exatos. A coluna do M6 é, literalmente, `bytes_do_tipo / total_bytes()`.

**Correção**: a share relevante é `TOK bytes / 11.121.991.680`. Os dois erros que importam:

- **`q3_k`: 7,9 % → 3,61 %** (−4,3 pontos). `token_embd.weight` é `q3_k` (546.304.000 B,
  `dims=[5120, 248320]`) e é contado inteiro pelo divisor antigo, mas o decode lê **uma
  linha** de 2,2 KB. Os 15 tensores `q3_k` do tronco somam 400.998.400 B.
- **`q6_k`: 2,9 % → 0,04 %** (−2,86 pontos). **Todo o bloco MTP é `q6_k`**
  (`blk.64.attn_q`, `blk.64.ffn_down` 73.113.600 B, `blk.64.nextn.eh_proj` 43.008.000 B …) e
  nunca executa. Resta **1** tensor `q6_k` no tronco, de 4.300.800 B.

Também vale corrigir `docs/rocm-estudo.md:139`, que atribui **42,7 MB** ao bloco MTP inteiro.
O bloco `blk.64.*` tem **15 tensores e 351.008.768 B** [gerado]; 43.008.000 B é o
`blk.64.nextn.eh_proj.weight` **sozinho** [gerado]. O "0,9 GB não lidos" do estudo fecha
exatamente, mas por outra decomposição [gerado]:

```
token_embd.weight        1 tensor    546.304.000 B
blk.64.* (MTP)          15 tensores  351.008.768 B
F32 / sem forma de matvec  353 tensores 10.582.016 B
                                     ------------
                                     907.894.784 B   = total_bytes() - tráfego/token
```

O que **não** muda: as bandas por tipo (GB/s) do M6 continuam válidas — elas foram medidas
por tensor, não ponderadas pela share. Só a coluna "share of model" e a projeção
"11,2 GiB per token" (`medicoes-m6.md:61-62`, e a mesma origem em `medicoes-m5.md:44`) é que
estavam erradas: o divisor é 11.122 GB, não 12.030 GB (o M5/M6 herdaram o `total_bytes()`).

### 3.4 Fatos estruturais que caíram fora desta tabela

- Os dois arquivos **compartilham** `token_embd.weight` (`q3_k`, 546.304.000 B),
  `output.weight` (`q5_k`, 874.086.400 B), `output_norm.weight` e **os 15 tensores do bloco
  MTP** (`q6_k`) byte a byte iguais — por isso o `PER_TOKEN_SKIPPED` é o mesmo
  907.894.784 nos dois [gerado]. A Unsloth diferençou só o tronco. (Isso resolve de uma vez a
  contradição do repo: `PLAN.md:302` diz que `output.weight` é `Q6_K` no IQ3_S —
  **está errado**, é `q5_k` nos dois arquivos [gerado]; e o embedding é `q3_k` nos **dois**.)
- A **cabeça LM é `q5_k` e lê 874.086.400 B por token** = **7,86 %** de todo o tráfego do
  token no IQ3_S [minha conta sobre os números gerados] — bate com os 874 MB e 1,39 ms de
  `docs/rocm-estudo.md:128-133`.
- 497 tensores são lidos por token (48 camadas GDN × 8 + 16 full × 7 + 1 LM head
  [minha conta]; confirma `docs/rocm-estudo.md:61,280`).

---

## 4. O que NÃO é suportado, e o que acontece quando um arquivo traz

Os `default: return false` existem e são a segunda linha de defesa. O ponto central é que a
**primeira** linha (a whitelist do loader) é exatamente o mesmo conjunto de tipos que os
kernels despacham, então um arquivo que abre é sempre executável.

| ponto | linha | comportamento |
|---|---|---|
| whitelist do loader | `src/backend/loader.cpp:62-69` (`dtype_from_ggml` → `std::nullopt`, `dtype.h:41-60`) | **falha no `open()`**, com `tensor '<nome>': ggml_type id N (<nome>) outside whitelist`, `fclose` e `return false` |
| `matvec_shape` | `matvec.cuh:413` | `return false` — só F32 e fora-da-união |
| `matvec_launch` | `matvec.cuh:537-538` | `return false` — `// no kernel: caller must fail loudly (SPEC 1.3)` |
| `matvec_launch_batch_n` | `matvec.cuh:578-579` | idem, `RD_BATCH` |
| `matvec_launch_batch` (N) | `matvec.cuh:595` | `return false` — N é instanciação compile-time, não botão de runtime |
| `dequant_row_launch` | `dequant_row.cuh:101` | `return false` (F32 sai antes pelo `memcpy`, `:67-70`) |
| `kv_row_bytes` | `kv.h:74` | **`return 0`** — ver alerta abaixo |
| `kv_type_parse` | `kv.h:52` | devolve string de erro → CLI aborta |
| `attn_launch` / `attn_launch_split` | `attn.cuh:267`, `:562` | `return false` (16 pares K/V instanciados) |

**O fracasso é alto?** Para o caminho de peso, **sim, e a cadeia está fechada**:
`proj`/`proj_qq` convertem o `false` em erro nomeado (`graph.cuh:599`
`"matvec_launch failed (proj_qq)"`), a camada propaga, `forward_run` devolve `false`, e o CLI
imprime em `stderr` e sai ≠ 0 (`src/main.hip:307-309` para o load,
`src/main.hip:340` para o `graph init`). O loader é ainda mais direto: erro no `open()`, sem
GPU, sem fallback. A regra está escrita em `SPEC.md:21` ("Qualquer outro tipo no arquivo →
erro explícito, não fallback silencioso") e repetida em `dtype.h:6-8`.

**Dois pontos que merecem atenção (não são falhas ativas hoje):**

1. **`kv_row_bytes` devolve `0` em silêncio** (`kv.h:66-75`). Hoje o `switch` cobre os 4
   enumeradores de `KvType` e o `return 0` é inalcançável. Mas se alguém acrescentar um
   `KvType` novo e esquecer o caso, a linha da KV passa a ter 0 byte, todas as cabeças leem a
   mesma linha, e **não há erro nenhum** — exatamente a classe de falha que o usuário teme.
   Um `default: __builtin_trap()` (ou `-Wswitch` + `= delete`) fecha isso.
2. **`SPEC.md:21` promete mais do que o código entrega**: o SPEC diz que `F16`/`BF16` são
   aceitos "se aparecerem"; `dtype_from_ggml` **rejeita** os dois (`dtype.h:41-60` não tem os
   ids 1 e 30) e o comentário em `dtype.h:10-11` assume isso ("F16/Q4_0 .. never as GGUF
   tensor types in the UD files"). O código está certo para estes dois arquivos; o SPEC é que
   está desatualizado. Vale corrigir o texto, não o código.

Nota de escopo: o motor também recusa qualquer GPU ≠ gfx1201 em runtime (`SPEC.md:27`), o que
já é o oposto do "fallback silencioso" — é recusa explícita.

---

## 5. Expectativa de acurácia (análise, não medição)

### 5.1 O que é lossy

**Todos os 14 tipos quantizados são lossy; só `F32` não é.** A referência "correta" do
projeto não é o F16 e sim o **llama.cpp rodando o mesmo arquivo GGUF** (`scripts/compare_ppl.sh`,
`docs/medicoes-m5.md:63-67`), o que isola o erro do *motor* do erro da *quantização*.

Hierarquia por bpw publicado (`dtype.h:78-97`, conferida contra o tamanho em disco [gerado]):

| faixa | tipos | bpw | o que se espera |
|---|---|---|---|
| < 2,0 | `IQ1_S` | 1,5625 | o mais agressivo; usado em **2 tensores** (0,31 % do tráfego) |
| 2,0-2,6 | `IQ2_XXS` 2,0625 · `IQ2_XS` 2,3125 · `IQ2_S` 2,5625 · `Q2_K` 2,625 | — | i-quants com grid + tabela de sinais; `Q2_K` é o mais fraco da faixa |
| 3,0-3,5 | `IQ3_XXS` 3,0625 · `Q3_K` 3,4375 · `IQ3_S` 3,4375 | — | `IQ3_S`/`IQ3_XXS` usam grid (melhor que `Q3_K` no mesmo bpw) |
| 4,0-4,6 | `IQ4_XS` 4,25 · `Q4_K` 4,5 · `IQ4_NL` 4,5 | — | quase transparente |
| 5,5-6,6 | `Q5_K` 5,5 · `Q6_K` 6,5625 | — | praticamente sem perda perceptível |
| 8,5 | `Q8_0` | 8,5 | só nas `ssm_alpha/beta` (48×2 tensores, 0,2 % do tráfego) |
| 32,0 | `F32` | 32 | normas, `ssm_a`, `ssm_dt`, `ssm_conv1d` — **sem perda** |

Nota de acurácia estrutural: `F32` **não** é "desperdício" nos sítios onde aparece — normas
RMS e o `ssm_a`/`ssm_dt` do GDN alimentam `expf`/`softplus`, onde o erro relativo é
amplificado; lê-los em f32 é correto. São 353 tensores e apenas 10.582.016 B [gerado].

### 5.2 Como um arquivo UD misto troca tamanho por qualidade

Os dois arquivos são **mistos por tensor**, não por camada: cada tensor recebe o tipo que
minimiza o erro sob um orçamento de bpw. A leitura dos inventários [gerado] mostra a regra:

- `IQ3_S` (3,4375 bpw de média, 11,2 GiB): empurra o tronco para baixo (`iq3_s` 33 %,
  `iq4_xs` 23 %, `iq3_xxs` 17 %) e **paga caro só onde dói** — `q5_k` em 16 tensores
  (incluindo a cabeça LM inteira, 874 MB!), `q4_k`/`q2_k` pontuais.
- `IQ4_XS` (4,25 bpw, 13,3 GiB): gasta 54 % em `iq4_xs` e 16 % em `q5_k`, e sobe o
  `q4_k` para 9,5 %.
- O ponto do "UD" (Unsloth Dynamic) é visível no mesmo tensor entre os arquivos:
  `blk.3.attn_q` é **`q2_k`** no `IQ3_S` e **`iq4_nl`** no `IQ4_XS`; `blk.0.ffn_up` é
  **`iq1_s`** num e **`iq2_s`** no outro [gerado]. Ou seja: os tipos de menor bpw ficam nas
  primeiras camadas / tensores menos sensíveis.
- Consequência de desempenho (importante e frequentemente mal lida): o `IQ4_XS` **decodifica
  quase tão rápido** quanto o `IQ3_S` apesar de ser 18 % maior, porque `iq3_s` é *issue-bound*
  (200-307 GB/s medidos, `docs/rocm-estudo.md:239-249`) e `iq4_xs`/`q4_k` são limitados por
  banda (376-442 GB/s). Mais bytes num tipo eficiente podem custar o mesmo tempo que menos
  bytes num tipo ineficiente. [medido em M5: 27,0 vs 26,8 tok/s; banda efetiva 364 vs 336 GB/s]

### 5.3 Qual arquivo para qual contexto

Dados de VRAM medidos (`docs/medicoes-m5.md:140-155`) e o teto do CLI:

| arquivo | ctx | KV | VRAM em uso | cabe? |
|---|---|---|---|---|
| `IQ3_S` | 4 096 | f16 / f16 | 11,87 GiB | sim, folgado (4,05 GiB livres) |
| `IQ3_S` | 32 768 | f16 / f16 | 13,61 GiB | sim |
| `IQ3_S` | **65 536** | **f16** / f16 | **~15,6 GiB** | sim, no limite |
| `IQ3_S` | 65 536 | q4_0 / q4_0 | 12,73 GiB | sim, com folga |
| `IQ3_S` | **131 072** | **q4_0** / q4_0 | **13,86 GiB** | **sim** |
| `IQ4_XS` | 4 096 | f16 / f16 | 13,91 GiB | sim |
| `IQ4_XS` | **32 768** | **f16** / f16 | **15,66 GiB** | sim, apertado (0,26 GiB) |
| `IQ4_XS` | 65 536 | q4_0 / q4_0 | 14,79 GiB | sim |
| `IQ4_XS` | **131 072** | q4_0 / q4_0 | — | **NÃO** (`hipMalloc failed`) |

Observação sobre a linha de 15,6 GiB do `IQ3_S` a 64K: a tabela do M5 **não** tem essa linha
(aquela rodada mediu 64K só com q4_0); o número vem do briefing e é consistente com o resto —
`13,61 GiB` a 32K f16 + o KV f16 de mais 32K (≈ 2 GB) ≈ 15,6 GiB. Trato como
**[estimativa do briefing]**, não como medição desta sessão.

Recomendação, por contexto:

| contexto | arquivo | por quê |
|---|---|---|
| ≤ 24K, qualidade máxima | **`IQ4_XS`, KV f16** | cabe com 13,91-15,16 GiB; é consistentemente melhor na PPL (3,799 vs 3,823 no chunk 0) e a ~27 tok/s de decode contra 26,8 |
| 32K | **`IQ4_XS`, KV f16** (15,66 GiB, apertado) ou `IQ3_S` f16 se quiser folga | o `info` recusa o `IQ4_XS` 32K f16 por ser conservador (16,27 GiB de orçamento contra 15,66 reais — `medicoes-m5.md:167-170`); o run real cabe |
| 64K | **`IQ4_XS` com KV q4_0** (14,79 GiB) se couber o KV quantizado; senão **`IQ3_S` f16** (~15,6 GiB) | o `IQ4_XS` f16 a 64K não é medido e provavelmente não cabe; o `IQ3_S` **f16** cabe e o f16 é **38 % mais rápido** que o q4_0 na atenção (174 vs 239 ms/token a 64K, `medicoes-m5.md:130-131`) |
| 131K | **`IQ3_S`, KV q4_0** (13,86 GiB) | única combinação que cabe; o `IQ4_XS` falha com `hipMalloc failed` |

Ou seja: **a escolha por contexto é uma decisão de VRAM, não de qualidade.** Até 32K o
`IQ4_XS` é o melhor nas duas dimensões; acima disso o `IQ3_S` é obrigatório, e a partir de
64K o KV **tem** de ser `q4_0` só para caber — aceitando os 38 % de custo na atenção, que é
*issue-bound na desquantização*, não limitado por banda.

### 5.4 A acurácia do motor não é o gargalo

A PPL por posição sobre wikitext-2 (10 chunks de 512, 5120 tokens pontuados,
`docs/medicoes-m5.md:68-85`) fica **dentro de 0,250 %** (IQ3_S) e **0,150 %** (IQ4_XS) da
referência, no pior chunk — e a maior parte disso é ordem de redução, não quantização: o
motor faz um GEMV por token enquanto o llama.cpp em batch usa outro caminho de `MUL_MAT`
(`PLAN.md:223`). O pior desvio de posição única é 0,458 nats (IQ3_S) e 0,873 nats (IQ4_XS).
**Qualquer mudança de kernel que fique dentro de ~1e-7 relativo é invisível nessa régua**,
que é justamente o que torna os gates de bit-exatidão (`check-matvec-gpu`,
`check-batch-gpu`, `check-golden_run.sh`) mais exigentes — e mais úteis — que a PPL.

Ressalva honesta: `IQ4_XS` ser "melhor" na PPL **não** foi medido como diferença
estatisticamente significativa (n = 10 chunks, 1 semente); é a direção esperada por ser um
quant maior (`medicoes-m5.md:88`), e é o que se vê. Não use esses 0,024 de diferença como
argumento forte de qualidade.
