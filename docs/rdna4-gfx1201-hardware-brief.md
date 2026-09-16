# RDNA4 / gfx1201 (RX 9070 XT) — brief de hardware para kernels HIP

Fontes: ISA RDNA4, tabela ROCm, kernel Linux, e **medições neste host** (`rocminfo`, ROCm LLVM 22, microbenchmarks HIP com o contador `SHADER_CYCLES`).
**[D]** documentado · **[M]** medido aqui · **[I]** inferido/incerto. Assinaturas de builtins: [BuiltinsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td), [AMDGPUUsage](https://llvm.org/docs/AMDGPUUsage.html), [AMDGPUModifierSyntax](https://llvm.org/docs/AMDGPUModifierSyntax.html).

## 1. Estrutura da CU / WGP

**[D]** "Compute Unit (CU) — One half of a WGP. Contains 2 SIMD32's"; WGP = 4 SIMD32; "The WGP supports up to 32 work-groups with a maximum of **1024 work-items**" — [ISA §1.2.1/§2.3](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content).
**[M]** `rocminfo`: 64 CU, **SIMDs per CU: 2**, Wavefront 32, "Max Waves Per CU: **32**", Workgroup Max 1024. `multiProcessorCount` = **32** → o "CU" do HIP é o **WGP**; 32 WGP = 64 CU (driver) = **128 SIMD32**, wave32.
**[D]** LDS: "Each WGP has a **128kB** memory space … 64 banks, each 512 entries of 4 bytes … A single work-group may allocate up to **64kB**" ([ISA §1.2.2.1/§3.3.5](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content)). Modos **CU mode / WGP mode** (§2.3).
**[D]** VGPRs: "a shader may have up to **256 VGPRs**", quantum 16 (wave32) — ou **24** em "devices that have **1536 VGPRs per SIMD**" (§3.3.2.1).
**[M]** `regsPerMultiprocessor` = **196608 DWORDs = 768 KiB/WGP** (= coluna "VGPR File" da [tabela ROCm](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)) → 49152 DWORDs/SIMD32 = **1536 VGPR/lane/SIMD32**; confirma o quantum 24.
**[D]** Caches ([ROCm](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)): L0 Vector **32 KiB**, L0 Scalar 16, L0 Inst 32, **L2 = 8 MiB**, **Infinity Cache = 64 MiB**, "Graphics L1 Cache: N/A". **[M]** `l2CacheSize` = **8 MiB**; `clockRate` = 2400 MHz.
> ⚠️ **Premissa a corrigir:** os 64 MB são o **Infinity Cache (L3/MALL)**, **não** o L2 — o L2 é **8 MB**.
**[I]** Associatividade do L2 e particionamento 2×32 MB **não são documentados** pela AMD; nenhuma medição publicada encontrada.

## 2. Throughput de instruções

**[D]** Todas existem em gfx1201 ([ISA §7.7 Table 37](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content)): `V_DOT2_F32_F16`(19), `V_DOT4_I32_IU8`(22), `V_DOT8_I32_IU4`(24), `V_PK_FMA_F16`(14), `V_PK_ADD_F16`(15), `V_PERM_B32`. **[M]** `llvm-mc -mcpu=gfx1201` monta todas.
**[M] Medição (ciclos via `s_getreg SHADER_CYCLES_LO`, 8 cadeias independentes, 2 configurações de ocupação):**

| instrução | ciclos | vs `v_fma_f32` |
|---|---|---|
| `v_fma_f32` | 480312 / 600362 | 1.00 |
| `v_perm_b32` | 480393 / 600386 | **1.00 — full-rate** |
| `v_dot4_i32_iu8` | 480403 / 600393 | **1.00 — full-rate** |
| `v_dot2_f32_f16` | 400321 / 400382 | **1.50×** |
| `v_pk_fma_f16` | 400339 / 400403 | **1.50×** |

→ **`v_dot4_i32_iu8` não é emulado nem tem rate reduzido em gfx1201** (mesmos ciclos de `v_fma_f32`). `v_pk_fma_f16`/`v_dot2_f32_f16` são mais rápidos que FMA; `v_perm_b32` e (por extensão) `v_cndmask`/`v_cvt_f32_f16` são full-rate. *(O rate absoluto instr/SIMD/clk ficou ILP-limitado no teste (0,33–1,07) — vale a razão relativa.)*
**[D]** Não há tabela de throughput de VALU no ISA. **[I]** A alegação "v_dot é emulado/baixo throughput em algumas gerações RDNA" **não tem fonte primária** e **não se confirma** aqui. Oficial: FP32 vetorial = FP16 vetorial = 48,7 TFLOPS ([amd.com](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html)) ⇒ 2 FMA/SIMD/clk via VOPD.
**[D]** VOPD é **só wave32**; pares legais incluem `DOT2ACC_F16/BF16`, `CNDMASK`, `ADD_NC_U32`, `LSHLREV_B32` — **nenhum par para `DOT4`/`DOT8`** ([ISA §7.8](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content)).

## 3. Intrinsics (verificado nesta toolchain)

**[M]** `__builtin_amdgcn_sudot4(bool neg_a,int a,bool neg_b,int b,int c,bool clamp)` (**6** args) → `v_dot4_i32_iu8`; `sudot8` idem. `__builtin_amdgcn_udot4(a,b,c,clamp)` (**4** args) → `v_dot4_u32_u8`. `sdot4`/`sdot8` (`dot1-insts`) **não existem** em gfx1201. `__builtin_amdgcn_perm(a,b,sel)` → `v_perm_b32` (também `alignbit`, `alignbyte`, `ds_bpermute`, `ds_swizzle`, `mov_dpp`, `readlane`, `fence`, `s_sleep`).
**[M]** **Não existem** `__builtin_amdgcn_s_wait_loadcnt/dscnt/storecnt` — só `s_waitcnt(n)` (imediato: `[2:0]=expcnt`, `[9:4]=lgkmcnt`, `[15:10]=vmcnt`). Os mnemônicos porém existem em gfx1201 (`s_wait_loadcnt`=0xc0, `_storecnt`=0xc1, `_samplecnt`=0xc2, `_bvhcnt`=0xc3, `_expcnt`=0xc4, `_dscnt`=0xc6, `_kmcnt`=0xc7) — **usar inline asm**. `s_memtime` não existe.
**[D]** Contadores ([ISA §5.7.1](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content)): LOADcnt **6 bits → ≤63 loads em voo por wave**, STOREcnt 6, DScnt 6, KMcnt 5, EXPcnt 3, BVHcnt 3.
**[M]** **`global_load_lds` NÃO existe em gfx1201**: exige target feature `vmem-to-lds-load-insts`, ausente em gfx1100/1200/1201 e presente em gfx90a/942/950. Nenhum opcode `*_LDS` no ISA RDNA4.
**[M]** `raw_buffer_load_b32/b64/b128` existem mas exigem `__amdgpu_buffer_rsrc_t` (`make_buffer_rsrc`). **[D]** Em gfx12 **não há bits GLC/SLC/DLC**: o aux é `[0-2]=th`, `[3-4]=scope`, `[6]=swz` — não-temporal via `th:TH_LOAD_NT`(1), `TH_LOAD_HT`(2), `TH_LOAD_NT_RT`(4), `scope:SCOPE_SE`(8)/`SCOPE_DEV`(16) ([ModifierSyntax](https://llvm.org/docs/AMDGPUModifierSyntax.html)).
**[M]** **`__builtin_prefetch` é NO-OP em gfx1201**. Substituto: `__builtin_amdgcn_s_prefetch_data(const void*, len)` → `s_prefetch_data …` (len 0-31 = chunks de 128 B). `buffer_prefetch` **não existe** em gfx12.
**[M]** `sched_barrier`/`sched_group_barrier` emitem **só anotação**, não instrução. `__syncthreads()` gera **`s_barrier_signal -1` + `s_barrier_wait -1`**; `wave_barrier()` só um comentário. LDS: sem builtins `ds_read_*`; usar ponteiros `address_space(3)` (acessos adjacentes → `ds_load_2addr_b32`).

## 4. Ocupação (curva medida)

**[M]** `hipOccupancyMaxActiveBlocksPerMultiprocessor` + `hipFuncGetAttributes` (tiles FMA):

| VGPRs/thread | 19 | 49 | 82 | 103 | 146 | 192 | 212 | 256 |
|---|---|---|---|---|---|---|---|---|
| **waves/WGP** | 64 | 64 | 56 | 56 | 36 | 32 | 24 | 24 |

Máximo **64 waves wave32/WGP** (16/SIMD32, 2048 threads). Até **~96 VGPRs/thread → ocupação máxima**; 256 VGPRs → 24 waves, e um tile 16×16 FP32/thread **derrama** (576 B). `maxThreadsPerBlock` 1024.
**[I]** A curva não segue `196608/(q24(V)·32)` em todos os pontos — usar a tabela, não uma fórmula.

## 5. Memória

**[D]** 640 GB/s de pico (GDDR6 256-bit @20 Gbps), 16 GB ([amd.com](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html)); "Coalesced access: 70-90% of peak" ([HIP](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.3/understand/hardware_implementation.html)). Latência DRAM **226 ns** (RDNA2/GDDR6); Infinity Cache "+50 ns" sobre hit de L2 ([C&C](https://chipsandcheese.com/p/measuring-gpu-memory-latency), [C&C RDNA4](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot)).
**[I]** Little: 640 GB/s × 226 ns ≈ **145 KB em voo**. Teto de MLP: 63 × 64 waves × 32 WGP × 128 B ≈ 16 MB — folga ampla; o gargalo é ILP/ocupação.
**[D]** Cache por `SCOPE` (0=CU,1=SE,2=DEV,3=SYS) + `TH`: se `ISA.scope > Cache-scope`, "**load cannot hit in this cache**" ([ISA §4.1.1](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content)) → **é assim que se pula o L0/L1** em gfx12.

## 6. Relógio / DPM

**[D]** Nenhum doc AMD cita 9-16 MHz nem "deep sleep" para RDNA4. Kernel: em `pp_dpm_sclk`, "If deep sleep is applied to a clock, the level will be denoted by a special level '**S:**' E.g. S: 19Mhz"; `power_dpm_force_performance_level` = auto/low/high/manual/profile_*, e nos *profiles* "**clock and power gating are disabled**" ([kernel](https://docs.kernel.org/gpu/amdgpu/thermal.html)). **[D]** [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html): "On RDNA3 … and RDNA4 … the AUTO performance mode … **the perfmon clock is gated off** … Setting the performance level to **STABLE_STD** turns the perfmon clock back on" (= `profile_standard`); `amd-smi set -l HIGH` ([docs](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html)).
**[I]** Comunidade em gfx1201: SCLK preso em 1/41/59 MHz entre kernels, corrigido por `echo high > .../power_dpm_force_performance_level` ([llama.cpp #20881](https://github.com/ggml-org/llama.cpp/discussions/20881), [ROCm #6289](https://github.com/ROCm/ROCm/issues/6289)); os "~300 ms de warm-up" só têm relato antigo (Vega20, [hip#1304](https://github.com/ROCm/hip/issues/1304)). Sem guidance oficial.

## 7. Ações

1. int8: `sudot4`/`sudot8` (não `sdot*`); packed half (`v_pk_fma_f16`, `v_dot2_f32_f16`) dá 3× o trabalho de FMA por instrução a 1,5× o rate.
2. `v_perm_b32`/`v_cndmask`/`v_cvt_f32_f16` são full-rate — não vale contorná-las.
3. Esquecer `global_load_lds` e `__builtin_prefetch`; usar `s_prefetch_data` e `GLOBAL_LOAD_BLOCK`/`GLOBAL_STORE_BLOCK` (até 32 VGPRs por instrução, novo em RDNA4, [ISA §11.5](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content)).
4. ≤96 VGPRs/thread para ocupação máxima.
5. Antes de medir: `echo high > /sys/class/drm/card*/device/power_dpm_force_performance_level` e conferir ausência de nível `S:` em `pp_dpm_sclk`.

## 8. Micro-otimizações de banda — missão memória (2026-09-16, READ-ONLY)

Origem: auditoria arquitetura+ISA+assembly sobre `include/rdna4/{matvec,vecdotq,gemm,attn,gdn}.cu{h}`,
`kv.h`, `tuning.h`, `quants.h` (nenhum arquivo do repo foi editado; tudo abaixo é proposta +
receita de medição). Convenção **[D]/[M]/[I]** como no resto do brief. Contexto medido
(`docs/medicoes-banda-e-gargalos.md` §2.2): LM-head GEMV ~620 GB/s (98% do pico de 632,9),
tronco matvec 436 GB/s (69%), GEMM staging 132–275 GB/s, GDN 152 GB/s. Pico teórico 640 GB/s
(256-bit × 20 Gbps / 8); 644,6 GB/s implicaria ~20,14 Gbps efetivos.

### 8.1 Correções de contas que circulavam (usar os §§1–5 deste brief, não os números antigos)
- **[M]** Little com a latência **medida** (§5): 640 GB/s × 226 ns ≈ **145 KB em voo** no
  chip ≈ **4,5 KB ≈ 36 linhas de 128 B por WGP** (32 WGPs). É o que cada WGP precisa
  sustentar — não 27 linhas/CU. Teto de MLP (63 loads/wave × 64 waves/WGP) tem folga
  ampla; o gargalo é ILP/ocupação, como já diz o §5.
- **[D]** LDS tem **64 bancos × 4 B por WGP** (§1), não 32. Toda a matemática de
  conflito abaixo usa `banco = (addr/4) % 64` por WGP (modo CU/WGP, §2.3 do ISA, pode
  mudar o particionamento — checar o modo antes de micro-otimizar bancos).
- **[D]** `sched_barrier`/`sched_group_barrier` emitem **só anotação** (§3): NÃO servem
  para fixar schedule de loads. Pipelining de software aqui = hoisting manual +
  barreira asm (`s_wait_loadcnt` & cia via **inline asm**, §3) + verificação no dump.
- **[D]** `global_load_lds` **não existe** em gfx1201 (§3): nenhum caminho de staging
  pode contar com load global direto para LDS. O veículo de alargamento novo em RDNA4
  é o **`GLOBAL_LOAD_BLOCK`** (até 32 VGPRs por instrução, ISA §11.5, §7 item 3).

### 8.2 Achados novos (não estão nos outros docs)
1. **[I→receita] `kv_load8` quantizado faz `uint64` DESALINHADO.** `block_q8_0.qs` está
   em +2 (lanes atingem +2/+10/+18/+26 mod 8), `block_q4_0`/`block_q4_1` em +4,
   `block_q5_0` em +6 (`kv.h:270-334`). Todo load vetorial de 8 B nesses caminhos é
   split e/ou cruza setor de 32 B. Fix barato: 2× `dwordx2` em offsets 4-alinhados
   (só `kv.h`, valores bit-idênticos); fix caro: pad nos structs (muda layout em
   disco + loader + re-gate `check-kvctx-gpu`). Receita: `--save-temps`, contar
   `global_load_b64` splitados vs pares `b32` no `kv_load8<Q*_0>`.
2. **[I→receita] Staging do GEMM usa ~23% da linha.** Por tarefa (linha, 32 pesos),
   `SolverIq3S::load` (`gemm.cuh:200`) puxa qs0+qs1 (8 B) + qh (1 B) + sg (4 B) + dsc
   (4 B) ≈ 17 B em 4–5 pontos de um bloco de 110 B; janela BK=64 = 2 sub-blocos/linha
   ≈ 29–34 B úteis por linha de 128 B; 16 linhas/warp ⇒ ~464 B úteis / 2048 B
   buscados. 0,23 × 640 ≈ 147 GB/s + ativações ≈ a faixa medida 132–275 GB/s. Todos
   os `Solver*::load` têm a mesma doença. Fix estrutural: W com campos
   intercalados/offline-transposto por janela (loads viram 1–2× b128 sequenciais) —
   ver R1 abaixo.
3. **[D, derivado de `quants.h`] Strides reais por linha (tensores classe K=5120):**
   q4_K 20×144 = 2880 B; iq3_s 20×110 = 2200 B; q6_K 20×210 = 4200 B; q5_K (LM head)
   20×176 = 3520 B. Linhas adjacentes de CTAs adjacentes são contíguas — o
   desperdício do GEMV de tronco é granularidade **intra-warp** (`get_int_b2/b4` =
   b32/b64 por lane, `vecdotq.cuh:102-124`), não inter-linha.
4. **[M, cruzado]** O doc de banda (`medicoes-banda-e-gargalos.md` §4.1) atribui o
   gap do tronco a issue (171 instr/220 B em iq3_s, teto 395 GB/s previsto vs 436
   medido). O item 3 aqui é o mecanismo complementar no lado memória: ~10–20 ops
   VMEM/lane/bloco nos k-quants (21,8% dos bytes). As duas vistas concordam no fix
   (alargar vetorização dos loads / `perm`+`dp4a` no q3_K).

### 8.3 Lista rankeada (mecanismo / delta / risco / alvo exato)
- **R1 — Staging coalescido via W com campos intercalados (offline).** Utilização da
  linha 23%→~85–95% no stream de staging (hoje ~17,5% do kernel); 1,5–2,5× nos
  shapes staging-bound (M≤128): 132–275 → 400–500 GB/s. **[I]** Risco MÉDIO
  (loader + pares `Solver::load/store` em `gemm.cuh:189-880` + re-gate bit-exato vs
  `vec_dot`). Receita antes: contadores TCC_MISS/byte confirmando overfetch ~4×
  (quando `rocprofv3` com `STABLE_STD`, §6).
- **R2 — Política TH_NT nos reads sem reuso.** `__builtin_nontemporal_load` em
  `get_int_b2/b4` (`vecdotq.cuh:113-124`), `Solver*::load`, `kv_load8` quant,
  walks do GDN. Codificação exata já neste brief: `th:TH_LOAD_NT`(1),
  `TH_LOAD_HT`(2), `scope:SCOPE_DEV`(16) (§3). **[I]** +3–8% GEMV/GEMM (~+20–50 GB/s;
  0,5–1,3 ms/token em 10,36 GiB). Risco BAIXO-MÉDIO (só codegen; LUT e ativações
  ficam TH_RT; conferir `th:1` no dump).
- **R3 — Alargar loads estreitos (b128 ou GLOBAL_LOAD_BLOCK).** Cooperativo por warp
  nos k-quants; `GLOBAL_LOAD_BLOCK` (32 VGPRs/instr, §7.3) é o veículo preferido a
  `dwordx4` onde couber. **[I]** +5–12% nos tipos k-quant. Risco MÉDIO
  (`vecdotq.cuh:380-560` + walk em `matvec.cuh:372-410`; bit-exatidão pela regra
  mesma-ordem do UNROLL).
- **R4 — Alinhar/dividir `kv_load8` quant (§8.2.1).** **[I]** +2–5% (variante b32,
  risco BAIXO, só `kv.h:270-334`) a +10–15% (pad + broadcast de escala via
  `readfirstlane`+`s_load`, risco MÉDIO). Relevante onde KV domina (64K).
- **R5 — Fundir as 2 passadas do GDN dentro do token.** Estender o path com tile em
  LDS (`gdn.cuh:479-580`) para ler cada linha 1×/token (2 acumuladores ou linha
  residente em LDS). **[I]** +10–20% no GDN de prefill; ~0 delta de DRAM no decode
  (decode-VEC 1496 GB/s é residente em IC — não perseguir como DRAM). Risco MÉDIO
  (ordem da recorrência bit-exata; maquinaria RPT em `gdn.cuh:107-200` mostra o
  padrão). Alvo: `gdn.cuh:363-430` + variante tile.
- **R6 — MINB no path de produção para ≤96 VGPRs.** `matvec_launch` passa MINB=0
  (`matvec.cuh:760`); iq4_xs usa 119 VGPR (~8 waves/CU) — cap por tipo (8–12) dobra
  os misses em voo/CU nos tipos latency-bound. **[I]** +2–6%. Risco BAIXO
  (bit-exato; zerar `scratch_*` no dump). Alvo: `matvec.cuh:624-633` + `tuning.h`.
- **R7 — Pipeline de `pf_load` sobre o consumo (asm, sem sched_barrier).**
  Hoist de `pf_load(k0+BK)` + interleave `load[i+1]`/store`[i]` (`gemm.cuh:898-945`,
  loop 979-1311); fixar com `s_wait_loadcnt`-via-asm (§3) se o compilador afundar os
  loads. **[I]** +2–5%. Risco BAIXO.
- **R8 — Broadcast via scalar cache do que é uniforme no warp.** `scales[sub>>1]`,
  `d` do bloco, `dsc` do iq3_s, `gate[h]/beta[h]` (`gdn.cuh:119-120`):
  `v_readfirstlane` + `s_load` via K$ (§1). **[I]** +1–3% no GEMV IQ. Risco BAIXO.
  (Linha `q` da atenção NÃO é uniforme — fora.)
- **R9 — Stream-K nos GEMVs pequenos do tronco.** Cauda de 436 GB/s = poucas CTAs ou
  linhas ≤20 blocos (rampa/cauda dominam); dividir K longo entre CTAs com redução
  em 2 estágios (padrão já provado pelo `attn_split`, `attn.cuh:496-660`).
  **[I]** +5–15% na cauda, ~+2–4% no tronco agregado. Risco MÉDIO (kernel novo +
  entrada em `tuning.h`; ordem de soma muda ⇒ gate de equivalência numérica como
  `scripts/check_attn_split.sh`). Alvo: família `matvec_kernel_batch`
  (`matvec.cuh:447`) + `tuning.h`.
- **R10 — FLAT→MUBUF com scope:DEV explícito.** `__builtin_amdgcn_raw_buffer_load`
  + `__amdgpu_buffer_rsrc_t` (§3): offset em 1 VGPR, melhor info p/ coalescer.
  **[I]** +1–2%. Risco BAIXO-MÉDIO (setup nos launch helpers; conferir
  `buffer_load_* scope:2` no dump). Alvos: walk de `rowp` (`matvec.cuh:328`),
  `gemm.cuh:911`, `kv.h:251`.
- **R11 — LDS-LUT nos iq2_* condicional a R9.** Hoje é perda medida (0,60–0,91×)
  porque LUT (2–8 KB) ≥ peso/CTA (regra 2×, `matvec.cuh:806-826`); com mais
  linhas/CTA o denominador vira e o precedente iq3_s (1,070×)/iq3_xxs (1,182×)
  pode se repetir. **[I]** até +7–18% nesses tipos SE a razão virar. Risco BAIXO
  (código existe, `matvec.cuh:828-882`). Follow-up, não standalone.
- **Abaixo da barra de 1% (excluídos com motivo):** merge/split-partials da atenção,
  RoPE, `deinterleave_q_gate`, `kv_store_row` — O(KB) contra 10,36 GiB/token. (Nota:
  a 4K/16K a atenção é cache-bound a ~1,2–1,3 TB/s emitidos pelo 6× do GQA e a 64K
  q4_0 é issue-bound a 346 GB/s — `medicoes-banda-e-gargalos.md` §§2.3/3; nada disso
  é DRAM: não aplicar R2/R10 ali esperando banda.)

### 8.4 Ordem de A/B sugerida
Instrumentação (§6 + contadores TCC quando disponíveis) → R2 → R6 → R4-b32 → R3
(q4_K) → R1 (protótipo iq3_s) → R7 → R5 → R9 → R10 → R11-condicional. Cada passo:
diff de ISA (`--save-temps`) + atribuição (contadores ou A/B de bytes) + gates
existentes (`check-matvec-gpu --check-lds`, `check-matmul-gpu`, `check-batch-gpu`,
`check-kvctx-gpu`, `check_attn_split.sh`).

### 8.5 Vereditos da fila (sessão 16/09 — commits citados)
- **R2 TH_NT: MORTO (-11..-14% tok/s). [M]** `__builtin_nontemporal_load` emite
  `th:TH_LOAD_NT` de verdade (sonda), mas os dois gêmeos medem mais lentos:
  b2 (2× ushort NT, transações partidas) -14%, b4 (q2_K + laço aux iq4_xs) -11%.
  Mecanismo: essas leituras são críticas em latência load→uso (load → perm/tabela
  → dp4a); trocar hits ~30 ns por DRAM ~300 ns na cadeia dependente anula qualquer
  economia de poluição. Revertido; lápide em `vecdotq.cuh`.
- **R6 MINB: premissa morta.** Sonda de atributos nos kernels exatos que embarcam:
  iq4_xs = 40 VGPR (não 119), iq2_s = 71/77 (não 80–120) — ambos já em ocupação
  máxima; só iq2_xs passa de 96 (99) e derrama ao capar (rejeitado). Knob `kMtMinb`
  embarcado como no-op (iq4_xs=8, resto 0). Sem +2–6% disponível por este mecanismo.
- **R1 staging coalescido: FAIL 1,049× (< 1,3×). [M]** Protótipo bench-only
  (repack window-major + kernel mesmo-tile): 0,753 → 0,718 ms, piso de ruído
  1,001×, bit-idêntico (0/2228224 bits). Staging não é limitado por utilização
  de linha. Negativo documentado no bench, sem proposta de produção.
- **R9 stream-K: FAIL 0,55–0,82× em todos os S. [M]** Protótipo bench-only:
  S=1 → 0,820×, S=2 → 0,678×, S=4 → 0,555× no agregado da cauda (ruído 1,002×),
  perde nas 23 classes de cauda. A cauda é limitada por piso de
  lançamento/sync (coluna ms-fix): um segundo lançamento só soma taxa.
  Negativo documentado no bench, sem proposta de produção.
- **Política de splits dependente de chaves: MORTA com medida (sessão 17/09).
  [M]** A alegação pendente (+8,7% kernel com 24 splits @131K q4_0) não
  reproduz nesta árvore: 24×16 = 2,1913 ms vs política 16×16 = 2,1960 ms =
  **1,002×** com piso de ruído 0,999× (0,075 ms/token em jogo, ~0,1% do token
  de 131K — abaixo da barra de 1%). A 64K q4_0 a melhor célula também == a
  política. Provável causa: o hoist de unpack em `kv.h` (commit 81da3e5)
  deslocou o ótimo. Sem mudança de política.
- **E1 row-tile em q4_0 @64K: FAIL 0,575× (regime que faltava). [M]** 2,0864 vs
  1,1998 ms da política; grade fina + loads de 1 warp + barriers por chave
  perdem também onde o unpack domina. E1 morto nos dois regimes; E2 fechado.
- **R7 sched-barrier: PULADO com motivo.** O hoist que ele queria
  (`pf_load(k0+BK)` acima do consume) já embarca no laço principal; o restante
  é só anotação de escalonador (~0 esperado). Sem medição.
- **R10 MUBUF: MORTO com motivo (sem medição). [M]** Intrínsecos provados reais
  nesta toolchain (`__amdgpu_buffer_rsrc_t` +
  `__builtin_amdgcn_raw_buffer_load_b32` → `buffer_load_b32 v, s[0:3], offen`),
  mas a superfície cobre todas as assinaturas `vec_dot` (SRD por tensor de peso)
  para +1–2% esperado — abaixo da barra dado o kill-rate desta fila.
- **R4 b32 em kv_load8: EMBARCADO (+0,9%). R3 fusão dm+scales q4_K/q5_K:
  EMBARCADO (+9,5% no tipo, neutro no decode por share). R5 fusão 2-pass GDN:
  EMBARCADO (+4,1% prefill, bit-exato).** R11-condicional continua condicionado
  a R9 (morto) — arquivado.
