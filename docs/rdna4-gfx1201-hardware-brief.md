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
