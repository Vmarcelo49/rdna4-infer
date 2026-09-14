#!/bin/bash
# Gate de regressao do bug de stride de V no caminho em lote (docs/estudo-prefill.md 3h).
#
# O bug: kv_write_batch e o kernel de atencao em lote usavam kv_bytes_ (o stride de K)
# tambem para a V. d_v_ e' alocado com n_attn * kv_v_bytes, entao sempre que a linha de
# K fosse MAIOR que a de V a camada il era enderecada fora do slot dela:
#   K=f16 /V=q4_1 -> 86,5 MB alem do fim  => page fault
#   K=q5_0/V=q4_1 ->  3,9 MB alem do fim  => page fault  (o par recomendado para 131K)
#   K=q8_0/V=q4_1 -> 27,5 MB alem do fim  => passava CORROMPENDO memoria alheia
# O gate antigo (check-batch-gpu) cravava F16/F16, entao nenhum dos tres tinha cobertura.
#
# Este script roda o gate por PAR de tipos de KV. Cada invocacao constroi um Graph
# (uma copia dos pesos), entao a lista e' curta de proposito: os tres pares que
# quebravam, os dois de controle (linhas iguais) e o default.
#
# uso: ./scripts/check_kvbatch.sh [modelo.gguf]
set -u
cd "$(dirname "$0")/.."
MODEL="${1:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
PARES="f16:f16 q5_0:q5_0 f16:q4_1 q5_0:q4_1 q8_0:q4_1"
falhas=0
for par in $PARES; do
  K="${par%%:*}"; V="${par##*:}"
  printf '%-28s ' "KV K=$K V=$V"
  out=$(./scripts/gpu-lock.sh timeout 900 ./build/check-batch-gpu "$MODEL" 2 4 16 \
          --kv-k "$K" --kv-v "$V" 2>&1)
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FALHA (rc=$rc)"; echo "$out" | tail -4; falhas=$((falhas+1)); continue
  fi
  if ! echo "$out" | grep -q "^check-batch-gpu: OK"; then
    echo "FALHA (sem OK)"; echo "$out" | tail -4; falhas=$((falhas+1)); continue
  fi
  # Bit-exatidao: o gate imprime `rel-L2 0.000e+00 ... BIT-EXACT` por caso. A checagem
  # tem de ser "toda linha que fala de rel-L2 carrega 0.000e+00" -- a versao anterior
  # usava `grep -E "rel-L2 [^0]"` e casava com o CABECALHO da tabela
  # (`rel-L2     max|d|`), reprovando os 5 pares por engano.
  ruins=$(echo "$out" | grep "rel-L2" | grep -v "0.000e+00" | grep -v "max|d|" | head -3)
  if [ -n "$ruins" ]; then
    echo "FALHA (nao bit-exato)"; echo "$ruins"; falhas=$((falhas+1)); continue
  fi
  if ! echo "$out" | grep -q "BIT-EXACT"; then
    echo "FALHA (sem BIT-EXACT)"; falhas=$((falhas+1)); continue
  fi
  echo "OK"
done
if [ $falhas -ne 0 ]; then
  echo "check_kvbatch: FALHA em $falhas de $(echo $PARES | wc -w) pares"
  exit 1
fi
echo "check_kvbatch: OK ($(echo $PARES | wc -w) pares de KV, todos bit-exatos no caminho em lote)"
