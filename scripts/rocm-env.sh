# Adds /opt/rocm/bin to PATH for hipcc, rocm-smi, etc.
# Usage: source scripts/rocm-env.sh
export PATH="/opt/rocm/bin:${PATH}"
export HIP_PATH="${HIP_PATH:-/opt/rocm}"
