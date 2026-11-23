#!/bin/bash
# Bootstraps Docker + NVIDIA Container Toolkit and starts Inkling-2-Mini via vLLM.
# Env file: /etc/inkling.env (written by CDK user-data before this script).
set -euxo pipefail

ENV_FILE=/etc/inkling.env
# shellcheck disable=SC1090
source "${ENV_FILE}"

LOG=/var/log/inkling-bootstrap.log
exec > >(tee -a "${LOG}") 2>&1

echo "=== inkling2 EC2 bootstrap $(date -Is) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "VLLM_IMAGE=${VLLM_IMAGE}"
echo "VLLM_PORT=${VLLM_PORT}"

dnf -y update || true
dnf -y install docker jq curl || yum -y install docker jq curl

systemctl enable --now docker

# NVIDIA Container Toolkit (AL2023 / RHEL-like)
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  dnf -y install nvidia-container-toolkit || yum -y install nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

mkdir -p /opt/inkling/hf-cache
chmod 777 /opt/inkling/hf-cache

EXTRA_ARGS=()
if [[ -n "${SERVE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${SERVE_ARGS})
fi

DSPARK_ARGS=()
if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
  DSPARK_ARGS=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

HF_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ENV+=(-e "HF_TOKEN=${HF_TOKEN}")
fi

docker pull "${VLLM_IMAGE}"

docker rm -f inkling-vllm >/dev/null 2>&1 || true

docker run -d --name inkling-vllm --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p "${VLLM_PORT}:8000" \
  -v /opt/inkling/hf-cache:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e PYTHONUNBUFFERED=1 \
  "${HF_ENV[@]}" \
  "${VLLM_IMAGE}" \
  "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --reasoning-parser inkling \
  --tool-call-parser inkling \
  --enable-auto-tool-choice \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  "${DSPARK_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

cat >/usr/local/bin/inkling-health <<'EOF'
#!/bin/bash
curl -fsS "http://127.0.0.1:${VLLM_PORT:-8000}/v1/models" >/dev/null
EOF
# rewrite with concrete port
cat >/usr/local/bin/inkling-health <<EOF
#!/bin/bash
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null
EOF
chmod +x /usr/local/bin/inkling-health

echo "=== bootstrap finished; waiting for vLLM in background ==="
(
  for i in $(seq 1 1800); do
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM ready after ${i} checks" | tee -a /var/log/inkling-ready.log
      exit 0
    fi
    sleep 2
  done
  echo "vLLM not ready within timeout" | tee -a /var/log/inkling-ready.log
  docker logs --tail 200 inkling-vllm | tee -a /var/log/inkling-ready.log || true
  exit 1
) &
