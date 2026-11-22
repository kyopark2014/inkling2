# SageMaker inference image for Inkling-2-Mini via vLLM day-0 image.
# Built by CDK (ContainerImageBuild). Locally:
#   docker build --platform linux/amd64 -t inkling2-sagemaker -f Dockerfile .
FROM vllm/vllm-openai:inkling-2-mini

USER root
WORKDIR /opt/program

COPY sagemaker_entrypoint.py /opt/program/sagemaker_entrypoint.py
COPY serve.sh /opt/program/serve.sh
RUN chmod +x /opt/program/serve.sh /opt/program/sagemaker_entrypoint.py

ENV SM_VLLM_MODEL=thinkingmachines/Inkling-2-Mini \
    SM_VLLM_TENSOR_PARALLEL_SIZE=1 \
    SM_VLLM_MAX_MODEL_LEN=65536 \
    SM_VLLM_GPU_MEMORY_UTILIZATION=0.90 \
    SM_VLLM_ENABLE_DSPARK=0 \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
# SageMaker appends "serve"; serve.sh ignores extra args.
ENTRYPOINT ["/opt/program/serve.sh"]
