# SageMaker CDK (Inkling-2-Mini + vLLM)

`thinkingmachines/Inkling-2-Mini`를 vLLM day-0 이미지(`vllm/vllm-openai:inkling-2-mini`)로 SageMaker Real-time Endpoint에 올립니다.

루트에서 `python installer.py`를 사용하는 것을 권장합니다.

```bash
npm install
npx cdk deploy \
  -c model=thinkingmachines/Inkling-2-Mini \
  -c instance_type=ml.p5.4xlarge \
  -c fallback_instance_types=ml.g5.xlarge \
  --require-approval never
```

컨텍스트 키: `model`, `model_revision`, `instance_type`, `instance_count`, `tensor_parallel_size`, `max_model_len`, `enable_dspark`, `serve_args`, `fallback_instance_types`.
