# Inkling-2-Mini on EC2 (vLLM)

**EC2 GPU**에 `vllm/vllm-openai:inkling-2-mini`를 직접 올립니다.

## 배포

루트의 `installer_ec2.py`를 사용하세요.

```bash
python installer_ec2.py
python installer_ec2.py --instance-type p5.4xlarge
python installer_ec2.py --allowed-cidr 203.0.113.10/32
```

수동 CDK:

```bash
npm install
npx cdk deploy \
  -c model=thinkingmachines/Inkling-2-Mini \
  -c instance_type=p5.4xlarge \
  -c allowed_cidr=0.0.0.0/0 \
  --require-approval never
```

## 구성

| 항목 | 기본값 |
| --- | --- |
| 인스턴스 | `p5.4xlarge` (H100×1) |
| AMI | DLAMI AL2023 + OSS NVIDIA driver (SSM) |
| 이미지 | `vllm/vllm-openai:inkling-2-mini` |
| API | `http://<public-ip>:8000/v1` |
| 모델 | 기동 시 HF Hub 다운로드 |

## 삭제

```bash
python uninstaller_ec2.py --yes
```
