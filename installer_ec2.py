#!/usr/bin/env python3
"""EC2 GPU 설치 스크립트 (SageMaker 대안).

deploy/ec2 CDK로 Inkling-2-Mini (vLLM)를 EC2에 배포하고,
결과를 config.json 의 ec2 섹션에 저장합니다.

SageMaker ml.* 쿼터/용량 실패를 피하기 위해 EC2 p5/g* 인스턴스에
공식 이미지 `vllm/vllm-openai:inkling-2-mini` 를 직접 기동합니다.

기본값:
  model.hub_id      = thinkingmachines/Inkling-2-Mini
  instance_type     = p5.4xlarge
  region            = AWS 세션 기본 리전 (없으면 us-west-2)
  allowed_cidr      = 설치 머신 공인 IP /32 (실패 시 0.0.0.0/0)

실행:
  python installer_ec2.py
  python installer_ec2.py --instance-type p5.4xlarge
  python installer_ec2.py --allowed-cidr 203.0.113.10/32
  python installer_ec2.py --model thinkingmachines/Inkling-2-Mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
CDK_DIR = SCRIPT_DIR / "deploy" / "ec2"

DEFAULT_MODEL = "thinkingmachines/Inkling-2-Mini"
DEFAULT_INSTANCE_TYPE = "p5.4xlarge"
DEFAULT_REGION = "us-west-2"
DEFAULT_REVISION = "main"
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:inkling-2-mini"
DEFAULT_VLLM_PORT = 8000
DEFAULT_VOLUME_GB = 250
READY_TIMEOUT_S = 3600
READY_POLL_S = 15
# Prefer non-a AZs first when a region often lacks p5 capacity in -a.
DEFAULT_AZ_ORDER = {
    "us-west-2": ["us-west-2b", "us-west-2c", "us-west-2d", "us-west-2a"],
    "ap-northeast-1": [
        "ap-northeast-1c",  # p5.4xlarge supported here; 1a often is not
        "ap-northeast-1d",
        "ap-northeast-1a",
    ],
    "us-east-1": [
        "us-east-1e",
        "us-east-1f",
        "us-east-1c",
        "us-east-1d",
        "us-east-1b",
        "us-east-1a",
    ],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def stack_name_for(model: str) -> str:
    """deploy/ec2/lib/inkling-ec2-stack.ts 와 동일한 규칙."""
    name = (model.split("/")[-1] if "/" in model else model).lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name).strip("-")
    if not re.fullmatch(r"[a-z0-9](-*[a-z0-9]){0,50}", name):
        raise ValueError(f"스택 이름을 만들 수 없습니다: {model}")
    return f"{name}-ec2"


def normalize_ec2_instance_type(raw: str) -> str:
    t = raw.strip()
    if t.startswith("ml."):
        return t[3:]
    return t


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            log(f"config.json 파싱 실패, 새로 작성합니다: {exc}")
    return {}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def resolve_identity(region: str) -> tuple[str, str]:
    session = boto3.Session(region_name=region)
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    return identity["Account"], identity["Arn"]


def detect_public_cidr() -> str:
    try:
        with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=5) as resp:
            ip = resp.read().decode("utf-8").strip()
        if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", ip):
            return f"{ip}/32"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log(f"공인 IP 감지 실패 ({exc}); allowed_cidr=0.0.0.0/0")
    return "0.0.0.0/0"


def ensure_defaults(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    session = boto3.Session()
    region = (
        args.region
        or config.get("region")
        or session.region_name
        or DEFAULT_REGION
    )
    account_id, _ = resolve_identity(region)

    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    ec2 = config.get("ec2") if isinstance(config.get("ec2"), dict) else {}
    cdk = ec2.get("cdk_context") if isinstance(ec2.get("cdk_context"), dict) else {}

    hub_id = args.model or model.get("hub_id") or cdk.get("model") or DEFAULT_MODEL
    instance_type = normalize_ec2_instance_type(
        args.instance_type
        or ec2.get("instance_type")
        or cdk.get("instance_type")
        or DEFAULT_INSTANCE_TYPE
    )
    revision = model.get("revision") or cdk.get("model_revision") or DEFAULT_REVISION
    stack_name = stack_name_for(hub_id)
    tensor_parallel = int(
        args.tensor_parallel_size
        if args.tensor_parallel_size is not None
        else cdk.get("tensor_parallel_size", 1)
    )
    max_model_len = int(
        args.max_model_len
        if args.max_model_len is not None
        else cdk.get("max_model_len", 65536)
    )
    enable_dspark = (
        bool(args.enable_dspark)
        if args.enable_dspark is not None
        else bool(cdk.get("enable_dspark", False))
    )
    vllm_port = int(
        args.vllm_port
        if args.vllm_port is not None
        else cdk.get("vllm_port", DEFAULT_VLLM_PORT)
    )
    volume_size = int(
        args.volume_size_gb
        if args.volume_size_gb is not None
        else cdk.get("volume_size_gb", DEFAULT_VOLUME_GB)
    )
    vllm_image = (
        args.vllm_image
        or ec2.get("vllm_image")
        or cdk.get("vllm_image")
        or DEFAULT_VLLM_IMAGE
    )
    if args.allowed_cidr:
        allowed_cidr = args.allowed_cidr
    elif ec2.get("allowed_cidr") or cdk.get("allowed_cidr"):
        allowed_cidr = str(ec2.get("allowed_cidr") or cdk.get("allowed_cidr"))
    else:
        allowed_cidr = detect_public_cidr()

    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or cdk.get("hf_token")
        or ""
    )
    availability_zone = (
        args.availability_zone
        or ec2.get("availability_zone")
        or cdk.get("availability_zone")
        or ""
    )
    # Drop stale AZ from a previous region (e.g. us-west-2c after switching to Tokyo).
    if availability_zone and not availability_zone.startswith(f"{region}"):
        log(f"이전 리전 AZ 무시: {availability_zone} (현재 region={region})")
        availability_zone = ""
    if not availability_zone:
        availability_zone = (DEFAULT_AZ_ORDER.get(region) or [""])[0] or ""

    deep_update(
        config,
        {
            "project": config.get("project") or "inkling2",
            "account_id": account_id,
            "region": region,
            "model": {
                "hub_id": hub_id,
                "revision": revision,
                "served_name": hub_id.split("/")[-1],
            },
            "ec2": {
                "stack_name": stack_name,
                "instance_type": instance_type,
                "availability_zone": availability_zone,
                "vllm_port": vllm_port,
                "vllm_image": vllm_image,
                "allowed_cidr": allowed_cidr,
                "volume_size_gb": volume_size,
                "deploy_source": "deploy/ec2",
                "cdk_context": {
                    "model": hub_id,
                    "model_revision": revision,
                    "instance_type": instance_type,
                    "availability_zone": availability_zone,
                    "vllm_port": vllm_port,
                    "vllm_image": vllm_image,
                    "allowed_cidr": allowed_cidr,
                    "volume_size_gb": volume_size,
                    "tensor_parallel_size": tensor_parallel,
                    "max_model_len": max_model_len,
                    "enable_dspark": enable_dspark,
                    "gpu_memory_utilization": float(
                        cdk.get("gpu_memory_utilization", 0.9)
                    ),
                    "serve_args": cdk.get("serve_args", ""),
                    "hf_token": hf_token,
                },
            },
            "notes": {
                "destroy": "python uninstaller_ec2.py",
                "invoke": "OpenAI-compatible HTTP POST /v1/chat/completions",
                "sort_json_keys": True,
                "hardware": (
                    "Inkling-2-Mini NVFP4 needs GPU capability >= 90 "
                    "(Hopper/Blackwell). Default EC2 p5.4xlarge."
                ),
            },
        },
    )
    return config


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    log(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def ensure_node_deps() -> None:
    if not CDK_DIR.is_dir():
        raise FileNotFoundError(f"CDK 디렉터리가 없습니다: {CDK_DIR}")
    if not (CDK_DIR / "node_modules").is_dir():
        if not shutil.which("npm"):
            raise RuntimeError("npm 이 필요합니다 (Node.js 22+).")
        run(["npm", "install"], cwd=CDK_DIR)


def ensure_cdk_bootstrap(account_id: str, region: str) -> None:
    cfn = boto3.client("cloudformation", region_name=region)
    try:
        resp = cfn.describe_stacks(StackName="CDKToolkit")
        status = resp["Stacks"][0]["StackStatus"]
        if status.endswith("_COMPLETE"):
            log(f"CDK bootstrap 확인: CDKToolkit ({status})")
            return
    except ClientError:
        pass

    log("CDK bootstrap 실행...")
    env = os.environ.copy()
    env["AWS_DEFAULT_REGION"] = region
    run(
        ["npx", "cdk", "bootstrap", f"aws://{account_id}/{region}"],
        cwd=CDK_DIR,
        env=env,
    )


def cdk_context_args(config: dict[str, Any]) -> list[str]:
    cdk = config["ec2"]["cdk_context"]
    args = [
        "-c", f"model={cdk['model']}",
        "-c", f"model_revision={cdk.get('model_revision', DEFAULT_REVISION)}",
        "-c", f"instance_type={cdk['instance_type']}",
        "-c", f"availability_zone={cdk.get('availability_zone') or ''}",
        "-c", f"subnet_id={cdk.get('subnet_id') or ''}",
        "-c", f"vllm_port={cdk.get('vllm_port', DEFAULT_VLLM_PORT)}",
        "-c", f"vllm_image={cdk.get('vllm_image', DEFAULT_VLLM_IMAGE)}",
        "-c", f"allowed_cidr={cdk.get('allowed_cidr', '0.0.0.0/0')}",
        "-c", f"volume_size_gb={cdk.get('volume_size_gb', DEFAULT_VOLUME_GB)}",
        "-c", f"tensor_parallel_size={cdk.get('tensor_parallel_size', 1)}",
        "-c", f"max_model_len={cdk.get('max_model_len', 65536)}",
        "-c", f"enable_dspark={str(cdk.get('enable_dspark', False)).lower()}",
        "-c", f"gpu_memory_utilization={cdk.get('gpu_memory_utilization', 0.9)}",
        "-c", f"serve_args={cdk.get('serve_args') or ''}",
        "-c", f"hf_token={cdk.get('hf_token') or ''}",
    ]
    return args


def delete_stack_if_exists(region: str, stack_name: str) -> None:
    cfn = boto3.client("cloudformation", region_name=region)
    try:
        status = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]["StackStatus"]
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"ValidationError", "ResourceNotFoundException"}:
            return
        raise
    if status == "DELETE_COMPLETE":
        return
    log(f"기존 스택 삭제: {stack_name} ({status})")
    cfn.delete_stack(StackName=stack_name)
    waiter = cfn.get_waiter("stack_delete_complete")
    waiter.wait(
        StackName=stack_name,
        WaiterConfig={"Delay": 15, "MaxAttempts": 80},
    )
    log(f"스택 삭제 완료: {stack_name}")


def is_retryable_az_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "insufficientinstancecapacity",
        "insufficient capacity",
        "not have sufficient",
        "capacity in the availability zone",
        "not supported in your requested availability zone",
        "is not supported in your requested availability zone",
        "unsupported",
    )
    return any(m in text for m in markers)


def az_candidates(region: str, preferred: str | None) -> list[str]:
    ordered = list(DEFAULT_AZ_ORDER.get(region) or [])
    preferred_ok = preferred if preferred and preferred.startswith(f"{region}") else None
    if preferred_ok:
        ordered = [preferred_ok] + [az for az in ordered if az != preferred_ok]
    if not ordered:
        # Fallback: discover AZs from EC2 API.
        ec2 = boto3.client("ec2", region_name=region)
        ordered = [
            az["ZoneName"]
            for az in ec2.describe_availability_zones(
                Filters=[{"Name": "state", "Values": ["available"]}]
            )["AvailabilityZones"]
        ]
    # Keep only AZs that belong to this region.
    return [az for az in ordered if az.startswith(f"{region}")]


def find_default_public_subnet(region: str, availability_zone: str) -> str:
    """Return a default-VPC subnet id in the AZ with MapPublicIpOnLaunch."""
    ec2 = boto3.client("ec2", region_name=region)
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise RuntimeError(f"{region}: default VPC 가 없습니다")
    vpc_id = vpcs[0]["VpcId"]
    subnets = ec2.describe_subnets(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "availability-zone", "Values": [availability_zone]},
        ]
    )["Subnets"]
    if not subnets:
        raise RuntimeError(f"{region}/{availability_zone}: 서브넷이 없습니다")
    public = [s for s in subnets if s.get("MapPublicIpOnLaunch")]
    chosen = (public or subnets)[0]
    return chosen["SubnetId"]


def deploy_stack(config: dict[str, Any]) -> None:
    region = config["region"]
    env = os.environ.copy()
    env["AWS_DEFAULT_REGION"] = region
    env["CDK_DEFAULT_REGION"] = region
    env["CDK_DEFAULT_ACCOUNT"] = str(config["account_id"])

    cmd = [
        "npx",
        "cdk",
        "deploy",
        *cdk_context_args(config),
        "--require-approval",
        "never",
        "--progress",
        "events",
    ]
    run(cmd, cwd=CDK_DIR, env=env)


def deploy_with_az_fallback(config: dict[str, Any], *, lock_az: bool) -> None:
    """Deploy, retrying other AZs on InsufficientInstanceCapacity."""
    region = config["region"]
    stack_name = config["ec2"]["stack_name"]
    preferred = config["ec2"].get("availability_zone") or None
    candidates = [preferred] if lock_az and preferred else az_candidates(region, preferred)
    last_err: BaseException | None = None

    for idx, az in enumerate(candidates):
        config["ec2"]["availability_zone"] = az
        config["ec2"]["cdk_context"]["availability_zone"] = az
        try:
            subnet_id = find_default_public_subnet(region, az)
        except RuntimeError as exc:
            log(f"AZ {az} 서브넷 조회 실패 — 건너뜀: {exc}")
            continue
        config["ec2"]["subnet_id"] = subnet_id
        config["ec2"]["cdk_context"]["subnet_id"] = subnet_id
        save_config(config)
        log(f"AZ 시도 [{idx + 1}/{len(candidates)}]: {az} (subnet={subnet_id})")
        delete_stack_if_exists(region, stack_name)
        try:
            deploy_stack(config)
            return
        except subprocess.CalledProcessError as exc:
            last_err = exc
            # Pull CFN failure reason when possible.
            reason = ""
            try:
                cfn = boto3.client("cloudformation", region_name=region)
                events = cfn.describe_stack_events(StackName=stack_name).get(
                    "StackEvents", []
                )
                for ev in events:
                    if ev.get("ResourceStatus") == "CREATE_FAILED":
                        reason = ev.get("ResourceStatusReason") or ""
                        break
            except ClientError:
                pass
            combined = f"{exc}\n{reason}"
            if lock_az or not is_retryable_az_error(combined):
                raise
            log(f"AZ {az} 실패(용량/미지원) — 다음 AZ 시도. reason={reason[:240]}")
            continue

    if last_err:
        raise last_err
    raise RuntimeError("배포할 AZ가 없습니다")


def refresh_config_from_aws(config: dict[str, Any]) -> dict[str, Any]:
    region = config["region"]
    stack_name = config["ec2"]["stack_name"]
    cfn = boto3.client("cloudformation", region_name=region)
    stack = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}

    instance_id = outputs.get("InstanceId")
    public_ip = outputs.get("PublicIp")
    base_url = outputs.get("BaseUrl")
    if not base_url and public_ip:
        port = config["ec2"].get("vllm_port", DEFAULT_VLLM_PORT)
        base_url = f"http://{public_ip}:{port}/v1"

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    deep_update(
        config,
        {
            "ec2": {
                "stack_name": stack_name,
                "stack_arn": stack["StackId"],
                "instance_id": instance_id,
                "instance_type": outputs.get("InstanceType")
                or config["ec2"].get("instance_type"),
                "public_ip": public_ip,
                "private_ip": outputs.get("PrivateIp"),
                "vllm_port": int(outputs.get("VllmPort") or config["ec2"].get("vllm_port", 8000)),
                "base_url": base_url,
                "chat_completions_url": outputs.get("ChatCompletionsUrl")
                or (f"{base_url}/chat/completions" if base_url else None),
                "vllm_image": outputs.get("VllmImage") or config["ec2"].get("vllm_image"),
                "model_source": outputs.get("ModelSource", f"hf://{config['model']['hub_id']}"),
                "security_group_id": outputs.get("SecurityGroupId"),
                "allowed_cidr": outputs.get("AllowedCidr")
                or config["ec2"].get("allowed_cidr"),
                "availability_zone": outputs.get("AvailabilityZone")
                or config["ec2"].get("availability_zone"),
                "status": "bootstrapping",
                "deployed_at": now,
                "deploy_source": "deploy/ec2",
            }
        },
    )
    return config


def wait_for_vllm(config: dict[str, Any], *, timeout_s: int = READY_TIMEOUT_S) -> None:
    base_url = (config.get("ec2") or {}).get("base_url")
    if not base_url:
        raise RuntimeError("config.ec2.base_url 이 없습니다")
    url = f"{base_url.rstrip('/')}/models"
    log(f"vLLM 준비 대기: {url} (timeout={timeout_s}s)")
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    body = resp.read().decode("utf-8", errors="replace")
                    log(f"vLLM ready: {body[:200]}")
                    config["ec2"]["status"] = "InService"
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = str(exc)
        # Also surface instance state for debugging.
        instance_id = config["ec2"].get("instance_id")
        if instance_id:
            try:
                ec2 = boto3.client("ec2", region_name=config["region"])
                st = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0][
                    "Instances"
                ][0]["State"]["Name"]
                log(f"  … waiting (instance={st}, last_err={last_err or 'http not ready'})")
            except (ClientError, IndexError, KeyError):
                log(f"  … waiting (last_err={last_err or 'http not ready'})")
        else:
            log(f"  … waiting (last_err={last_err or 'http not ready'})")
        time.sleep(READY_POLL_S)
    config["ec2"]["status"] = "TimedOut"
    raise TimeoutError(
        f"vLLM 이 {timeout_s}s 내에 준비되지 않았습니다. "
        f"SSM으로 인스턴스 접속 후 `docker logs inkling-vllm` / "
        f"`/var/log/inkling-bootstrap.log` 를 확인하세요. last_err={last_err}"
    )


def smoke_test(config: dict[str, Any]) -> dict[str, Any] | None:
    base_url = config["ec2"]["base_url"].rstrip("/")
    hub_id = config["model"]["hub_id"]
    sort_keys = bool((config.get("notes") or {}).get("sort_json_keys", True))
    payload = {
        "model": hub_id,
        "messages": [
            {"role": "user", "content": "한 문장으로 자기소개를 해주세요."},
        ],
        "max_tokens": 64,
        "temperature": 0.2,
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    smoke = {
        "content_preview": content[:200],
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage"),
        "model": result.get("model"),
        "transport": "ec2-http",
    }
    config["smoke_test"] = smoke
    log(
        f"스모크 테스트 OK: finish_reason={smoke.get('finish_reason')}  "
        f"model={smoke.get('model')}  preview={smoke.get('content_preview')!r}"
    )
    return smoke


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inkling-2-Mini EC2 (vLLM) 설치")
    parser.add_argument("--region", default=None, help="AWS 리전 (기본: config 또는 us-west-2)")
    parser.add_argument("--model", default=None, help=f"HF Hub id (기본: {DEFAULT_MODEL})")
    parser.add_argument(
        "--instance-type",
        default=None,
        help=f"EC2 인스턴스 (기본: {DEFAULT_INSTANCE_TYPE}; ml. 접두사는 자동 제거)",
    )
    parser.add_argument(
        "--allowed-cidr",
        default=None,
        help="API(:8000) 허용 CIDR (기본: 이 머신의 공인 IP/32)",
    )
    parser.add_argument(
        "--availability-zone",
        default=None,
        help="EC2 AZ (기본: us-west-2b 우선, 용량 부족 시 자동 재시도)",
    )
    parser.add_argument(
        "--lock-az",
        action="store_true",
        help="지정/기본 AZ만 사용 (용량 부족 시 다른 AZ로 재시도하지 않음)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        help=f"vLLM 포트 (기본: {DEFAULT_VLLM_PORT})",
    )
    parser.add_argument(
        "--volume-size-gb",
        type=int,
        default=None,
        help=f"루트 EBS GiB (기본: {DEFAULT_VOLUME_GB})",
    )
    parser.add_argument(
        "--vllm-image",
        default=None,
        help=f"Docker 이미지 (기본: {DEFAULT_VLLM_IMAGE})",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (기본: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="max model length (기본: 65536)",
    )
    parser.add_argument(
        "--enable-dspark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="DSpark speculative decoding 활성화 (기본: off)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face 토큰 (또는 HF_TOKEN env)",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="배포 후 vLLM ready 대기 생략",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="배포 후 스모크 테스트 생략",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=READY_TIMEOUT_S,
        help=f"vLLM ready 대기 초 (기본: {READY_TIMEOUT_S})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(SCRIPT_DIR)

    log("=" * 60)
    log("inkling2 — EC2 설치 (vLLM / Inkling-2-Mini)")
    log("=" * 60)

    try:
        config = ensure_defaults(load_config(), args)
        save_config(config)

        log(f"project     : {config.get('project')}")
        log(f"account_id  : {config.get('account_id')}")
        log(f"region      : {config.get('region')}")
        log(f"model       : {config['model']['hub_id']}")
        log(f"instance    : {config['ec2']['instance_type']}")
        log(f"az          : {config['ec2'].get('availability_zone') or '(auto)'}")
        log(f"stack       : {config['ec2']['stack_name']}")
        log(f"image       : {config['ec2']['vllm_image']}")
        log(f"allowed_cidr: {config['ec2']['allowed_cidr']}")
        log(
            "주의: Hopper/Blackwell GPU가 필요합니다. "
            "EC2는 기동 중 시간당 과금됩니다. 실험 후 uninstaller_ec2.py 로 삭제하세요."
        )

        if not shutil.which("npx"):
            log("Error: npx(Node.js 22+)가 필요합니다.")
            return 1

        ensure_node_deps()
        ensure_cdk_bootstrap(str(config["account_id"]), config["region"])
        deploy_with_az_fallback(config, lock_az=bool(args.lock_az))
        config = refresh_config_from_aws(config)
        save_config(config)

        if not args.skip_wait:
            wait_for_vllm(config, timeout_s=args.ready_timeout)
            save_config(config)

        if not args.skip_smoke:
            if args.skip_wait and config["ec2"].get("status") != "InService":
                log("skip-wait 상태라 스모크를 건너뜁니다. 준비 후 수동 호출하세요.")
            else:
                smoke_test(config)
                save_config(config)

    except NoCredentialsError:
        log("Error: AWS 자격 증명을 찾을 수 없습니다.")
        return 1
    except subprocess.CalledProcessError as exc:
        log(f"Error: 명령 실패 (exit {exc.returncode})")
        return exc.returncode or 1
    except (ClientError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
        log(f"Error: {exc}")
        return 1

    log("\n" + "=" * 60)
    log("설치 완료")
    log("=" * 60)
    log(f"instance : {config['ec2'].get('instance_id')}")
    log(f"status   : {config['ec2'].get('status')}")
    log(f"base_url : {config['ec2'].get('base_url')}")
    log(f"config   : {CONFIG_PATH}")
    log("테스트    : python run_inkling2.py")
    log("삭제      : python uninstaller_ec2.py --yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
