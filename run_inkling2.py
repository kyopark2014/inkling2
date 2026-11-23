#!/usr/bin/env python3
"""Inkling-2-Mini Chat Completions 예제를 실행합니다.

config.json 에서 배포 정보를 읽고 testcases.json cases 를 호출합니다.
- ec2.base_url 이 있으면 OpenAI-compatible HTTP (/v1/chat/completions)
- 아니면 sagemaker.endpoint_name 으로 InvokeEndpoint

실행:
  python run_inkling2.py
  python run_inkling2.py --only 1,4,8
  python run_inkling2.py --list
  python run_inkling2.py --config ./config.json --cases ./testcases.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_CASES = ROOT / "testcases.json"


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config 파일이 없습니다: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config 루트는 object여야 합니다: {path}")
    return data


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"testcases 파일이 없습니다: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = data.get("cases")
    else:
        raise ValueError(f"지원하지 않는 testcases 형식: {path}")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"cases 가 비어 있거나 배열이 아닙니다: {path}")
    return cases


def preview_messages(messages: Any, limit: int = 240) -> str:
    text = json.dumps(messages, ensure_ascii=False)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def format_message(message: dict[str, Any] | None) -> list[str]:
    if not message:
        return ["  result: (빈 message)"]
    lines: list[str] = []
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    tool_calls = message.get("tool_calls")
    if content:
        lines.append(f"  content: {content}")
    if reasoning:
        preview = reasoning if len(str(reasoning)) < 400 else str(reasoning)[:397] + "..."
        lines.append(f"  reasoning: {preview}")
    if tool_calls:
        lines.append(f"  tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}")
    if not lines:
        lines.append(f"  message: {json.dumps(message, ensure_ascii=False)}")
    return lines


def invoke_endpoint(
    *,
    runtime: Any,
    endpoint_name: str,
    payload: dict[str, Any],
    sort_keys: bool = True,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys).encode("utf-8")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=body,
    )
    raw = response["Body"].read()
    return json.loads(raw.decode("utf-8"))


def invoke_openai_http(
    *,
    base_url: str,
    payload: dict[str, Any],
    sort_keys: bool = True,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as decode_exc:
            raise RuntimeError(f"HTTP {exc.code}: {raw[:500]}") from decode_exc


def resolve_transport(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ('ec2'|'sagemaker', meta). Prefer live EC2 when base_url is set."""
    ec2 = config.get("ec2") if isinstance(config.get("ec2"), dict) else {}
    sm = config.get("sagemaker") if isinstance(config.get("sagemaker"), dict) else {}
    status = str(ec2.get("status") or "").lower()
    if ec2.get("base_url") and status != "deleted":
        return "ec2", ec2
    if sm.get("endpoint_name") and str(sm.get("endpoint_status") or "") != "Deleted":
        return "sagemaker", sm
    if sm.get("endpoint_name"):
        return "sagemaker", sm
    raise ValueError(
        "배포 정보가 없습니다. python installer_ec2.py 또는 python installer.py 를 먼저 실행하세요."
    )


def run_examples(
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    selected: list[int] | None = None,
    *,
    cases_path: Path | None = None,
) -> int:
    notes = config.get("notes") or {}
    model = config.get("model") or {}
    region = config.get("region") or "us-west-2"
    sort_keys = bool(notes.get("sort_json_keys", True))

    try:
        transport, meta = resolve_transport(config)
    except ValueError as exc:
        print(str(exc))
        return 1

    runtime = None
    endpoint_name = None
    base_url = None
    if transport == "ec2":
        base_url = meta.get("base_url")
        print(
            f"transport=ec2  base_url={base_url}  region={region}  "
            f"instance={meta.get('instance_type')}  "
            f"model={model.get('served_name') or model.get('hub_id')}"
        )
    else:
        endpoint_name = meta.get("endpoint_name")
        runtime = boto3.client("sagemaker-runtime", region_name=region)
        print(
            f"transport=sagemaker  endpoint={endpoint_name}  region={region}  "
            f"instance={meta.get('instance_type')}  "
            f"model={model.get('served_name') or model.get('hub_id')}"
        )
    if cases_path is not None:
        print(f"cases={cases_path} ({len(cases)}개)")
    print("=" * 72)

    failures = 0
    timings_ms: list[float] = []
    total_started = time.perf_counter()

    for idx, example in enumerate(cases, start=1):
        if selected is not None and idx not in selected:
            continue

        request = example.get("request")
        if not isinstance(request, dict):
            print(f"\n[{idx:02d}] {example.get('name')} — request 가 object가 아님")
            failures += 1
            continue

        print(f"\n[{idx:02d}] {example.get('name')}")
        print(f"     {example.get('description')}")
        if example.get("expect"):
            print(f"  expect: {example['expect']}")
        print(f"  messages: {preview_messages(request.get('messages'))}")

        started = time.perf_counter()
        try:
            if transport == "ec2":
                assert base_url is not None
                result = invoke_openai_http(
                    base_url=base_url,
                    payload=request,
                    sort_keys=sort_keys,
                )
            else:
                assert runtime is not None and endpoint_name is not None
                result = invoke_endpoint(
                    runtime=runtime,
                    endpoint_name=endpoint_name,
                    payload=request,
                    sort_keys=sort_keys,
                )
        except (ClientError, BotoCoreError, json.JSONDecodeError, RuntimeError, OSError) as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings_ms.append(elapsed_ms)
            print(f"  ERROR: {exc}")
            print(f"  실행시간: {elapsed_ms:.2f} ms (클라이언트)")
            failures += 1
            continue
        except Exception as exc:  # noqa: BLE001 - runner script
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings_ms.append(elapsed_ms)
            print(f"  ERROR: {exc}")
            print(f"  실행시간: {elapsed_ms:.2f} ms (클라이언트)")
            failures += 1
            continue

        elapsed_ms = (time.perf_counter() - started) * 1000
        timings_ms.append(elapsed_ms)

        if "error" in result and "choices" not in result:
            print(f"  ERROR body: {json.dumps(result, ensure_ascii=False)}")
            failures += 1
        else:
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            for line in format_message(message):
                print(line)
            print(f"  finish_reason: {choice.get('finish_reason')}")
            usage = result.get("usage") or {}
            print(
                f"  usage: prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} "
                f"total={usage.get('total_tokens')}"
            )
            print(f"  model={result.get('model')}")

        print(f"  실행시간: {elapsed_ms:.2f} ms (클라이언트)")

    total_ms = (time.perf_counter() - total_started) * 1000
    print("\n" + "=" * 72)
    if timings_ms:
        avg_ms = sum(timings_ms) / len(timings_ms)
        print(
            f"실행시간 요약: 총 {total_ms:.2f} ms | "
            f"예제 {len(timings_ms)}개 | "
            f"평균 {avg_ms:.2f} ms | "
            f"최소 {min(timings_ms):.2f} ms | "
            f"최대 {max(timings_ms):.2f} ms"
        )
    if failures:
        print(f"완료: 실패 {failures}건")
        return 1
    print("완료: 모든 예제 호출 성공")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="config.json + testcases.json → EC2 HTTP 또는 SageMaker endpoint 실행"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"config 경로 (기본: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help=f"testcases 경로 (기본: {DEFAULT_CASES})",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="실행할 예제 번호 (쉼표 구분, 예: 1,4,8)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="예제 목록만 출력",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"testcases 로드 실패: {exc}")
        return 1

    if args.list:
        print(f"cases file: {args.cases}")
        for idx, example in enumerate(cases, start=1):
            print(f"{idx:02d}  {example.get('name')}  —  {example.get('description')}")
        return 0

    selected = None
    if args.only:
        selected = [int(x.strip()) for x in args.only.split(",") if x.strip()]

    try:
        config = load_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"config 로드 실패: {exc}")
        return 1

    return run_examples(config, cases, selected, cases_path=args.cases)


if __name__ == "__main__":
    sys.exit(main())
