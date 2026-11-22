import * as path from "path";
import { CfnOutput, IgnoreMode, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as sagemaker from "aws-cdk-lib/aws-sagemaker";
import { Construct } from "constructs";
import { ContainerImageBuild } from "@cdklabs/deploy-time-build";

/**
 * SageMaker endpoint name from Hugging Face Hub id:
 * thinkingmachines/Inkling-2-Mini -> inkling-2-mini
 */
export function endpointNameFor(model: string): string {
  const name = (model.split("/").pop() ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!/^[a-z0-9](-*[a-z0-9]){0,62}$/.test(name)) {
    throw new Error(`${model}: cannot make a SageMaker endpoint name from it`);
  }
  return name;
}

const GPU_AMI_VERSION = "al2023-ami-sagemaker-inference-gpu-4-1";

export function deviceFor(instanceType: string): "cuda" | "cpu" {
  const family = instanceType.replace(/^ml\./, "").split(".")[0];
  if (/^(inf|trn)\d/.test(family)) {
    throw new Error(`${instanceType}: Inferentia and Trainium are not supported`);
  }
  if (/^[a-z]+\d+[a-z]*g[a-z]*$/.test(family) && !/^g\d/.test(family) && !/^p\d/.test(family)) {
    throw new Error(`${instanceType}: Graviton is not supported`);
  }
  return /^(g|p)\d/.test(family) ? "cuda" : "cpu";
}

export function hasInstanceStorage(instanceType: string): boolean {
  const family = instanceType.replace(/^ml\./, "").split(".")[0];
  return (
    /^(g4dn|g5|g6|g6e|g7|g7e|p4d|p4de|p5|p5e|p5en|p6)/.test(family) ||
    /^[a-z]+\d+[a-z]*d[a-z]*$/.test(family)
  );
}

export function instancePoolsFor(instanceType: string, fallbacks: unknown): string[] {
  const list = Array.isArray(fallbacks) ? fallbacks.map(String) : String(fallbacks ?? "").split(",");
  const pools = [instanceType, ...list.map((t) => t.trim()).filter(Boolean)];
  for (const t of pools) {
    if (!/^ml\.[a-z0-9-]+\.[a-z0-9]+$/.test(t)) {
      throw new Error(`${t}: not a SageMaker instance type such as ml.p5.4xlarge`);
    }
  }
  if (pools.length > 5) {
    throw new Error(`fallback_instance_types: at most 4, got ${pools.length - 1}`);
  }
  if (new Set(pools).size !== pools.length) {
    throw new Error(`fallback_instance_types must be distinct: ${pools.join(", ")}`);
  }
  for (const t of pools.slice(1)) {
    if (deviceFor(t) !== deviceFor(instanceType)) {
      throw new Error(`fallback_instance_types: ${t} device differs from ${instanceType}`);
    }
    if (hasInstanceStorage(t) !== hasInstanceStorage(instanceType)) {
      throw new Error(`fallback_instance_types: ${t} and ${instanceType} differ in instance storage`);
    }
  }
  return pools;
}

export interface InklingEndpointStackProps extends StackProps {
  readonly endpointName: string;
}

/**
 * Deploys Inkling-2-Mini with a vLLM BYOC image.
 *
 * Model weights are pulled from the Hugging Face Hub at container startup
 * (`SM_VLLM_MODEL=<hub id>`). Pre-syncing ~20GB via CodeBuild was too slow /
 * timeout-prone for the default 60-minute NodejsBuild window.
 */
export class InklingEndpointStack extends Stack {
  constructor(scope: Construct, id: string, props: InklingEndpointStackProps) {
    super(scope, id, props);
    const endpointName = props.endpointName;
    const ctx = (key: string) => this.node.tryGetContext(key);
    const modelId: string = ctx("model");
    const instanceType: string = ctx("instance_type");
    const device = deviceFor(instanceType);

    if (device !== "cuda") {
      throw new Error(`${instanceType}: Inkling-2-Mini requires a GPU instance`);
    }

    const role = new iam.Role(this, "ExecutionRole", {
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
    });
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        resources: [`arn:${this.partition}:logs:${this.region}:${this.account}:log-group:/aws/sagemaker/*`],
      }),
    );
    role.addToPolicy(new iam.PolicyStatement({ actions: ["cloudwatch:PutMetricData"], resources: ["*"] }));

    const imageUri = this.buildImage(role);
    const instancePools = instancePoolsFor(instanceType, ctx("fallback_instance_types"));
    const tensorParallel = String(ctx("tensor_parallel_size") || "1");
    const maxModelLen = String(ctx("max_model_len") || "65536");
    const enableDspark = String(ctx("enable_dspark") || "false").toLowerCase() === "true" ? "1" : "0";
    const extraArgs = String(ctx("serve_args") || "");
    const hfToken = String(ctx("hf_token") || "");

    const model = new sagemaker.CfnModel(this, "Model", {
      executionRoleArn: role.roleArn,
      primaryContainer: {
        image: imageUri,
        environment: {
          SM_VLLM_MODEL: modelId,
          SERVED_MODEL_NAME: modelId,
          SM_VLLM_TENSOR_PARALLEL_SIZE: tensorParallel,
          SM_VLLM_MAX_MODEL_LEN: maxModelLen,
          SM_VLLM_ENABLE_DSPARK: enableDspark,
          SM_VLLM_GPU_MEMORY_UTILIZATION: String(ctx("gpu_memory_utilization") || "0.90"),
          ...(extraArgs ? { SM_VLLM_EXTRA_ARGS: extraArgs } : {}),
          ...(hfToken ? { HF_TOKEN: hfToken } : {}),
          // Hub download at startup (not offline S3 sync).
          HF_HUB_OFFLINE: "0",
          TRANSFORMERS_OFFLINE: "0",
        },
      },
    });
    model.node.addDependency(role);

    const config = new sagemaker.CfnEndpointConfig(this, "EndpointConfig", {
      productionVariants: [
        {
          variantName: "AllTraffic",
          modelName: model.attrModelName,
          ...(instancePools.length > 1
            ? { instancePools: instancePools.map((t, i) => ({ instanceType: t, priority: i + 1 })) }
            : { instanceType }),
          initialInstanceCount: Number(ctx("instance_count") || 1),
          initialVariantWeight: 1,
          inferenceAmiVersion: ctx("inference_ami_version") || GPU_AMI_VERSION,
          // Room for HF cache (~20GB weights + KV) when instance has no local NVMe.
          volumeSizeInGb: hasInstanceStorage(instanceType) ? undefined : 128,
          containerStartupHealthCheckTimeoutInSeconds: 3600,
        },
      ],
    });

    const endpoint = new sagemaker.CfnEndpoint(this, "Endpoint", {
      endpointName,
      endpointConfigName: config.attrEndpointConfigName,
    });

    new CfnOutput(this, "EndpointName", { value: endpoint.attrEndpointName });
    new CfnOutput(this, "Device", { value: device });
    new CfnOutput(this, "ImageUri", { value: imageUri });
    new CfnOutput(this, "ModelSource", { value: `hf://${modelId}` });
    new CfnOutput(this, "InvokeUrl", {
      value: `https://runtime.sagemaker.${this.region}.${this.urlSuffix}/endpoints/${endpointName}/invocations`,
    });
  }

  private buildImage(role: iam.IRole): string {
    const image = new ContainerImageBuild(this, "Image", {
      directory: path.join(__dirname, ".."),
      file: "Dockerfile",
      exclude: ["*", "!Dockerfile", "!sagemaker_entrypoint.py", "!serve.sh"],
      ignoreMode: IgnoreMode.DOCKER,
      platform: ecrAssets.Platform.LINUX_AMD64,
    });
    image.repository.grantPull(role);
    return image.imageUri;
  }
}
