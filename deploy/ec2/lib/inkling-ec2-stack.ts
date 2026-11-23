import * as fs from "fs";
import * as path from "path";
import {
  CfnOutput,
  RemovalPolicy,
  Stack,
  StackProps,
  Tags,
} from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

/**
 * Stack / resource name from Hugging Face Hub id:
 * thinkingmachines/Inkling-2-Mini -> inkling-2-mini-ec2
 */
export function stackNameFor(model: string): string {
  const name = (model.split("/").pop() ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!/^[a-z0-9](-*[a-z0-9]){0,50}$/.test(name)) {
    throw new Error(`${model}: cannot make an EC2 stack name from it`);
  }
  return `${name}-ec2`;
}

/** Strip accidental SageMaker `ml.` prefix. */
export function normalizeEc2InstanceType(raw: string): string {
  const t = raw.trim();
  if (t.startsWith("ml.")) {
    return t.slice(3);
  }
  return t;
}

export function assertGpuInstanceType(instanceType: string): void {
  const family = instanceType.split(".")[0];
  if (!/^(g|p)\d/.test(family)) {
    throw new Error(`${instanceType}: Inkling-2-Mini requires a GPU instance (g*/p*)`);
  }
}

const DLAMI_SSM =
  "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id";

export interface InklingEc2StackProps extends StackProps {
  readonly stackNameHint: string;
}

/**
 * EC2 host running the public vLLM day-0 image for Inkling-2-Mini.
 * Model weights are pulled from the Hugging Face Hub at container startup.
 */
export class InklingEc2Stack extends Stack {
  constructor(scope: Construct, id: string, props: InklingEc2StackProps) {
    super(scope, id, props);

    const ctx = (key: string) => this.node.tryGetContext(key);
    const modelId = String(ctx("model"));
    const instanceType = normalizeEc2InstanceType(String(ctx("instance_type") || "p5.4xlarge"));
    assertGpuInstanceType(instanceType);

    const vllmPort = Number(ctx("vllm_port") || 8000);
    const volumeSize = Number(ctx("volume_size_gb") || 250);
    const allowedCidr = String(ctx("allowed_cidr") || "0.0.0.0/0");
    const tensorParallel = String(ctx("tensor_parallel_size") || "1");
    const maxModelLen = String(ctx("max_model_len") || "65536");
    const gpuUtil = String(ctx("gpu_memory_utilization") || "0.90");
    const enableDspark = String(ctx("enable_dspark") || "false").toLowerCase() === "true";
    const serveArgs = String(ctx("serve_args") || "");
    const vllmImage = String(ctx("vllm_image") || "vllm/vllm-openai:inkling-2-mini");
    const hfToken = String(ctx("hf_token") || "");
    const speculative = String(
      ctx("speculative_config") ||
        '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}',
    );

    const availabilityZone = String(ctx("availability_zone") || "").trim();

    const vpc = ec2.Vpc.fromLookup(this, "Vpc", { isDefault: true });

    const sg = new ec2.SecurityGroup(this, "SecurityGroup", {
      vpc,
      description: `inkling2 vLLM OpenAI API (:${vllmPort})`,
      allowAllOutbound: true,
    });
    sg.addIngressRule(
      ec2.Peer.ipv4(allowedCidr),
      ec2.Port.tcp(vllmPort),
      `vLLM OpenAI API from ${allowedCidr}`,
    );

    const role = new iam.Role(this, "InstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"),
        iam.ManagedPolicy.fromAwsManagedPolicyName("CloudWatchAgentServerPolicy"),
      ],
    });

    const machineImage = ec2.MachineImage.fromSsmParameter(DLAMI_SSM, {
      os: ec2.OperatingSystemType.LINUX,
    });

    const userData = ec2.UserData.forLinux();
    const bootstrapPath = path.join(__dirname, "..", "scripts", "bootstrap.sh");
    const bootstrap = fs.readFileSync(bootstrapPath, "utf8");

    // Escape for single-quoted heredoc body that itself contains quotes carefully:
    // write env + script as separate files via base64 to avoid shell injection.
    const envBody = [
      `MODEL_ID=${shellQuote(modelId)}`,
      `VLLM_IMAGE=${shellQuote(vllmImage)}`,
      `VLLM_PORT=${vllmPort}`,
      `TENSOR_PARALLEL_SIZE=${shellQuote(tensorParallel)}`,
      `MAX_MODEL_LEN=${shellQuote(maxModelLen)}`,
      `GPU_MEMORY_UTILIZATION=${shellQuote(gpuUtil)}`,
      `ENABLE_DSPARK=${enableDspark ? "1" : "0"}`,
      `SPECULATIVE_CONFIG=${shellQuote(speculative)}`,
      `SERVE_ARGS=${shellQuote(serveArgs)}`,
      `HF_TOKEN=${shellQuote(hfToken)}`,
    ].join("\n");

    userData.addCommands(
      "set -euxo pipefail",
      `cat >/etc/inkling.env <<'INKLING_ENV_EOF'\n${envBody}\nINKLING_ENV_EOF`,
      `cat >/opt/inkling-bootstrap.sh <<'INKLING_BOOT_EOF'\n${bootstrap}\nINKLING_BOOT_EOF`,
      "chmod +x /opt/inkling-bootstrap.sh",
      "nohup /opt/inkling-bootstrap.sh >/var/log/inkling-bootstrap.nohup.log 2>&1 &",
    );

    if (availabilityZone && !availabilityZone.startsWith(this.region)) {
      throw new Error(
        `availability_zone=${availabilityZone} does not match stack region ${this.region}`,
      );
    }

    // Prefer an explicit subnet id from the installer. Default-VPC lookups via
    // SubnetType.PUBLIC are unreliable across accounts/regions.
    const subnetId = String(ctx("subnet_id") || "").trim();
    let subnetSelection: ec2.SubnetSelection;
    if (subnetId) {
      if (!availabilityZone) {
        throw new Error("subnet_id requires availability_zone");
      }
      const subnet = ec2.Subnet.fromSubnetAttributes(this, "SelectedSubnet", {
        subnetId,
        availabilityZone,
      });
      subnetSelection = { subnets: [subnet] };
    } else {
      subnetSelection = availabilityZone
        ? { subnetType: ec2.SubnetType.PUBLIC, availabilityZones: [availabilityZone] }
        : { subnetType: ec2.SubnetType.PUBLIC };
      if (vpc.selectSubnets(subnetSelection).subnets.length === 0) {
        throw new Error(
          `No public subnets found in default VPC` +
            (availabilityZone ? ` for AZ ${availabilityZone}` : "") +
            `. Pass -c subnet_id=... from installer_ec2.py`,
        );
      }
    }

    const instance = new ec2.Instance(this, "Instance", {
      vpc,
      vpcSubnets: subnetSelection,
      instanceType: new ec2.InstanceType(instanceType),
      machineImage,
      securityGroup: sg,
      role,
      userData,
      blockDevices: [
        {
          deviceName: "/dev/xvda",
          volume: ec2.BlockDeviceVolume.ebs(volumeSize, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
            deleteOnTermination: true,
          }),
        },
      ],
      requireImdsv2: true,
      // Default-VPC subnets already have MapPublicIpOnLaunch=true. Setting
      // associatePublicIpAddress requires SubnetType.PUBLIC which imported
      // subnet ids do not satisfy in CDK.
      detailedMonitoring: false,
    });
    // Prefer replace on user-data changes when redeploying.
    instance.instance.applyRemovalPolicy(RemovalPolicy.DESTROY);
    Tags.of(instance).add("Name", props.stackNameHint);
    Tags.of(instance).add("Project", "inkling2");
    Tags.of(instance).add("Component", "vllm");
    if (availabilityZone) {
      Tags.of(instance).add("AvailabilityZone", availabilityZone);
    }

    new CfnOutput(this, "InstanceId", { value: instance.instanceId });
    new CfnOutput(this, "PublicIp", { value: instance.instancePublicIp });
    new CfnOutput(this, "PrivateIp", { value: instance.instancePrivateIp });
    new CfnOutput(this, "InstanceType", { value: instanceType });
    new CfnOutput(this, "AvailabilityZone", {
      value: availabilityZone || instance.instanceAvailabilityZone,
    });
    new CfnOutput(this, "VllmPort", { value: String(vllmPort) });
    new CfnOutput(this, "BaseUrl", {
      value: `http://${instance.instancePublicIp}:${vllmPort}/v1`,
    });
    new CfnOutput(this, "ChatCompletionsUrl", {
      value: `http://${instance.instancePublicIp}:${vllmPort}/v1/chat/completions`,
    });
    new CfnOutput(this, "ModelSource", { value: `hf://${modelId}` });
    new CfnOutput(this, "VllmImage", { value: vllmImage });
    new CfnOutput(this, "SecurityGroupId", { value: sg.securityGroupId });
    new CfnOutput(this, "AllowedCidr", { value: allowedCidr });
  }
}

function shellQuote(value: string): string {
  // Values land inside a single-quoted heredoc; keep them literal lines.
  return value.replace(/\n/g, " ").trim();
}
