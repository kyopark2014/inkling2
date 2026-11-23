#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { InklingEc2Stack, stackNameFor } from "../lib/inkling-ec2-stack";

const app = new App();
const model = String(app.node.tryGetContext("model") || "thinkingmachines/Inkling-2-Mini");
const name = stackNameFor(model);

new InklingEc2Stack(app, name, {
  stackNameHint: name,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});
