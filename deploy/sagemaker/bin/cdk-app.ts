#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { InklingEndpointStack, endpointNameFor } from "../lib/inkling-endpoint-stack";

const app = new App();
const endpointName = endpointNameFor(app.node.tryGetContext("model"));

new InklingEndpointStack(app, endpointName, {
  endpointName,
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
