#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { GeoFMDemoStack } from '../lib/geofm-demo-stack';

const app = new cdk.App();

// Get environment from context or default to 'dev'
const envName = app.node.tryGetContext('env') || 'dev';

const geoFMDemoStack = new GeoFMDemoStack(app, `GeoFMDemoStack${envName}`, {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
  stackName: `geofm-demo-${envName}`,
  envName: envName
});

