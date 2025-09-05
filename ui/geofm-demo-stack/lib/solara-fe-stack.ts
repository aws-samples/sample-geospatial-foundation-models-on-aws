import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecsPatterns from 'aws-cdk-lib/aws-ecs-patterns';
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Duration, RemovalPolicy } from 'aws-cdk-lib';

import * as iam from 'aws-cdk-lib/aws-iam';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import { Construct } from 'constructs';
import * as path from 'path';


export interface SolaraFEStackProps extends cdk.NestedStackProps {
  // cloudFrontDistribution: cloudfront.IDistribution;
  customHeaderName: string;
  customHeaderValue: string;
  envName: string;
  geoTiffBucket: s3.IBucket;
  // userPool: cognito.IUserPool;
  // authorizerFunction: cloudfront.experimental.EdgeFunction;
}

export class SolaraFEStack extends cdk.NestedStack {
  public readonly vpc: ec2.Vpc;
  public readonly solaraOriginLBDnsName: string;
  public readonly publicLoadBalancer: elbv2.ApplicationLoadBalancer;
  public readonly solaraSGs: string[];

  constructor(scope: Construct, id: string, props: SolaraFEStackProps) {
    super(scope, id, props);

    const S3_BUCKET_NAME = "gfm-demo-bucket"
    /////////////////// Create NEW VPC

    // Create a new VPC with public, private with egress, and private isolated subnets
    this.vpc = new ec2.Vpc(this, "geofm-demo-ecs-vpc", {
      vpcName: `geofm-demo-ecs-vpc-${props.envName}`,
      maxAzs: 2,
      subnetConfiguration: [
        {
          name: `public-subnet-${props.envName}`,
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: `private-subnet-ecs-${props.envName}`,
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
        {
          name: `isolated-subnet-rds-${props.envName}`,
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
      natGateways: 1,
    });


    // create a flow log to be associated with VPC and that sends logs in Cloudwatch
    // this is a best practice but not strictly required
    this.vpc.addFlowLog('FlowLogCloudWatchGeoFMDemo', {
      destination: ec2.FlowLogDestination.toCloudWatchLogs(),
    });

    // Create a security group for the application
    const appSG = new ec2.SecurityGroup(this, "solaraAppSecurityGroup", {
      vpc: this.vpc,
      description: "GeoFMDemo is internet exposed",
      allowAllOutbound: true,
    });

    // Allow inbound traffic on port 8000 (Solara's port used in Docker)
    appSG.addIngressRule(
        ec2.Peer.anyIpv4(),
        ec2.Port.tcp(8000),
        'Allow inbound traffic to Solara'
    );

    // create a bucket for enabling load balancer and distribution logs logs
    const logBucket = new s3.Bucket(this, 'geofmDemoLogsBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: false,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      serverAccessLogsPrefix: `bucket-access-logs-${props.envName}`,
      objectOwnership: s3.ObjectOwnership.OBJECT_WRITER
    });

    // Create a public Application Load Balancer
    this.publicLoadBalancer = new elbv2.ApplicationLoadBalancer(this, "geofmDemoPublicLoadBalancer", {
      vpc: this.vpc,
      internetFacing: true,
      vpcSubnets: { subnets: this.vpc.publicSubnets },
      securityGroup: appSG
    });
    
    // enabling logs
    this.publicLoadBalancer.logAccessLogs(logBucket, "ApplicationLoadBalancerLogs");





    /////////////////// Rest of ECS

    // Create ECS cluster
    const cluster = new ecs.Cluster(this, 'SolaraCluster', {
      vpc: this.vpc,
    });

    // Create Fargate task definition
    const taskDefinition = new ecs.FargateTaskDefinition(this, 'SolaraTaskDefinition', {
      memoryLimitMiB: 16384,
      cpu: 8192,
    });

    // Add container to task definition
    const container = taskDefinition.addContainer('SolaraContainer', {
      image: ecs.ContainerImage.fromAsset(path.join(__dirname, '../solara-fe'), {
        platform: cdk.aws_ecr_assets.Platform.LINUX_AMD64
      }),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: 'SolaraBackend' }),
      environment: {
        'SOLARA_APP': 'app:app',
        'SOLARA_ASSETS_PREFIX': '/solara/',
        "GEOTIFF_BUCKET_URL": props.geoTiffBucket.urlForObject()
      }
    });

    container.addPortMappings({
      containerPort: 8000,
    });

    // Create Fargate service
    const fargateService = new ecsPatterns.ApplicationLoadBalancedFargateService(this, 'SolaraService', {
      cluster,
      taskDefinition,
      publicLoadBalancer: true,
      desiredCount: 1,
      listenerPort: 8000,
      securityGroups: [appSG],
      taskSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
      }
    });

    // Add custom header to the ALB
    fargateService.targetGroup.configureHealthCheck({
      path: '/readyz', // Solara health endpoint when runnning with --production
      timeout: cdk.Duration.seconds(5),
      unhealthyThresholdCount: 2,
      healthyThresholdCount: 2,
      interval: cdk.Duration.seconds(6),
    });

    this.solaraOriginLBDnsName = fargateService.loadBalancer.loadBalancerDnsName;

    // TODO add Listener Rule
    // Get the listener from the Fargate service
    const listener = fargateService.listener;

    // Add a listener rule
    const rule = listener.addAction('CustomHeaderRule', {
      priority: 10,
      conditions: [
        elbv2.ListenerCondition.httpHeader(props.customHeaderName, [props.customHeaderValue])
      ],
      action: elbv2.ListenerAction.forward([fargateService.targetGroup]),
    });

    //change the default listener rule to return fixed response
    listener.addAction('DefaultAction', {
    action: elbv2.ListenerAction.fixedResponse(403, { contentType: 'text/plain', messageBody: 'Forbidden' })  
    });
    
    fargateService.targetGroup.enableCookieStickiness(Duration.days(7));

    // Allow the ECS task to retrieve S3 objects from bucket wec-sample-data, key wec_regulation/*
    fargateService.taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        resources: ["*"],
        actions: [
          "s3:Get*",
          "s3:List*",
          "s3:Put*",
          "aoss:List*",
          "aoss:Batch*",
          "aoss:Get*",
          "aoss:Search*",
          "aoss:APIAccessAll"
        ],
      })
    );

    fargateService.taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        resources: [`arn:aws:s3:::${S3_BUCKET_NAME}`,`arn:aws:s3:::${S3_BUCKET_NAME}/*`],
        actions: [
          "s3:Get*",
          "s3:List*",
          "s3:Put*"
        ],
      })
    );

    this.solaraSGs = fargateService.loadBalancer.connections.securityGroups.map(sg => sg.securityGroupId);

    // Output the ALB DNS name
    new cdk.CfnOutput(this, 'LoadBalancerDNS', {
      value: fargateService.loadBalancer.loadBalancerDnsName,
    });

    // Add this to your stack
    new cdk.CfnOutput(this, 'SecurityGroupIds', {
      value: JSON.stringify(this.solaraSGs),
      description: 'Security Group IDs used by Fargate service'
    });
  }
}