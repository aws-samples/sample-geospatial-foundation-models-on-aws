import * as cdk from 'aws-cdk-lib';
import { CfnOutput, Duration, NestedStack, NestedStackProps, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as path from 'path';

import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as lambdaNodeJS from 'aws-cdk-lib/aws-lambda-nodejs'
import * as iam from 'aws-cdk-lib/aws-iam';
import { RetentionDays } from 'aws-cdk-lib/aws-logs';

import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import { CfnWebACL, CfnWebACLAssociation } from 'aws-cdk-lib/aws-wafv2';


export interface BackendStackProps extends NestedStackProps {
    readonly userPool: cognito.IUserPool;
    readonly customHeaderName: string;
    readonly customHeaderValue: string;
    readonly userPoolClientId: string;
    readonly envName: string;
}

export class BackendStack extends NestedStack {
    public readonly tilesApi: apigw.RestApi;
    public readonly s3VpcPointId: string;
    public readonly vpc: ec2.Vpc;

    constructor(scope: Construct, id: string, props: BackendStackProps) {
        super(scope, id, props);
  
        const apiAuthorizerFunction = new lambdaNodeJS.NodejsFunction(this, 'auth-lambda', {
            entry: path.join(__dirname, '../lambdas/api-auth/index.js'),
            handler: 'handler',
            runtime: lambda.Runtime.NODEJS_18_X,
            environment: {
                HEADER_NAME: props.customHeaderName,
                HEADER_VALUE: props.customHeaderValue
            }
        });


        // Titiler part
        this.vpc = new ec2.Vpc(this, 'vpc', {
            vpcName: `geofm-demo-vpc-${props.envName}`,
            cidr: '10.42.0.0/16',
            natGateways: 1,
            maxAzs: 3,
            subnetConfiguration: [
                {
                    cidrMask: 24,
                    subnetType: ec2.SubnetType.PUBLIC,
                    name: 'public'
                },
                {
                    cidrMask: 24,
                    subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    name: 'private'
                },
                {
                    cidrMask: 28,
                    subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
                    name: 'isolated'
                }
            ]
        });
        const vpc = this.vpc;

        const s3AccessPoint = this.vpc.addGatewayEndpoint('s3', { service: ec2.GatewayVpcEndpointAwsService.S3 });
        s3AccessPoint.addToPolicy(new iam.PolicyStatement({
            principals: [ new iam.StarPrincipal() ],
            actions: ['s3:GetObject'],
            resources: ['*']            
        }));
        this.s3VpcPointId = s3AccessPoint.vpcEndpointId;
        
        const tilTilerSG = new ec2.SecurityGroup(this, 'lambdaTilTilerSecurityGroup', {
            vpc,
            allowAllOutbound: true,
        });

        tilTilerSG.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.allTcp(), 'Allow all inbound from subnet');
        const redisSG = new ec2.SecurityGroup(this, 'redisSecurityGroup', {
            vpc,
            allowAllOutbound: true
        });
        redisSG.addIngressRule(tilTilerSG, ec2.Port.tcp(6379), 'Allow Redis connection');

        const subnetIds = this.vpc.isolatedSubnets.map((subnet) => (subnet.subnetId));
        const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, 'redisSubnetGroup', {
            subnetIds,
            description: 'subnet group for geofm-demo redis'
        });

        const redisCluster = new elasticache.CfnCacheCluster(
            this,
            "redisCluster", {
                clusterName: `geofm-demo-redis-${props.envName}`,
                engine: "redis",
                cacheNodeType: "cache.t4g.medium",
                cacheSubnetGroupName: redisSubnetGroup.ref,
                vpcSecurityGroupIds: [redisSG.securityGroupId],
                numCacheNodes: 1
            }
        );

        new CfnOutput(this, 'redisEndpoint', {
            value: `redis://${redisCluster.attrRedisEndpointAddress}:${redisCluster.attrRedisEndpointPort}`,
            description: 'Redis Cluster Endpoint'
        });

        const tilTilerFunction = new lambda.Function(this, "TilTilerFunction", {
            functionName: `geofm-demo-tiltiler-${props.envName}`,
            vpc,
            vpcSubnets: {
                subnets: vpc.privateSubnets
            },
            securityGroups: [tilTilerSG],
            runtime: lambda.Runtime.PYTHON_3_9,
            memorySize: 1024,
            code: lambda.Code.fromDockerBuild(path.join(__dirname, '..', 'lambdas', 'tiltiler')),
            handler: "handler.handler",
            environment: {
                "CACHE_ENDPOINT": `redis://${redisCluster.attrRedisEndpointAddress}:${redisCluster.attrRedisEndpointPort}`,
                "CACHE_TTL": "3600",
                "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
                "GDAL_CACHEMAX": "200",  //  200 mb
                "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
                "GDAL_INGESTED_BYTES_AT_OPEN": "32768",  // get more bytes when opening the files.
                "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
                "GDAL_HTTP_MULTIPLEX": "YES",
                "GDAL_HTTP_VERSION": "2",
                "PYTHONWARNINGS": "ignore",
                "VSI_CACHE": "TRUE",
                "VSI_CACHE_SIZE": "5000000",
                "TITILER_API_CORS_ORIGINS": '*',
                "TITILER_API_DEBUG": "True",

                // other options : https://developmentseed.org/titiler/intro/#settings
            },
            logRetention: RetentionDays.ONE_WEEK
            }
        )

        this.tilesApi = new apigw.RestApi(this, 'GeoFMDemoTilesAPI', {
            restApiName: `geofm-demo-tiles-api-${props.envName}`,
            endpointConfiguration: {
                types: [apigw.EndpointType.REGIONAL]
            },
            binaryMediaTypes: ['*/*'], // we treat everything as binary because this API only delivers images
            deployOptions: {
                stageName: 'tile',
            },
            defaultIntegration: new apigw.LambdaIntegration(tilTilerFunction, {
                requestParameters: {
                    'integration.request.querystring.url': "'method.request.querystring.url'",
                    'integration.request.path.proxy': "'method.request.path.proxy'"
                },
            }),
            defaultMethodOptions: {
                authorizer: new apigw.RequestAuthorizer(this, 'TilesApiAuthorizer', {
                    handler: apiAuthorizerFunction,
                    identitySources: [apigw.IdentitySource.header(props.customHeaderName)],
                    resultsCacheTtl: Duration.minutes(60),                    
                })
            },
        });

        this.tilesApi.root.addProxy({
            defaultMethodOptions: {
                requestParameters: {
                    'method.request.path.proxy': true,
                    'method.request.querystring.url': true
                }
            }
        });

        new CfnOutput(
            this, 'TilTilerAPIEndpoint', {
                value: this.tilesApi.url || '',
                description: 'Tiles Server Endpoint'
            }
        );
  
        // Only allow API GW calls from CloudFront
        const webACL = new CfnWebACL(this, 'GeoFMDemoWebACL', {
            defaultAction: {
                block: {},
            },
            scope: 'REGIONAL',
            visibilityConfig: {
                cloudWatchMetricsEnabled: false,
                sampledRequestsEnabled: false,
                metricName: 'geofm-demo-web-acl'
            },
            rules: [
                {
                    name: 'AllowCloudFrontRequests',
                    priority: 0,
                    visibilityConfig: {
                        cloudWatchMetricsEnabled: false,
                        metricName: 'geofm-demo-web-acl',
                        sampledRequestsEnabled: false,
                    },
                    action: {
                        allow: {},
                    },
                    statement: {
                        byteMatchStatement: {
                            fieldToMatch: {
                                singleHeader: {
                                    Name: props.customHeaderName,
                                },
                            },
                            searchString: props.customHeaderValue,
                            positionalConstraint: 'EXACTLY',
                            textTransformations: [
                                {
                                    priority: 0,
                                    type: 'NONE',
                                },
                            ],
                        },
                    }
                }
            ],
        });

        new CfnWebACLAssociation(this, 'WebACLAssociationTilesApi', {
            webAclArn: webACL.attrArn,
            resourceArn: this.tilesApi.deploymentStage.stageArn,
        });
    }
  }