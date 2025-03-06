import { CfnResource, Duration, Fn, NestedStack, NestedStackProps, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as cf from 'aws-cdk-lib/aws-cloudfront';
import * as cfo from 'aws-cdk-lib/aws-cloudfront-origins';
import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as cr from 'aws-cdk-lib/custom-resources';



import * as path from 'path';


export interface FrontendStackStackProps extends NestedStackProps {
    readonly customHeaderName: string;
    readonly customHeaderValue: string;
    readonly tilesApi: apigw.RestApi;
    readonly geotiffBucketVpcEndpointId: string;
    readonly solaraSGs: string[];
    readonly solaraOriginLBDnsName: string;
    readonly envName: string;
}

export class FrontendStack extends NestedStack {

    public readonly staticContentBucket: s3.Bucket;
    public readonly geoTiffBucket: s3.Bucket;
    public readonly frontendUrl: string;
    public readonly geotiffUrl: string;
    public readonly geotiffBucketName: string;
    public readonly cloudFrontDistribution: cf.Distribution;
    public readonly authorizerFunction: cf.experimental.EdgeFunction;

    constructor(scope: Construct, id: string, props: FrontendStackStackProps) {
        super(scope, id, props);

        this.staticContentBucket = new s3.Bucket(this, 'StaticContent', {
            bucketName: `aws-geofm-fe-bucket-${this.account}-${this.region}-${props.envName}`,
            blockPublicAccess: {
                blockPublicAcls: true,
                restrictPublicBuckets: true,
                blockPublicPolicy: true,
                ignorePublicAcls: true,
            },
            removalPolicy: RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
            accessControl: s3.BucketAccessControl.PRIVATE, 
            enforceSSL: true
        });

        const staticContentOriginAccessIdentity = new cf.OriginAccessIdentity(this, 'StaticContentOriginAccessIdentity');
        this.staticContentBucket.grantRead(staticContentOriginAccessIdentity);

        this.geoTiffBucket = new s3.Bucket(this, 'GeoTiffBucket', {
            bucketName: `aws-geofm-geotiff-bucket-${this.account}-${this.region}-${props.envName}`,
            blockPublicAccess: {
                blockPublicAcls: true,
                restrictPublicBuckets: true,
                blockPublicPolicy: true,
                ignorePublicAcls: true
            },
            removalPolicy: RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
            accessControl: s3.BucketAccessControl.PRIVATE, 
            enforceSSL: true,
        });

        this.geotiffUrl = this.geoTiffBucket.urlForObject();
        this.geotiffBucketName = this.geoTiffBucket.bucketName;

        this.geoTiffBucket.addToResourcePolicy(new iam.PolicyStatement({
            actions: ['s3:GetObject'],
            principals: [new iam.AccountPrincipal(this.account)],
            resources: [ this.geoTiffBucket.arnForObjects('*') ],
            conditions: {
                StringEquals: {
                    "aws:sourceVpce": `${props.geotiffBucketVpcEndpointId}`
                }
            }
        }));

        // Allow direct access to GeoTiff images from the bucket using OAI
        const geoTiffOriginAccessIdentity = new cf.OriginAccessIdentity(this, 'GeoTiffOriginAccessIdentity');
        this.geoTiffBucket.grantRead(geoTiffOriginAccessIdentity);

        this.authorizerFunction = new cloudfront.experimental.EdgeFunction(this, 'CloudFrontAuthorizer', {
            runtime: lambda.Runtime.NODEJS_18_X,
            handler: 'index.handler',
            code: lambda.Code.fromAsset(path.join(__dirname, '../lambdas/authorizer')),
            functionName: `CloudFrontAuthorizer-${props.envName}`,
            description: 'Lambda@Edge performing the authorization with Cognito',
        });

        // This Lambda@Edge will be replicated, so attempting to delete it will fail.
        // The function has to be deleted manually later, when the (deleted) CloudFront distribution releases all replicas.
        (this.authorizerFunction.node.defaultChild as CfnResource).applyRemovalPolicy(RemovalPolicy.RETAIN);

        this.authorizerFunction.addToRolePolicy(new iam.PolicyStatement({
            actions: ['ssm:DescribeParameters', 'ssm:GetParameter', 'ssm:GetParameters'],
            effect: iam.Effect.ALLOW,
            resources: ['*']
        }));

        ////===== FIX BUG: CloudFront error saying can't do lambda:GetFunction
        // Explicitly publish a version of the function
        const version = this.authorizerFunction.currentVersion;

        // Add a specific policy for Lambda@Edge
        const edgeLambdaPolicy = new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            'lambda:GetFunction',
            'lambda:EnableReplication*',
            'lambda:GetFunctionConfiguration',
            'lambda:GetLayerVersion',
            'lambda:InvokeFunction',
          ],
          resources: [
            this.authorizerFunction.functionArn,
            `${this.authorizerFunction.functionArn}:*`,
            version.functionArn
          ],
        });

        // Add the policy to the Lambda function's role
        this.authorizerFunction.role?.addToPrincipalPolicy(edgeLambdaPolicy);

        // Grant permissions for CloudFront to use the function
        version.grantInvoke(new iam.ServicePrincipal('edgelambda.amazonaws.com'));
        version.grantInvoke(new iam.ServicePrincipal('lambda.amazonaws.com'));

        // Add a resource-based policy to allow CloudFront to use the function
        version.addPermission('EdgeLambdaPermission', {
          principal: new iam.ServicePrincipal('edgelambda.amazonaws.com'),
          action: 'lambda:GetFunction',
          sourceAccount: this.account,
          sourceArn: `arn:aws:cloudfront::${this.account}:distribution/*`,
        });
        ////////===== END OF BUG FIX

        const edgeLambda = [{
            eventType: cf.LambdaEdgeEventType.VIEWER_REQUEST,
            functionVersion: version
        }];

        // Create the CloudFront distribution
        this.cloudFrontDistribution = new cf.Distribution(this, 'MainDistribution', {
            comment: 'GeoFM Demo Distribution',
            // defaultRootObject: 'index.html',
            defaultBehavior: {
                // origin is from solara ecs alb
                origin: new cfo.HttpOrigin(`${props.solaraOriginLBDnsName}`, {
                    httpPort: 8000,
                    customHeaders: { [props.customHeaderName]: props.customHeaderValue },
                    originSslProtocols: [cloudfront.OriginSslPolicy.TLS_V1_2],
                    protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
                }),
                viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.ALLOW_ALL,
                allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
                cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
                originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
                edgeLambdas: edgeLambda
            }
            // defaultBehavior: {
            //     origin: new cfo.S3Origin(this.staticContentBucket, { originAccessIdentity }),
            //     edgeLambdas: edgeLambda
            // },
        });
        const defaultCachePolicy = new cf.CachePolicy(this, 'GeoFMDemoCachePolicy', {
            // cachePolicyName: 'GeoFMDemoCachePolicy', 
            comment: 'A default policy for GeoFM Demo',
            cookieBehavior: cf.CacheCookieBehavior.all(),
            headerBehavior: cf.CacheHeaderBehavior.none(),
            queryStringBehavior: cf.CacheQueryStringBehavior.all(),
            enableAcceptEncodingGzip: true,
        });

        this.cloudFrontDistribution.addBehavior('tile/*', new cfo.RestApiOrigin(props.tilesApi, {
            customHeaders: { [props.customHeaderName]: props.customHeaderValue },
            }),
            {
                edgeLambdas: edgeLambda,
                cachePolicy: defaultCachePolicy,
            });

        // The Geotiff images are located in geotiff/*.tif
        this.cloudFrontDistribution.addBehavior('geotiff/*', new cfo.S3Origin(this.geoTiffBucket, {
            originAccessIdentity: geoTiffOriginAccessIdentity
            }),
            {
                edgeLambdas: edgeLambda,
                cachePolicy: defaultCachePolicy
            });

        // The Static assets should be located under files/
        this.cloudFrontDistribution.addBehavior('files/*', new cfo.S3Origin(this.staticContentBucket, {
            originAccessIdentity: staticContentOriginAccessIdentity
            }),
            {
                edgeLambdas: edgeLambda,
                cachePolicy: defaultCachePolicy
            });

        // Getting Cloudfront Prefix Id using Custom Resource 
        const getPrefixListId = new cr.AwsCustomResource(this, 'GetCloudFrontPrefixList', {
            onCreate: {
            service: 'EC2',
            action: 'describeManagedPrefixLists',
            parameters: {
                Filters: [
                {
                    Name: 'prefix-list-name',
                    Values: ['com.amazonaws.global.cloudfront.origin-facing']
                }
                ]
            },
            physicalResourceId: cr.PhysicalResourceId.of('CloudFrontPrefixList'),
            },
            policy: cr.AwsCustomResourcePolicy.fromStatements([
            new iam.PolicyStatement({
                effect: iam.Effect.ALLOW,
                actions: ['ec2:DescribeManagedPrefixLists'],
                resources: ['*']
            })
            ])
        });
        
        // Get the prefix list ID from the custom resource
        const prefixListId = getPrefixListId.getResponseField('PrefixLists.0.PrefixListId');

        const importedSecurityGroup1 = ec2.SecurityGroup.fromSecurityGroupId(
            this, 
            'ImportedSG1', 
            props.solaraSGs[0]
        );

        importedSecurityGroup1.addIngressRule(
           ec2.Peer.prefixList(prefixListId),          
           ec2.Port.tcp(8000),
          "allow 8000 access from cloudfront1"
        );

        this.frontendUrl = 'https://' + this.cloudFrontDistribution.domainName;
    }
}
