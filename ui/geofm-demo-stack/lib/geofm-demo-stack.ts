import { CfnOutput, CfnParameter, Stack, StackProps } from 'aws-cdk-lib';
import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { BackendStack } from './backend-stack';
import { AuthStack } from './auth-stack';
import { FrontendStack } from './frontend-stack';
import { SolaraFEStack } from './solara-fe-stack';
import * as cr from 'aws-cdk-lib/custom-resources';

export interface GeoFMDemoStackProps extends cdk.StackProps {
    envName: string;
  }

export class GeoFMDemoStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: GeoFMDemoStackProps) {
    super(scope, id, props);

    // const emailParameter = new CfnParameter(this, 'UserEmail', {
    //     type: 'String',
    //     description: 'The email that will receive the login credentials',
    //     minLength: 3,
    //     allowedPattern: `^\\w+([-+.']\\w+)*@\\w+([-.]\\w+)*\\.\\w+([-.]\\w+)*$`
    // });

    const secureHeaderName = 'WafProtection';
    const secureHeaderValue = this.stackId;

    // The code that defines your stack goes here
    const auth = new AuthStack(this, 'AuthStack', {
        email: this.node.tryGetContext('userEmail'),
        envName: props?.envName || 'dev'
    });

    const backend = new BackendStack(this, 'BackendStack', {
        userPool: auth.userPool,
        customHeaderName: secureHeaderName,
        customHeaderValue: secureHeaderValue,
        userPoolClientId: auth.userPoolClient.userPoolClientId,
        envName: props?.envName || 'dev'
    });

    const solaraBackend = new SolaraFEStack(this, 'SolaraFEStack', {
        // cloudFrontDistribution: frontend.cloudFrontDistribution,
        customHeaderName: secureHeaderName,
        customHeaderValue: secureHeaderValue,
        envName: props?.envName || 'dev',

        // userPool: auth.userPool,
        // authorizerFunction: frontend.authorizerFunction,
    });

    const frontend = new FrontendStack(this, 'FrontendStack', {
        solaraSGs: solaraBackend.solaraSGs,
        solaraOriginLBDnsName: solaraBackend.solaraOriginLBDnsName,
        customHeaderName: secureHeaderName,
        customHeaderValue: secureHeaderValue,
        tilesApi: backend.tilesApi,
        geotiffBucketVpcEndpointId: backend.s3VpcPointId,
        envName: props?.envName || 'dev'
    });

    this.applyConfiguration(frontend, auth);



    new CfnOutput(this, 'CloudFrontURL', {
        value: frontend.frontendUrl,
        description: 'The URL to access the demo page'
    });

  }

  private applyConfiguration(frontend: FrontendStack, auth: AuthStack) {

    const putUserPoolClientConfig: cr.AwsSdkCall = {
        service: 'CognitoIdentityServiceProvider',
        action: 'updateUserPoolClient',
        parameters: {
            ClientId: auth.userPoolClient.userPoolClientId,
            UserPoolId: auth.userPool.userPoolId,
            AllowedOAuthFlows: ['code' , 'implicit'],
            AllowedOAuthFlowsUserPoolClient: true,
            AllowedOAuthScopes: [
                'profile',
                'email',
                'openid',
                'aws.cognito.signin.user.admin'],
            CallbackURLs: [frontend.frontendUrl],
            ClientName: auth.userPoolClient.userPoolClientName,
            ExplicitAuthFlows: [
                "ALLOW_USER_PASSWORD_AUTH",
                "ALLOW_USER_SRP_AUTH",
                "ALLOW_REFRESH_TOKEN_AUTH"
            ],
            SupportedIdentityProviders: [ 'COGNITO' ],
        },
        physicalResourceId: cr.PhysicalResourceId.of(auth.userPool.userPoolArn),
    };

    const customResourceUserPoolClientConfig = new cr.AwsCustomResource(this, 'UpdateUserPoolClientConfig', {
        onCreate: putUserPoolClientConfig,
        onUpdate: putUserPoolClientConfig,
        policy: cr.AwsCustomResourcePolicy.fromSdkCalls({
            resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE,
        }),
        installLatestAwsSdk: false
    });
    customResourceUserPoolClientConfig.node.addDependency(frontend.cloudFrontDistribution, auth.userPoolClient);

    const config = {
        tiles_backend_url: frontend.frontendUrl + '/tile/cog/tiles',
        cloudfront_url: frontend.frontendUrl,
        geotiff_bucket_url: frontend.geotiffUrl
    };

    const putConfigCall: cr.AwsSdkCall = {
        service: 'S3',
        action: 'putObject',
        parameters: {
            Bucket: frontend.staticContentBucket.bucketName,
            Key: 'config.json',
            Body: JSON.stringify(config),
        },
        physicalResourceId: cr.PhysicalResourceId.of(frontend.staticContentBucket.bucketArn),
    };

    const customResourceFrontendConfig = new cr.AwsCustomResource(this, 'PutFrontendConfig', {
        onCreate: putConfigCall,
        onUpdate: putConfigCall,
        policy: cr.AwsCustomResourcePolicy.fromSdkCalls({
            resources: [frontend.staticContentBucket.bucketArn + '/*']
        }),
        installLatestAwsSdk: false
    });
    customResourceFrontendConfig.node.addDependency(frontend.geoTiffBucket, frontend.cloudFrontDistribution);
  }
}
