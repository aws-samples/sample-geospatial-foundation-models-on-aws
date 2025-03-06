import { Duration, NestedStack, NestedStackProps, RemovalPolicy } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as cr from 'aws-cdk-lib/custom-resources';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as lambdaNodeJS from 'aws-cdk-lib/aws-lambda-nodejs'
import * as path from 'path';

export interface AuthStackStackProps extends NestedStackProps {
    readonly email: string;
    readonly envName: string;
}

export class AuthStack extends NestedStack {

    public readonly userPool: cognito.UserPool;
    public readonly userPoolClient: cognito.UserPoolClient;
    public readonly envName: string;

    constructor(scope: Construct, id: string, props: AuthStackStackProps) {
        super(scope, id, props);

        const signupLambda = new lambdaNodeJS.NodejsFunction(this, 'UserPoolSignUpTrigger', {
            entry: path.join(__dirname, '../lambdas/cognito-singup/index.js'),
            description: 'function to trigger cognito',
            handler: 'handler',
            runtime: lambda.Runtime.NODEJS_18_X,
            bundling: {
                minify: true,
                sourceMap: false,
                target: 'es2020',
                externalModules: ['aws-sdk'], // AWS SDK is already available in the Lambda environment
            },
        });

        this.envName = props.envName;

        this.userPool = new cognito.UserPool(this, "UserPool", {
            userPoolName: `GeoFMDemoUsers-${props.envName}`,
            removalPolicy: RemovalPolicy.DESTROY,
            mfa: cognito.Mfa.REQUIRED,
            
            signInAliases: { email: true, username: true },
            autoVerify: { email: true },
            accountRecovery: cognito.AccountRecovery.EMAIL_ONLY,
            userInvitation: {
                emailBody: 'Welcome to the GeoFM Demo! Use this user name to log in: {username} and your temporary password: {####}',
                emailSubject: 'GeoFM Demo log in credentials'
            },
            userVerification: {
                emailBody: 'Verify your GeoFM Demo account by clicking on {##Verify Email##}',
                emailStyle: cognito.VerificationEmailStyle.LINK,
                emailSubject: 'GeoFM Demo email verification'
            },
            // TODO FIX this if required!
            // lambdaTriggers: {
            //     preSignUp: signupLambda,
            // },            
        });

        const userPoolDomain = this.userPool.addDomain('UserPoolDomain', {
            cognitoDomain: {
                domainPrefix: 'geofm-demo-' + this.account
            }
        });

        this.userPoolClient = this.userPool.addClient("GeoFMDemoWebClient", {
            userPoolClientName: `GeoFMDemoWebClient-${props.envName}`,
            idTokenValidity: Duration.days(1),
            accessTokenValidity: Duration.days(1),
            authFlows: {
                userPassword: true,
                userSrp: true,
            },
        });

        this.createDemoUser(this.userPool.userPoolId, props.email);

        const authorizerConfig = [
            this.region,
            this.userPool.userPoolId,
            this.userPoolClient.userPoolClientId,
            userPoolDomain.baseUrl().replace('https://', '')]
            .join(';');

        // Using a custom resource to put the SSM Parameter in us-east-1 region
        // so that the Lambda@Edge is able to retrieve it in that specific region
        new cr.AwsCustomResource(this, 'AuthorizerConfigParameter', {
            onCreate: this.getSsmSdkCall('putParameter', authorizerConfig),
            onUpdate: this.getSsmSdkCall('putParameter', authorizerConfig),
            onDelete: this.getSsmSdkCall('deleteParameter'),
            installLatestAwsSdk: false,
            policy: cr.AwsCustomResourcePolicy.fromSdkCalls({
                resources: ['*']
            }),
        });
    }

    private createDemoUser(userPoolId: string, email: string) {
        new cognito.CfnUserPoolUser(this, 'DefaultUserAccount', {
            userPoolId: userPoolId,
            username: 'demo',
            desiredDeliveryMediums: ["EMAIL"],
            userAttributes: [{
                name: 'email',
                value: email,
            }, {
                name: 'email_verified',
                value: 'True'
            }],
        });
    }

    private getSsmSdkCall(action: string, value?: string): cr.AwsSdkCall {
        const parameterName = `/GeoFMDemo/${this.envName}/AuthorizerConfig`;
        let parameters: {[k: string]: any} = {
            'Name': parameterName,
        };
        if (action !== 'deleteParameter') {
            parameters['Type'] = 'String'
            parameters['Overwrite'] = true
        }
        const result: cr.AwsSdkCall = {
            service: 'SSM',
            action: action,
            region: 'us-east-1',
            physicalResourceId: cr.PhysicalResourceId.of(parameterName),
            parameters
        };

        if (value != null) {
            result.parameters.Value = value;
            result.parameters.Type = 'String';
            result.parameters.Overwrite = true;
        }

        return result;
    }
}
