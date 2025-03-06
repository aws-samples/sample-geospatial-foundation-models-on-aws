const { Authenticator } = require('cognito-at-edge');
const { SSMClient, GetParameterCommand } = require('@aws-sdk/client-ssm');


let configItems = [];

exports.handler = async (request) => {
    if (configItems.length === 0) {
        console.log('Getting authorizer config...');

        const functionName = process.env.AWS_LAMBDA_FUNCTION_NAME;
        console.log('Function name: ' + functionName);

        // Assuming the format is CloudFrontAuthorizer-{envName}. Must take last part.
        const envName = functionName.split('-').pop() || 'dev'; // Default to 'dev'
        console.log('Environment: ' + envName);

        const ssmClient = new SSMClient({ region: 'us-east-1' });

        // Create the command to get the '/GFMDemo/AuthorizerConfig' parameter
        const command = new GetParameterCommand({
            Name: `/GeoFMDemo/${envName}/AuthorizerConfig`
        });

        try {
            const ssmResponse = await ssmClient.send(command);
            console.log("Parameter value:", ssmResponse.Parameter.Value);
            const config = ssmResponse.Parameter.Value;
            configItems = config.split(';');
            console.log('Authorizer config: ' + configItems.join(' '));
        } catch (error) {
            console.error('Error fetching SSM parameter:', error);
            throw error;
        }
    }

    const authenticator = new Authenticator({
        region: configItems[0],
        userPoolId: configItems[1],
        userPoolAppId: configItems[2],
        userPoolDomain: configItems[3]
    });

    return await authenticator.handle(request);
};


// const { Authenticator } = require('cognito-at-edge');

// const AWS = require('aws-sdk');

// var configItems = [];

// exports.handler = async (request) => {
//     if (configItems.length == 0) {
//         console.log('Getting authorizer config...');
//         const ssmResponse = await new AWS.SSM({ region: 'us-east-1'}).getParameter({ Name: '/GFMDemo/AuthorizerConfig' }).promise();
//         const config = ssmResponse.Parameter.Value;
//         configItems = config.split(';');
//         console.log('Authorizer config: ' + configItems.join(' '));
//     }

//     const authenticator = new Authenticator({
//         region: configItems[0],
//         userPoolId: configItems[1],
//         userPoolAppId: configItems[2],
//         userPoolDomain: configItems[3]
//     });

//     return await authenticator.handle(request);
// }
