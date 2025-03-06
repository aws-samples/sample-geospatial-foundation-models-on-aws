const headerName = process.env.HEADER_NAME;
const headerValue = process.env.HEADER_VALUE;

var generatePolicy = function (principalId, resource) {
    return {
        principalId,
        policyDocument: {
            Version: '2012-10-17',
            Statement: [{
                Action: 'execute-api:Invoke',
                Effect: 'Allow',
                Resource: resource
            }]
        }
    };
}

var generateArn = function (methodArn) {
    var arnParts = methodArn.split(':');
    var awsAccountId = arnParts[4];
    var region = arnParts[3];
    return `arn:aws:execute-api:${region}:${awsAccountId}:*/*/*/*`;
}

exports.handler = function (event, context, callback) {
    
    const resourceArn = generateArn(event.methodArn);

    if (event.headers[headerName] === headerValue) {
        callback(null, generatePolicy('API', resourceArn));
    } else {
        callback('Unauthorized');
    }
}
