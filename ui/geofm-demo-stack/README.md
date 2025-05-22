# Solara UI Stack

This folder contains the CDK stack to deploy the GeoFM demo UI.

## How to deploy

### Prerequisites

You need to have CDK and AWS Cli installed on your device to deploy this stack.

### Deployment

1. Bootstrap AWS region: If running on `us-east-1` run the following command and move to step 2.

   ```
   npm install
   export AWS_REGION=us-east-1
   cdk bootstrap
   ```
   
   (OPTIONAL) If running on a different region (i.e. `us-west-2`) bootstrap both regions.  
   ```
   npm install
   export AWS_REGION=us-east-1
   cdk bootstrap
   export AWS_REGION=us-west-2
   cdk bootstrap
   ```

2. Install libs
   ```
   cd lambdas/authorizer/ && npm i
   ```

3. Deploy Stack to dev environment:
   ```
   cd ../../
   cdk deploy --all -c env=dev -c userEmail=YOUR_EMAIL_FOR_COGNITO
   ```
