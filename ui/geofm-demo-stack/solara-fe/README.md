## Introduction

**A Solara Web App for GFM Demo**


## Local setup

* `conda create -n gfm-demo python=3.12`
* `source activate gfm-demo`
* `pip install -r requirements.txt`

## RUN
`solara run pages/similarity_search.py`

## TODOS

* [x] Create TileServer on AWS Lambda with Redis Cache to serve COG files
* [x] Deploy to CloudFront
* [x] Add Auth



### Run Locally

1. build image

```
docker build --no-cache -t solara-fe .
```

2. Run Locally

```
docker run -i -p 8000:8000 solara-fe
```

3. (Optional) get your creds with something like this:

```
isengardcli credentials ACCOUNT_EMAIL --role CONSOLE_ROLE
```

4. export your AWS creds and run docker

```
docker run  -i -e SOLARA_TELEMETRY_MIXPANEL_ENABLE=False -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID     -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY     -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN     -e AWS_REGION=us-west-2  -p 8000:8000 solara-fe
```

### Update ECR Container 


