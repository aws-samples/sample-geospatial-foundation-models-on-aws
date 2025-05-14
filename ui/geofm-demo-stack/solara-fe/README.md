## Introduction

**A Solara Web App for AWS GeoFM Demo**


## Local setup

* `conda create -n geofm-demo python=3.12`
* `source activate geofm-demo`
* `pip install -r requirements.txt`

## Run app

* Run on Single page: `solara run pages/similarity_search.py`

### Run on Local Docker container

1. build image

```
docker build --no-cache -t solara-fe .
```

2. Run Locally

```
docker run -i -p 8000:8000 solara-fe
```

3. Ensure you have AWS CLI access locally

4. export your AWS creds and run docker

```
docker run  -i -e SOLARA_TELEMETRY_MIXPANEL_ENABLE=False -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID     -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY     -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN     -e AWS_REGION=us-west-2  -p 8000:8000 solara-fe
```


