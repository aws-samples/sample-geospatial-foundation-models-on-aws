import json
from typing import Dict, Any
import boto3
from PIL import Image
import io
s3_res = boto3.resource('s3')


def read_s3_config(bucket_name: str, key: str):
    """
    Reads the frontend JSON config created by CDK.

    Args:
        bucket_name: Bucket containing the FE config JSON file (i.e. geofm-demo-xxx-us-west-2-dev)
        key: config file name (i.e config.json)

    Returns:
        Dict containing the FE configuration
        {
            "tiles_backend_url": "https://xxx.cloudfront.net/tile/cog/tiles",
            "cloudfront_url":"https://xxx.cloudfront.net/api/",
            "geotiff_bucket_url":"https://s3.us-west-2.amazonaws.com/geofm-demo-xxx-us-west-2-dev-geotiff"
        }
    """
    s3_client = boto3.client('s3')

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        json_content = json.loads(response['Body'].read().decode('utf-8'))
        return json_content
    except Exception as e:
        print(f"Error reading FE JSON from S3: {str(e)}")
        return None

# Custom decoder for "True"/"False" strings
def bool_decoder(dct):
    for k, v in dct.items():
        if isinstance(v, str) and v.lower() in ('true', 'false'):
            dct[k] = v.lower() == 'true'
    return dct

def load_config(config_path: str = 'demo_config.json', demo_id: int = None) -> Dict[str, Any]:
    """
    Load configuration for a specific demo from a JSON file.
    
    Args:
        config_path: Path to the config JSON file
        demo_id: ID of the demo to load, uses current_demo_id if None
    
    Returns:
        Dict containing the demo configuration
        
    Raises:
        Exception: If demo is not found
    """
    with open(config_path, 'r') as file:
        config = json.load(file, object_hook=bool_decoder)
        if demo_id == None:
            demo_id = config["current_demo_id"]
        demo_config = [obj for obj in config["demos"] if obj["demo_id"] == demo_id]

    if len(demo_config) > 0:
        # read FE config created by CDK and add missing values
        fe_config = read_s3_config(demo_config[0]["config"]["fe_bucket_name"], "config.json")
        
        # add new keys to the config
        if fe_config != None:
            for k in fe_config:
                demo_config[0]["config"][k] = fe_config[k]
        return demo_config[0]
    else:
        raise Exception("Demo not found")
    
def image_from_s3(s3_path):
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    bucket = s3_res.Bucket(bucket)
    image = bucket.Object(key)
    #print(key)
    img_data = image.get().get('Body').read()
    return Image.open(io.BytesIO(img_data))