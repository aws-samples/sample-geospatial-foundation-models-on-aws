import json
from typing import Dict, Any
import boto3
from PIL import Image
import io
s3_res = boto3.resource('s3')


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