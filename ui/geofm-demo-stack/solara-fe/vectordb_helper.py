import io
import os
import sys
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import boto3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
DATA_DIR = Path(__file__).parent.parent / "data"

from lancedb_helper import LanceDBHelper
from oss_helper_v2 import OpenSearchHelper

class VectorDBHelper:
    _instance = None

    def __new__(cls, config=None):
        if cls._instance is None:
            cls._instance = super(VectorDBHelper, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config=None):
        if self._initialized:
            return
            
        if config is None:
            raise ValueError("Config is required for first initialization")
            
        self.config = config
        self.s3_res = boto3.resource('s3')
        self.vectordb_type = config["config"].get("vectordb_type")
        self._initialize_helper()
        self._initialized = True

    def _initialize_helper(self):
        if self.vectordb_type == "lancedb":
            lance_config = {
                "bucket_name": self.config["config"].get("bucket_name"),
                "aoi_name": self.config["config"].get("aoi_name"),
                "table_name": self.config["config"].get("vectordb_collection_name"),
            }
            self.helper = LanceDBHelper(lance_config, create_new_table=False)
        elif self.vectordb_type == "opensearch":
            self.helper = OpenSearchHelper(self.config)
        else:
            raise Exception("Invalid vectordb_type in config")

    def image_from_s3(self, s3_path: str) -> Image:
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        bucket = self.s3_res.Bucket(bucket)
        image = bucket.Object(key)
        img_data = image.get().get('Body').read()
        return Image.open(io.BytesIO(img_data))

    def get_cls_token_emb(self, s3_path: str, chip_id: str) -> Dict:
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        obj = self.s3_res.Object(bucket, key)
        with BytesIO(obj.get()["Body"].read()) as f:
            f.seek(0)
            np_data = np.load(f, allow_pickle=True)
            return {
                's3_location_netcdf': np_data['s3_location_netcdf'],
                'date': np_data['date'],
                'chip_id': chip_id,
                's2_tile_id': np_data['origin_tile'],
                'cls_emb': np_data['cls_token_embeddings']
            }

    def find_similar_items(self, query_emb: List[float], k: int, num_results: int, 
                          chip_ids: List[str] = None, date_filter: str = None) -> List[Dict]:
        return self.helper.find_similar_items_from_query(query_emb, k, num_results, chip_ids, date_filter)
