import boto3
from botocore.exceptions import ClientError
import lancedb
import numpy as np
from io import BytesIO
from PIL import Image
import pandas as pd
import multiprocessing
import concurrent.futures
from collections import defaultdict
import logging
from datetime import datetime
from typing import List, Dict
from io import BytesIO
import geopandas as gpd
import io
import time
import concurrent.futures
from functools import partial
from collections import defaultdict
import numpy as np
import multiprocessing
import logging

logger = logging.getLogger(__name__)
# Get the number of CPUs available on the system
num_cpus = multiprocessing.cpu_count()
s3_client = boto3.client('s3')
s3_res = boto3.resource('s3')

def readEmbeddingsFromS3(s3_bucket, s3_output_object, max_retries=5):
    for attempt in range(max_retries):
        try:
            # Get the object from the S3 bucket
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_output_object)

            # Read the object's content into a NumPy array
            file_content = response['Body'].read()
            np_array = np.load(BytesIO(file_content), allow_pickle=True)

            return np_array
        except ClientError as e:
            if e.response['Error']['Code'] in ['ThrottlingException', 'RequestLimitExceeded']:
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception as e:
            #logger.debug(f"Error reading embeddings from S3: {e}")
            raise

    raise Exception(f"Max retries reached for {s3_bucket}/{s3_output_object}")

def load_single_embedding(f):
    try:
        bucket = f.split("/")[2]
        obj_path = "/".join(f.split("/")[3:])
        emb = readEmbeddingsFromS3(bucket, obj_path)
        return emb['s3_location_netcdf'], emb
    except Exception as e:
        logger.debug(f"Exception: {e}")
        logger.debug(f"No embeddings for file {f}")
        return None

def load_embeddings_parallelized(s3_paths, max_workers=num_cpus):
    chip_embeddings = defaultdict(dict)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(load_single_embedding, f): f for f in s3_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            result = future.result()
            if result:
                key, emb = result
                chip_embeddings[key] = emb
    return dict(chip_embeddings)

def update_consolidated_results(consolidated_results, chip_embeddings_dict, emb_type="cls"):
    def add_embedding_to_consolidated_results(data_cube_path, chip_embeddings_dict, emb_key):
        try:
            emb = chip_embeddings_dict[data_cube_path][emb_key]
            return emb
        except:
            return np.nan
    
    if emb_type == "cls":
        consolidated_results[f"{emb_type}_embeddings"] = consolidated_results['s3_location_netcdf'].apply(lambda x: add_embedding_to_consolidated_results(x, chip_embeddings_dict,'cls_token_embeddings'))
    elif emb_type=="patch": 
        consolidated_results[f"{emb_type}_embeddings"] = consolidated_results['s3_location_netcdf'].apply(lambda x: add_embedding_to_consolidated_results(x, chip_embeddings_dict,'patch_embeddings'))
    return consolidated_results
    
def load_consolidated_results(bucket_name: str, aoi_name: str, tile_id: str, emb_type: str = "cls") -> pd.DataFrame:
    """Load consolidated results from S3"""
    s3_client = boto3.client('s3')
    
    # Get the number of CPUs available on the system
    num_cpus = multiprocessing.cpu_count()

    file_key = f"output/{aoi_name}/consolidated-output/{tile_id}/consolidated_output.parquet"
    consolidated_results = gpd.read_parquet(f"s3://{bucket_name}/{file_key}")
    s3_paths = list(consolidated_results[f"s3_location_{emb_type}_emb"].values)
    print(f"Loading {len(s3_paths)} files from S3")
    chip_embeddings_dict = load_embeddings_parallelized(s3_paths, max_workers=num_cpus)
    print(f"Adding {emb_type} embeddings to the parquet file")
    consolidated_results = update_consolidated_results(consolidated_results, chip_embeddings_dict, emb_type)
    return consolidated_results
    
class LanceDBHelper:
    def __init__(self, config, create_new_table=False):
        """
        Initialize LanceDB helper with configuration
        
        Args:
            config: Configuration dictionary containing bucket_name, aoi_name
            create_new_table: If True, will create a new table. If False, will use existing table
        """
        self.logger = logging.getLogger(__name__)
        self.num_cpus = multiprocessing.cpu_count()
        
        # S3 clients
        self.s3 = boto3.client('s3')
        self.s3_res = boto3.resource('s3')
        
        # LanceDB connection
        self.bucket_name = config["bucket_name"]
        self.aoi_name = config["aoi_name"]
        self.table_name = config["table_name"]
        self.lancedb_uri = f"s3://{self.bucket_name}/output/{self.aoi_name}/lance-db/"
        self.db = lancedb.connect(self.lancedb_uri)
        
        # Table connection
        if create_new_table:
            self.table = None  # Will be created later using create_table method
        else:
            try:
                self.table = self.db.open_table(self.table_name)
                print(f"Connected to existing table '{self.table_name}'")
                print(f"Number of rows: {self.table.count_rows()}")
                print(self.table.schema)
            except Exception as e:
                raise Exception(f"Error opening table {self.table_name}: {str(e)}")

    def get_cls_token_emb(self, s3_path: str, chip_id: str) -> Dict:
        """Get CLS token embeddings from S3 path"""
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

    def find_similar_items_from_query(
        self,
        query_emb: List[float],
        k: int,
        num_results: int,
        chip_ids: List[str] = None,
        date_filter: str = None
    ) -> List[Dict]:
        """Find similar items using vector search"""
        if self.table is None:
            raise Exception("No table available. Please create or connect to a table first.")
            
        # Start with basic search query
        search_query = self.table.search(
            query=query_emb,
            vector_column_name="vector"
        ).metric("cosine").limit(num_results)

        # Build filter conditions
        filters = []
        if date_filter:
            filters.append(f"date = date '{date_filter}'")
        if chip_ids:
            chip_ids_str = ", ".join([f"'{chip_id}'" for chip_id in chip_ids])
            filters.append(f"chip_id NOT IN ({chip_ids_str})")

        # Apply filters if any
        if filters:
            search_query = search_query.where(" AND ".join(filters), prefilter=True)

        # Execute search and format results
        results = search_query.to_pandas()

        # return results
        return [{
            "score": (1 - row["_distance"]), #TODO score is 1 - distance for now
            "chip_id": row["chip_id"],
            "s3_location_netcdf": row["s3_location_netcdf"],
            "s3_location_chip_png": row["s3_location_png_thumbnail"],
            "date": row["date"]
        } for _, row in results.iterrows()]

    def create_table(self, consolidated_results: pd.DataFrame, emb_type: str = "cls"):
        """Create or update LanceDB table"""
        table_data = consolidated_results.copy()
        table_data['bbox'] = table_data['bbox'].apply(lambda x: x.bounds)
        table_data["date"] = table_data["date"].apply(lambda x: x.date())
        
        table_data["vector"] = table_data[f"{emb_type}_embeddings"]
        table_data.drop(columns=[f"{emb_type}_embeddings"], inplace=True)
        
        self.table = self.db.create_table(
            self.table_name,
            data=table_data,
            mode='overwrite',
            on_bad_vectors="drop"
        )
        
        print(f"Table '{self.table_name}' created successfully in LanceDB.")
        print(f"Number of rows inserted: {self.table.count_rows()}")
        return self.table
