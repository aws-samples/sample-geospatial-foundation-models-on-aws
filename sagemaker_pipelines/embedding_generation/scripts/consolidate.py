from io import BytesIO
import os
import sys
import subprocess
import warnings
import time
import pandas as pd
import geopandas as gpd
import time
import concurrent.futures
from collections import defaultdict
import numpy as np
import multiprocessing
import logging
import lancedb
import argparse
import boto3
import json
import gc
from botocore.exceptions import ClientError
from pystac_client import Client
import traceback
warnings.filterwarnings('ignore') # Ignore all warnings

# Create a global S3 client with optimized settings
s3_client = boto3.client('s3', config=boto3.session.Config(
    max_pool_connections=50,
    retries={'max_attempts': 10}
))

# Initialize the STAC client
stac_api_url = "https://earth-search.aws.element84.com/v1"
client = Client.open(stac_api_url)

#HELPERS
def get_logger(log_level):    
    logger = logging.getLogger("processing")

    console_handler = logging.StreamHandler(sys.stdout)
    # include %(name)s to also include logger name
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    console_handler.setLevel(log_level)

    logger.addHandler(console_handler)
    logger.setLevel(log_level)
    return logger

def convert_scene_id_to_url(scene_id):
    """
    Convert a Sentinel-2 scene ID to its corresponding COG URL.

    Args:
        scene_id (str): Sentinel-2 scene ID in format 'S2C_30SYJ_20250124_0_L2A'

    Returns:
        str: URL to the TCI.tif file
    """
    # Base URL for Sentinel-2 COGs
    base_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs"

    # Parse the scene ID
    parts = scene_id.split('_')

    # Extract the tile ID (e.g., '30SYJ')
    tile_id = parts[1]

    # Split the tile ID into components
    utm_zone = tile_id[:2]  # e.g., '30'
    lat_band = tile_id[2]   # e.g., 'S'
    grid_square = tile_id[3:]  # e.g., 'YJ'

    # Extract date (e.g., '20250124')
    date_str = parts[2]
    year = date_str[:4]     # e.g., '2025'
    month = date_str[4:6]   # e.g., '01'
    # Remove leading zero from month
    month = str(int(month))

    # Construct the URL
    url = f"{base_url}/{utm_zone}/{lat_band}/{grid_square}/{year}/{month}/{scene_id}/TCI.tif"

    return url

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
            if e.response['Error']['Code'] in ['ThrottlingException', 'RequestLimitExceeded', 'SlowDown']:
                wait_time = min(30, 2 ** attempt)  # Exponential backoff with cap
                logger.warning(f"S3 throttling encountered, retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"S3 client error: {e}")
                raise
        except Exception as e:
            logger.debug(f"Error reading embeddings from S3: {e}")
            if attempt < max_retries - 1:
                wait_time = min(30, 2 ** attempt)
                logger.warning(f"Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
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

def process_in_batches(table_data, emb_type, batch_size=1000, max_workers=8):
    """Process embeddings in batches to avoid memory issues."""
    total_rows = len(table_data)
    logger.info(f"Processing {total_rows} rows in batches of {batch_size}")
    
    # Create a new column for embeddings
    table_data[f"{emb_type}_embeddings"] = None
    
    # Process in batches
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        logger.info(f"Processing batch {start_idx//batch_size + 1}: rows {start_idx} to {end_idx}")
        
        # Get the batch of paths
        batch_df = table_data.iloc[start_idx:end_idx]
        s3_paths = list(batch_df[f"s3_location_{emb_type}_emb"].values)
        netcdf_keys = list(batch_df["s3_location_netcdf"].values)
        
        # Load embeddings for this batch
        batch_embeddings = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(load_single_embedding, f): f for f in s3_paths}
            for future in concurrent.futures.as_completed(future_to_path):
                result = future.result()
                if result:
                    key, emb = result
                    batch_embeddings[key] = emb
        
        logger.info(f"Loaded {len(batch_embeddings)} embeddings for batch")
        
        # Apply embeddings to the current batch
        for i, row_idx in enumerate(range(start_idx, end_idx)):
            netcdf_key = netcdf_keys[i]
            try:
                if netcdf_key in batch_embeddings:
                    emb_key = 'cls_token_embeddings' if emb_type == 'cls' else 'patch_embeddings'
                    if emb_key in batch_embeddings[netcdf_key]:
                        table_data.at[row_idx, f"{emb_type}_embeddings"] = batch_embeddings[netcdf_key][emb_key]
            except Exception as e:
                logger.debug(f"Error processing embedding at index {row_idx}: {str(e)}")
        
        # Clear the batch embeddings to free memory
        del batch_embeddings
        gc.collect()
        
        logger.info(f"Completed batch {start_idx//batch_size + 1}")
    
    # Count valid embeddings
    valid_count = table_data[f"{emb_type}_embeddings"].count()
    logger.info(f"Processed {valid_count}/{total_rows} valid embeddings")
    
    return table_data

def generate_config_json(combined_gdf,bucket_name,env_name,account_number,region,aoi_name,tile_id,output_path,split_view_tiles_months=["06","09"]):
    

    stac_api_url = "https://earth-search.aws.element84.com/v1"
    client = Client.open(stac_api_url)
    
    max_date = str(combined_gdf["date"].max().date())
    baseline_year = str(combined_gdf["date"].min().year)
    monitoring_year = str(combined_gdf["date"].min().year+1)

    s2_cog_id_sim_search_date=combined_gdf[combined_gdf["date"]==max_date]["origin_tile"].unique()[0]

    # def get_bucket_region(bucket_name):
    #     try:
    #         s3 = boto3.resource('s3')
    #         bucket = s3.Bucket(bucket_name)
    #         return bucket.meta.client.meta.region_name
    #     except Exception as e:
    #         print(f"Error getting bucket region: {str(e)}")
    #         return None
    
    # region = get_bucket_region(bucket_name)
    
    # Set search parameters
    years = np.arange(combined_gdf["date"].min().year,combined_gdf["date"].max().year+1,1)
    
    img_meta={}
    original_tile_id = tile_id  # Replace with your actual tile ID

    for y in years:
        
        date_range = f"{y}-{split_view_tiles_months[0]}-01/{y}-{split_view_tiles_months[-1]}-01"
        logger.info(date_range)

        search_results = client.search(
            collections=["sentinel-2-l2a"],
            datetime=date_range,
            query={
                "grid:code": {"eq": f"MGRS-{original_tile_id}"},  # Use original tile ID
                "eo:cloud_cover": {"lt": 20}
            }
        )

        items = list(search_results.items())
        if items:
            cleanest_item = min(items, key=lambda x: x.properties['eo:cloud_cover'])
            # Store metadata without modifying original_tile_id
            img_meta[str(y)] = {
                "provider": "sentinel2",
                "satellite": cleanest_item.id.split("_")[0],
                "date": cleanest_item.id.split("_")[2],
                "month": pd.to_datetime(cleanest_item.id.split("_")[2]).month,
                "processing_level": cleanest_item.id.split("_")[-1]
            }

    config_json = {
        "current_demo_id":  str(aoi_name),
        "demos": [
        {
        "demo_id": str(aoi_name),
        "config": {
            "aoi_name": str(aoi_name),
            "s2_tile_id": str(tile_id),
            "aoi_geojson_s3_key": f"processing/{aoi_name}/input/aoi/aoi.geojson",
            "chip_grid_geojson_s3_key": f"output/{aoi_name}/consolidated-output/{tile_id}/chip_grid_full_s2_tile.geojson",
            "chip_grid_change_intensity_geojson": "",
            "metadata_path": f"output/{aoi_name}/consolidated-output/{tile_id}/",
            "metadata_file_name": "consolidated_output.parquet",
            "png_thumbnail_key": "s3_location_png_thumbnail", 
            "similarity_search_date": max_date,
            "baseline_start_date": f"{baseline_year}-01-01",
            "monitoring_start_date": f"{monitoring_year}-01-01",
            "change_threshold_linear_reg": [1e-7,1.75e-7],
            "aws_region": region,
            "bucket_name": str(bucket_name),
            "fe_bucket_name": f"aws-geofm-fe-bucket-{account_number}-{region}-{env_name}",
            "use_remote_s2_cogs": "True",
            "vectordb_type": "lancedb",
            "vectordb_collection_name": "vector_db",
            "metadata": {
            "base_maps": [
                "OpenTopoMap",
                "OpenStreetMap",
                "OpenStreetMap.HOT",
                "ESA WorldCover 2021"
            ],
            "spectral_indices": [
                "TCI",
                "NDVI",
                "EVI",
                "NBR"
                ]},
            "imagery_metadata": img_meta,
            "sentinel2_cog_source_url": convert_scene_id_to_url(s2_cog_id_sim_search_date),
            "background_video_file": "<TO BE FILLED>",
            "vectordb_collection_host": "<PLACEHOLDER>",
            }
        }]
    }

    # Save with indentation and sorting keys
    with open(f'{output_path}/config_{aoi_name}.json', 'w') as file:
        json.dump(config_json, file, indent=4,sort_keys=True)
    
    return logger.info("Config json saved...")
    

#MAIN
if __name__ == "__main__":

    logger = get_logger(logging.INFO) #INFO
    parser = argparse.ArgumentParser()
    
    #parse lance db URI
    parser.add_argument("--LANCEDB_URI", type=str, default="")
    parser.add_argument("--AOI_NAME", type=str, default="")
    parser.add_argument("--S2_TILE_ID", type=str, default="")
    parser.add_argument("--BUCKET_NAME", type=str, default="")
    parser.add_argument("--ENV_NAME", type=str, default="")
    parser.add_argument("--ACCOUNT_ID", type=str, default="")
    parser.add_argument("--REGION", type=str, default="")

    args = parser.parse_args()
    lancedb_uri=args.LANCEDB_URI
    aoi_name=args.AOI_NAME
    tile_id=args.S2_TILE_ID
    bucket_name=args.BUCKET_NAME
    env_name=args.ENV_NAME
    account_number=args.ACCOUNT_ID
    region=args.REGION
    
    # Get the number of CPUs available on the system
    num_cpus = multiprocessing.cpu_count()

    # Read meta file
    input_path = '/opt/ml/processing/input/embeddings'
    input_path_meta = '/opt/ml/processing/input/meta' #we'll use the raw meta files to generate a chip_grid_aoi.geojson
    input_path_aoi = '/opt/ml/processing/input/aoi'
    output_path = '/opt/ml/processing/output'
    subprocess.check_call(["sudo","chown","-R","sagemaker-user", output_path]) #ensure write permissions on output folder
    
    # List all Parquet files in the local directory
    unique_chip_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.parquet')]
    logger.info("{} unique chip IDs received...".format(len(unique_chip_files)))

    # Function to read a single Parquet file
    def read_parquet_file(file):
        try:
            gdf = gpd.read_parquet(file)
            return gdf
        except Exception as e:
            logger.info(f"Error reading {file}: {e}")
            return None

    # Use ThreadPoolExecutor to parallelize the reading of files
    geodataframes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(read_parquet_file, unique_chip_files))
    # Filter out None results (files that couldn't be read)
    geodataframes = [gdf for gdf in results if gdf is not None]
    
    # Combine all GeoDataFrames into one
    combined_gdf = gpd.GeoDataFrame(pd.concat(geodataframes, ignore_index=True))
    combined_gdf.reset_index(inplace=True, drop=True)
    
    #save combined file to output path
    fname=f"consolidated_output.parquet"
    file_out_path=os.path.join(output_path, fname)
    combined_gdf.to_parquet(file_out_path)       
    logger.info(f"Consolidated output saved to {file_out_path}...")
    
    #generate and save config JSON
    generate_config_json(combined_gdf=combined_gdf,
                         bucket_name=bucket_name,
                         env_name=env_name,
                         account_number=account_number,
                         region=region,
                         aoi_name=aoi_name,
                         tile_id=tile_id,
                         output_path=output_path,
                         split_view_tiles_months=["06","09"])
    
    logger.info("Config JSON saved successfully. Starting LanceDB routine...")
    
    try:
        ##Load Lance DB Routine
        table_data = combined_gdf.copy()
        logger.info(f"DataFrame columns: {table_data.columns.tolist()}")
        logger.info(f"DataFrame shape: {table_data.shape}")
        
        #load embeddings
        emb_type="cls"
        logger.info(f"Using embedding type: {emb_type}")
        
        # Check if column exists
        if f"s3_location_{emb_type}_emb" not in table_data.columns:
            logger.error(f"Column s3_location_{emb_type}_emb not found in dataframe")
            logger.error(f"Available columns: {table_data.columns.tolist()}")
            raise KeyError(f"Column s3_location_{emb_type}_emb not found")
        
        # Process embeddings in batches to avoid memory issues
        batch_size = 10000  # Adjust based on your memory constraints
        max_concurrent_workers = min(num_cpus, 8)  # Limit concurrent workers
        
        logger.info(f"Starting to process embeddings in batches of {batch_size} with {max_concurrent_workers} workers")
        start_time = time.time()
        
        try:
            table_data = process_in_batches(
                table_data=table_data,
                emb_type=emb_type,
                batch_size=batch_size,
                max_workers=max_concurrent_workers
            )
            logger.info(f"Embedding processing completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error processing embeddings in batches: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        logger.info(f"Embedding vectors added to dataframe")
        
        # Convert geometry to WKB and create metadata
        logger.info("Preparing data for LanceDB...")
        try:
            table_data['bbox'] = table_data['bbox'].apply(lambda x: x.bounds)
            table_data["date"] = table_data["date"].apply(lambda x: x.date())
            # Ensure vector is formatted as fixed_size_list, can achieve this by naming column "vector"
            table_data["vector"] = table_data[f"{emb_type}_embeddings"]
            table_data.drop(columns=[f"{emb_type}_embeddings"], inplace=True)
            
            # Remove rows with null vectors to avoid LanceDB errors
            initial_count = len(table_data)
            table_data = table_data.dropna(subset=["vector"])
            logger.info(f"Removed {initial_count - len(table_data)} rows with null vectors")
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Connect to LanceDB (this will create the database if it doesn't exist)
        logger.info(f"Connecting to LanceDB at {lancedb_uri}")
        try:
            db = lancedb.connect(lancedb_uri)
            logger.info("Successfully connected to LanceDB")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # Create a new table or overwrite if it already exists
        table_name = "vector_db"
        logger.info(f"Creating/overwriting table '{table_name}'")
        try:
            lance_table = db.create_table(table_name, data=table_data, mode='overwrite', on_bad_vectors="drop")
            logger.info(f"Table '{table_name}' created successfully in LanceDB.")
            logger.info(f"Number of rows inserted: {lance_table.count_rows()}")
            logger.info(lance_table.schema)
        except Exception as e:
            logger.error(f"Error creating LanceDB table: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    except Exception as e:
        logger.error(f"Overall error in LanceDB routine: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    # Note: for now we are not creating an index. This means LanceDB will conduct a full KNN search.
    # As the dataset becomes larger >>100k, consider creating and index to achieve better search latencies via Approximate 
    # see here: https://lancedb.github.io/lancedb/ann_indexes/

    logger.info("Starting chip_grid geojson generation...")
    try:
        ##Generate a chip_grid geojson
        # List all meta files in the local directory
        meta_files = [os.path.join(input_path_meta, f) for f in os.listdir(input_path_meta) if f.endswith('.parquet')]
        logger.info(f"Found {len(meta_files)} meta files")
        
        # Use ThreadPoolExecutor to parallelize the reading of files
        geodataframes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(read_parquet_file, meta_files))
        # Filter out None results (files that couldn't be read)
        geodataframes = [gdf for gdf in results if gdf is not None]
        logger.info(f"Successfully read {len(geodataframes)} meta files")
        
        # Combine all GeoDataFrames into one
        if geodataframes:
            meta_gdf = gpd.GeoDataFrame(pd.concat(geodataframes, ignore_index=True))
            meta_gdf.reset_index(inplace=True, drop=True)
            logger.info(f"Combined meta GeoDataFrame has {len(meta_gdf)} rows")
            
            logger.info(f"Reading AOI from {input_path_aoi}/aoi.geojson")
            aoi = gpd.read_file(f'{input_path_aoi}/aoi.geojson')
            
            full_tile_grid_gdf = meta_gdf[(meta_gdf["date"]==meta_gdf["date"].max())]
            logger.info(f"Filtered meta GDF to {len(full_tile_grid_gdf)} rows for latest date")
            
            crs = full_tile_grid_gdf.crs
            full_tile_grid_gdf.rename(columns={"geometry":"bbox"},inplace=True)
            full_tile_grid_gdf = full_tile_grid_gdf.set_geometry(col="bbox",crs="EPSG:4326")
            full_tile_grid_gdf = full_tile_grid_gdf[["chip_id","aoi_name","bbox"]]
            
            output_file = f"{output_path}/chip_grid_full_s2_tile.geojson"
            logger.info(f"Saving full tile grid to {output_file}")
            full_tile_grid_gdf.to_file(output_file, driver="GeoJSON")
            
            logger.info("Calculating intersection with AOI")
            intersection_gdf = gpd.overlay(full_tile_grid_gdf, aoi, how='intersection')
            intersected_chips = intersection_gdf["chip_id"].unique()
            logger.info(f"Found {len(intersected_chips)} chips intersecting with AOI")
            
            chip_grid_gdf = full_tile_grid_gdf[full_tile_grid_gdf["chip_id"].isin(intersected_chips)==True]    
            output_file = f"{output_path}/chip_grid_aoi.geojson"
            logger.info(f"Saving AOI chip grid to {output_file}")
            chip_grid_gdf.to_file(output_file, driver="GeoJSON")
            logger.info("Chip grid geojson files successfully created")
        else:
            logger.error("No geodataframes were successfully read from meta files")
            
    except Exception as e:
        logger.error(f"Error generating chip grid geojson: {str(e)}")
        logger.error(traceback.format_exc())
        raise