import argparse
import boto3
from io import BytesIO
import os
import sys
import subprocess
import fsspec
import warnings
import time
import numpy as np
import geopandas as gpd
import xarray as xr
import gc
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import subprocess
import boto3
import numpy as np
from io import BytesIO
from botocore.exceptions import ClientError
import time
import concurrent.futures
from collections import defaultdict
import gc

import warnings
warnings.filterwarnings('ignore') # Ignore all warnings

import multiprocessing
num_cpus = multiprocessing.cpu_count() # Get the number of CPUs available on the system

# Create a global S3 client with optimized settings
s3_client = boto3.client('s3', config=boto3.session.Config(
    max_pool_connections=50,
    retries={'max_attempts': 10}
))

bucket_name = "gfm-demo-bucket"

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

def readEmbeddingsFromS3(s3_bucket, s3_output_object):
    # Initialize a session using your credentials
    s3_client = boto3.client('s3')

    # Get the object from the S3 bucket
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_output_object)

    # Read the object's content into a NumPy array
    file_content = response['Body'].read()
    np_array = np.load(BytesIO(file_content), allow_pickle=True)

    return np_array

def load_embeddings(s3_paths):
    chip_embeddings = {}
    for f in s3_paths:
        try:
            bucket = f.split("/")[2]
            #print(bucket)
            obj_path = "/".join(f.split("/")[3:])
            emb = readEmbeddingsFromS3(bucket, obj_path)
            chip_embeddings[emb['s3_location_netcdf']] = emb
        except Exception as e:
            logger.debug(f"Exception: {e}")
            logger.debug(f"No embeddings for file {f}")
            
    return chip_embeddings

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
            logger.debug(f"Error reading embeddings from S3: {e}")
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

def load_embeddings_parallelized(s3_paths,max_workers=num_cpus):
    chip_embeddings = defaultdict(dict)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(load_single_embedding, f): f for f in s3_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            result = future.result()
            if result:
                key, emb = result
                chip_embeddings[key] = emb
    return dict(chip_embeddings)

def get_s2_cube(file_paths_list):
    scenes=[]
    for f in file_paths_list:#list(chip_meta["s3_location_netcdf"]): 
        with fsspec.open(f"{f}", mode="rb") as f:
            scene = xr.open_dataset(f, decode_coords="all")
        scenes.append(scene)
    #generate cube
    s2_chip_cube=xr.concat(objs=scenes, coords="minimal", dim="time",join='outer')
    s2_chip_cube = s2_chip_cube.sortby("time")
    return s2_chip_cube

def get_s2_cube_v2(file_paths_list):
    scenes = []
    s3_client = boto3.client('s3')
    
    for file_path in file_paths_list:
        # Parse S3 URI
        bucket_name, object_key = parse_s3_uri(file_path)
        
        # Download file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read()
        
        # Open dataset from memory and immediately delete the file content
        with BytesIO(file_content) as f:
            scene = xr.open_dataset(f, decode_coords="all")
            scene = scene.load()  # Load all data into memory
        
        scenes.append(scene)
        del file_content  # Explicitly delete the file content
    
    # Generate cube
    s2_chip_cube = xr.concat(objs=scenes, coords="minimal", dim="time", join='outer')
    s2_chip_cube = s2_chip_cube.sortby("time")
    return s2_chip_cube

def parse_s3_uri(uri):
    # Remove 's3://' prefix
    path = uri.replace('s3://', '')
    
    # Split into bucket and key
    parts = path.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    return bucket, key

def detect_outliers_by_year(df, column, year_column='year', threshold=2):
    """
    Detects outliers in each year separately based on a specified number of standard deviations.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column (str): The column name for which to detect outliers.
    year_column (str): The column name for the year.
    threshold (float): The number of standard deviations to use for outlier detection.

    Returns:
    DataFrame: A DataFrame with outliers removed.
    """
    def detect_outliers(group):
        mean = group[column].mean()
        std_dev = group[column].std()
        return group[(group[column] < mean - threshold * std_dev) | (group[column] > mean + threshold * std_dev)]

    # Group by year and apply outlier detection
    outliers = df.groupby(year_column).apply(detect_outliers).reset_index(drop=True)
    
    return outliers["date"]


def run_dimensionality_reduction(chip_meta,method='pca', emb_type="patch",n_components=3):
    embeddings = np.array([emb.flatten() for emb in chip_meta[f'{emb_type}_embeddings']])
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    else:
        raise ValueError("Invalid method. Choose 'pca', 'tsne', or 'umap'.")

    reduced_embeddings = reducer.fit_transform(embeddings)
    for i in range(n_components):
        chip_meta[f'{emb_type}_emb_{method}_{i+1}'] = reduced_embeddings[:, i]
    return chip_meta

#MAIN
if __name__ == "__main__":

    session = boto3.Session() 
    logger = get_logger(logging.INFO) #INFO
    parser = argparse.ArgumentParser()
    #Parameters
    parser.add_argument("--AOI_NAME", type=str, default="")
    parser.add_argument("--S2_TILE_ID", type=str, default="")
    parser.add_argument("--BUCKET_NAME", type=str, default="")
    parser.add_argument("--EMBEDDING_TYPE", type=str, default="patch")
    parser.add_argument("--CHIP_MAX_CLOUD", type=float, default=0.05)
    
    args = parser.parse_args()
    aoi_name=args.AOI_NAME
    s2_tile_id=args.S2_TILE_ID
    emb_type = args.EMBEDDING_TYPE
    max_cloud = args.CHIP_MAX_CLOUD
    bucket_name=args.BUCKET_NAME

    #####
    #read meta file
    meta_path_input = '/opt/ml/processing/input/meta'
    unique_chips_input = '/opt/ml/processing/input/chip_ids'
    unique_chips_output = '/opt/ml/processing/output'
    subprocess.check_call(["sudo","chown","-R","sagemaker-user", unique_chips_output]) #ensure write permissions on output folder
    
    #get list of unique chips
    # List all files in the folder
    unique_chip_files = os.listdir(unique_chips_input) #use this for distribution by chipID if desired
    unique_chips = [i.split(".")[0] for i in unique_chip_files]
    logger.info("{} unique chip IDs received...".format(len(unique_chips)))
    
    #load meta data and limit to chips in scope
    meta_df = gpd.read_parquet(meta_path_input) #can read multiple files simultaneously!
    meta_df = meta_df[(meta_df["cloud_cover_perc"]<=max_cloud) & (meta_df["missing_data_perc"]==0)]
    meta_df = meta_df[meta_df["chip_id"].isin(unique_chips)==True]

    logger.info("Metadata retrieved...")
    
    #get embeddings and merge
    s3_paths=list(meta_df[f"s3_location_{emb_type}_emb"])        

    #load embeddings and add to df
    chip_embeddings_dict = load_embeddings_parallelized(s3_paths,max_workers=num_cpus)
    logger.info("{} embedding vectors loaded".format(len(chip_embeddings_dict)))

    def add_embedding_vector(data_cube_path,chip_embeddings_dict,emb_key):
        try:
            emb = chip_embeddings_dict[data_cube_path][emb_key]
            return emb
        except:
            return np.nan

    if emb_type=="cls":
        meta_df[f"{emb_type}_embeddings"] = meta_df['s3_location_netcdf'].apply(lambda x: add_embedding_vector(x,chip_embeddings_dict,'cls_token_embeddings'))
    elif emb_type=="patch":
        meta_df[f"{emb_type}_embeddings"] = meta_df['s3_location_netcdf'].apply(lambda x: add_embedding_vector(x,chip_embeddings_dict,'patch_embeddings'))
    logger.info("Embeddings added to gdf...")
    
    #actively delete the chip_embeddings_dict to free up memory
    chip_embeddings_dict=None
    gc.collect()
    
    #add embeddings==nan indicator
    meta_df["embedding_contains_nan_yn"] = meta_df[f"{emb_type}_embeddings"].apply(lambda x:np.isnan(x).any())
    logger.info("Number of chips with nan embeddings detected: {}".format(meta_df["embedding_contains_nan_yn"].sum()))
    #remove emebdding verctors with nan (will throw error in PCA!)
    meta_df = meta_df[meta_df["embedding_contains_nan_yn"]==False]
    
    # Run PCA 
    #try:
    meta_df=run_dimensionality_reduction(meta_df,method="pca",n_components = 2)
    logger.info(f"PCA computed...")
    #except:
    #logger.info(f"PCA failed...")
    #    pass

    # Run t-sne
    #try:
    meta_df=run_dimensionality_reduction(meta_df,method="tsne",n_components = 2)
    logger.info(f"t-sne computed...")
    #except:
    #logger.info(f"t-sne failed...")
    #    pass
    
    #do chip-level analyses (for cosine sim)    
    for c in unique_chips:
        #get chip meta
        chip_meta = meta_df[meta_df["chip_id"]==c]
        if len(chip_meta)>0:
            #Compute cosine similarity
            #get baseline chip (for cosine similarity)
            date_baseline = chip_meta["date"].min()
            baseline_embeddings = chip_meta[chip_meta["date"]==date_baseline][f"{emb_type}_embeddings"].to_numpy()[0].flatten()
            #calc cosine sim
            y_col="cosine_sim"
            X=chip_meta[chip_meta["date"]==date_baseline][f"{emb_type}_embeddings"].to_numpy()[0].flatten().reshape(1, -1)            
            chip_meta[y_col] = chip_meta[f"{emb_type}_embeddings"].apply(lambda x: cosine_similarity(X,x.flatten().reshape(1, -1))[0][0])
            
        #drop embedding vector from file
        chip_meta.drop(columns=[f"{emb_type}_embeddings"],inplace=True)

        #save output      
        fname=f"{c}_processed_emb.parquet"
        file_out_path=os.path.join(unique_chips_output, fname)
        chip_meta.to_parquet(file_out_path)       
        logger.info(f"Output saved to {file_out_path}...")
   
    logger.info(f"cosine similarity computed...")
