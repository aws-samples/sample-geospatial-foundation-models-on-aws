import sys
sys.path.append("/home/sagemaker-user/clay-model")
import argparse
import os
import torch
import boto3
import pandas as pd
import logging
from src.model import ClayMAEModule
import os
import io
import geopandas as gpd
import yaml
from box import Box
from shapely.geometry import box
from torchvision.transforms import v2
import xarray as xr
import numpy as np
import math
import pickle
from s3fs.core import S3FileSystem
import fsspec
s3 = boto3.client('s3')
s3fs = S3FileSystem()

import warnings
warnings.filterwarnings('ignore') # Ignore all warnings


# Fixed paths
CHECKPOINT_PATH = "/home/sagemaker-user/clay-model/checkpoints/clay-v1-base.ckpt"
METADATA_PATH = "/home/sagemaker-user/clay-model/configs/metadata.yaml"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)

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

logger = get_logger(logging.INFO) #INFO

def get_all_s3_keys(bucket, prefix):
    s3 = boto3.client('s3')
    keys = []
    # Initialize the paginator
    paginator = s3.get_paginator('list_objects_v2')
    # Create a PageIterator from the Paginator
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    # Iterate through each page
    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                keys.append(obj["Key"])
    return keys

def read_csv_from_s3(bucket_name, file_key):
    # Create an S3 client
    s3_client = boto3.client('s3')
    try:
        # Download the file contents
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        # Read the CSV content
        csv_content = response['Body'].read().decode('utf-8')
        # Use pandas to read the CSV data
        df = pd.read_csv(io.StringIO(csv_content))
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def saveEmbeddingsToS3(s3_bucket, np_object, s3_output_object):
    with s3fs.open('{}/{}'.format(s3_bucket, s3_output_object), 'wb') as f:
        f.write(pickle.dumps(np_object))

def model_fn(model_dir="./"):
    # As we want to visualize the embeddings from the model,
    # we neither mask the input image or shuffle the patches
    model = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=os.path.join(model_dir, CHECKPOINT_PATH),
        metadata_path=os.path.join(model_dir, METADATA_PATH),
        mask_ratio=0.0,
        shuffle=False,
    )
    model.eval();
    model = model.to(device)
    return model

def stack_chips_by_time(chip_meta_df):
    #load all chip scenes
    scenes=[]
    for f in list(chip_meta_df["s3_location_netcdf"]): #first 2 files
        with fsspec.open(f) as f:
            scene = xr.open_dataset(f,decode_coords="all")
        scenes.append(scene)

    #concatenate chips to cube
    s2_chip_cube=xr.concat(objs=scenes, coords="minimal", dim="time",join='outer')
    s2_chip_cube=s2_chip_cube.sortby("time")

    return s2_chip_cube

def get_chip_metadata_local(meta_path,max_chip_cloud_cover_perc):

    meta_files = pd.read_parquet(meta_path)
    meta_files.reset_index(drop=True, inplace=True)

    #only retain files without missing data
    meta_files = meta_files[meta_files["missing_data_perc"]<=0]

    #only retain files with cloud cover
    meta_files = meta_files[meta_files["cloud_cover_perc"]<=max_chip_cloud_cover_perc]

    #get retained xy chip coord combinations
    meta_files["chip_id"] = meta_files.apply(lambda df: "{}_{}".format(df.x_dim,df.y_dim),axis=1)
    
    meta_files["date"] = pd.to_datetime(meta_files["date"])
    
    meta_files.sort_values(by="date",ascending=True, inplace=True)

    return meta_files

#see here: https://github.com/Clay-foundation/stacchip/blob/ef49c23584cb975ba92d5ce969498262d0f65d7a/stacchip/processors/prechip.py#L54C1-L59C78
def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

def normalize_latlon(lon,lat):
    #lon = bounds[0] + (bounds[2] - bounds[0]) / 2
    #lat = bounds[1] + (bounds[3] - bounds[1]) / 2
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

def get_sensor_properties(bands=["red", "green", "blue", "nir"]):
    platform="sentinel-2-l2a"
    metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))
    mean = []
    std = []
    waves = []
    #bands = ["red", "green", "blue", "nir"]
    for band_name in bands:
        mean.append(metadata[platform].bands.mean[band_name])
        std.append(metadata[platform].bands.std[band_name])
        waves.append(metadata[platform].bands.wavelength[band_name])
    return mean,std,waves

def compile_img_tensor(chip_cube,bands=["red", "green", "blue", "nir"]): 
    
    band_name_dict = {
    'coastal': 'B01', #not used in pretraining
    'blue': 'B02',
    'green': 'B03',
    'red': 'B04',
    'rededge1': 'B05',
    'rededge2': 'B06',
    'rededge3': 'B07',
    'nir': 'B08',
    'nir08': 'B8A',
    'swir09': 'B09', #not used in pretraining
    'swir14': 'B10', #not used in pretraining
    'swir16': 'B11',
    'swir22': 'B12',
    'sceneclass': 'SCL' #not used in pretraining
    }
    
    bands_s2 = [band_name_dict[b] for b in bands]
    img_dataset=chip_cube[bands_s2]
    concatenated_array = np.stack([img_dataset.sel(time=t).to_array() for t in img_dataset.time.values], axis=0)

    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(concatenated_array.astype(np.float32))
    return img_tensor

def prepare_clay_datacube_v2(week_norm, hour_norm, lat_norm, lon_norm, pixels, waves):
    pixel_resolution = 10
    platform = "sentinel-2-l2a"  
    
    device="cuda"
    clay_datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=device,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(10, device=device),
        "waves": torch.tensor(waves, device=device),
    }
    
    return clay_datacube

def prep_inputs_v2(chip_cube,bands=["red", "green", "blue", "nir"],platform="sentinel-2-l2a"):
        
    # Extract mean, std, and wavelengths from metadata
    metadata = Box(yaml.safe_load(open(METADATA_PATH)))
    mean = []
    std = []
    waves = []
    for band_name in bands:
        mean.append(metadata[platform].bands.mean[band_name])
        std.append(metadata[platform].bands.std[band_name])
        waves.append(metadata[platform].bands.wavelength[band_name])

    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )
    
    logger.info("{},{},{}".format(mean,std,waves))
    
    datetimes = pd.to_datetime(chip_cube.time.values)
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]
    #print(week_norm,hour_norm)

    min_x, min_y, max_x, max_y = chip_cube.rio.bounds()
    lon, lat = box(min_x, min_y, max_x, max_y).centroid.coords[0]
    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]
    
    final_pixels = compile_img_tensor(chip_cube,bands=bands)
    final_pixels = transform(final_pixels)
    logger.info("Generated image tensor of shape: {}".format(final_pixels.shape))
    
    clay_datacube = prepare_clay_datacube_v2(week_norm, hour_norm, lat_norm, lon_norm, final_pixels, waves)

    return clay_datacube


def predict_fn_v2(input_data,model):
    """
    Generate embeddings from the model.

    Parameters:
    model (ClayMAEModule): The pretrained model.
    datacube (dict): Prepared data cube.

    Returns:
    numpy.ndarray: Generated embeddings.
    """
    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(input_data)
        
    csl_token=unmsk_patch[:, 0, :].cpu().numpy()
    patch_embeddings=unmsk_patch[:, 1:, :].detach().cpu().numpy()

    # The first embedding is the class token, which is the
    # overall single embedding.
    return csl_token, patch_embeddings


###############

def main(args):
    logger.info("Starting the processing job...")
    
    #get arguments
    logger.info(args)
    bucket_name = args.BUCKET_NAME
    batch_size = args.BATCH_SIZE
    aoi_name=args.AOI_NAME
    processing_level = args.PROCESSING_LEVEL
    chip_size=args.CHIP_SIZE
    max_chip_cloud_cover_perc = args.MAX_CLOUD_COVER_PERC
    bands = args.S2_BANDS.split("_")
    
    meta_path_input = '/opt/ml/processing/input/meta'
    meta_path_output = '/opt/ml/processing/output/meta'
    chip_id_output = '/opt/ml/processing/output/chip_ids'
    
    chips_meta_df = get_chip_metadata_local(meta_path_input,max_chip_cloud_cover_perc)
    
    model = model_fn()
    logger.info("Clay Model instantiated")
    
    mgrs_grid_id = chips_meta_df["origin_tile"].iloc[-1].split("_")[1]
    
    logger.info("Chips to be processed:{}".format(len(chips_meta_df)))
    logger.info("Unique Chip IDs to be processed:{}".format(len(chips_meta_df["chip_id"].unique())))
    
    for xy in chips_meta_df["chip_id"].unique():
        
        logger.info(f"Started processing chip {xy}...")
        
        selected_chips_df=chips_meta_df[chips_meta_df["chip_id"]==xy]
        num_samples = len(selected_chips_df)
        total_batch_num = math.ceil(len(selected_chips_df)/batch_size)
        logger.info("Running {} batches of size {} with a total of {} samples".format(total_batch_num,batch_size,num_samples))
        
        #create batches at the chip level
        selected_chips_batches_df = [selected_chips_df.iloc[i:i + batch_size] for i in range(0, len(selected_chips_df), batch_size)]
        logger.info("Chip paths batched into {} batches".format(len(selected_chips_batches_df)))
        
        class_emb_all_batches_ls = []
        patch_emb_all_batches_ls = []
        batch_counter=1
        
        for chip_batch_df in selected_chips_batches_df: #OPTIONAL: no need to batch this by chip, potentially drop
            
            logger.info("Running batch No: {}".format(batch_counter))

            logger.info("Dates in batch: {}".format(chip_batch_df["date"].values))

            chip_batch_df.sort_values(by="date",inplace=True)

            chip_file_paths = list(chip_batch_df["s3_location_netcdf"])
            dates = list(chip_batch_df["date"])
            origin_tiles = list(chip_batch_df["origin_tile"])
            
            #generate combined xarray dataset per chip by stacking by time
            s2_chip_cube = stack_chips_by_time(chip_batch_df)
            logger.info("Data cube size: {}".format(s2_chip_cube.sizes))
            
            try:
                
                #batch together
                input_batch = prep_inputs_v2(chip_cube=s2_chip_cube,bands=bands)
                logger.info("Input data prepared...")

                #generate embeddings
                multi_cls_emb, multi_patch_embed = predict_fn_v2(input_batch, model)

                logger.info("{}".format(multi_cls_emb))

                assert len(multi_cls_emb)==len(chip_file_paths), "Number of embeddings generated and number of chips are not the same!"
                logger.info("Embeddings generated...")

                
            except:
                
                logger.info("Batch failed...")
                               
                # Create NaN-filled embeddings matching expected dimensions
                num_chips = len(chip_file_paths)
                
                nan_cls = np.full(768, np.nan)  # CLS token embedding shape
                nan_patches = np.full((1024, 768), np.nan)  # Patch embeddings shape

                multi_cls_emb = [nan_cls.copy() for _ in range(num_chips)]
                multi_patch_embed = [nan_patches.copy() for _ in range(num_chips)]

                logger.warning(f"Generated {len(multi_cls_emb)} NaN embeddings for failed batch")
                
                
            # Zip the arrays together
            batch_results_class_emb = list(zip(chip_file_paths, dates, origin_tiles, multi_cls_emb))
            class_emb_df = pd.DataFrame(batch_results_class_emb, columns=['s3_location_netcdf', 'date', 'origin_tile','cls_token_embeddings'])
            class_emb_all_batches_ls.append(class_emb_df)

            batch_res_patch_emb = list(zip(chip_file_paths, dates, origin_tiles,multi_patch_embed))
            patch_emb_df = pd.DataFrame(batch_res_patch_emb, columns=['s3_location_netcdf', 'date', 'origin_tile','patch_embeddings'])
            patch_emb_all_batches_ls.append(patch_emb_df)

            batch_counter+=1
        
        #save results as individual files
        class_emb_all_batches_df = pd.concat(class_emb_all_batches_ls)
        patch_emb_all_batches_df = pd.concat(patch_emb_all_batches_ls)
        
        row_dicts = [row.to_dict() for index, row in class_emb_all_batches_df.iterrows()]
        # Save each row dictionary as a .npy file
        for index, row_dict in enumerate(row_dicts):
            origin_tile=row_dict["origin_tile"]
            year=pd.to_datetime(row_dict["date"]).year
            month=pd.to_datetime(row_dict["date"]).month
            file_path=f"output/{aoi_name}/embeddings/{processing_level}/{mgrs_grid_id}/{year}/{month}/{origin_tile}_{chip_size}_chip_cls_embeddings_{bands}_{xy}.npy"
            saveEmbeddingsToS3(s3_bucket=bucket_name,np_object=row_dict,s3_output_object=file_path)

        row_dicts = [row.to_dict() for index, row in patch_emb_all_batches_df.iterrows()]
        # Save each row dictionary as a .npy file
        for index, row_dict in enumerate(row_dicts):
            origin_tile=row_dict["origin_tile"]
            year=pd.to_datetime(row_dict["date"]).year
            month=pd.to_datetime(row_dict["date"]).month
            file_path=f"output/{aoi_name}/embeddings/{processing_level}/{mgrs_grid_id}/{year}/{month}/{origin_tile}_{chip_size}_chip_patch_embeddings_{bands}_{xy}.npy"
            saveEmbeddingsToS3(s3_bucket=bucket_name,np_object=row_dict,s3_output_object=file_path)

        logger.info("Embeddings saved to S3...")
        logger.info(f"Finished processing chip {xy}...")
                    
    #update metadata          
    #get all parquet files in the directory
    meta_files = [f for f in os.listdir(meta_path_input) if f.endswith('.parquet')]
    
    for file_name in meta_files:
        file_in_path = os.path.join(meta_path_input, file_name)

        # Read the Parquet file using GeoPandas
        gdf = gpd.read_parquet(file_in_path)
        
        # apply filters as above
        gdf.reset_index(drop=True, inplace=True)
        #some processing
        gdf["date"] = pd.to_datetime(gdf["date"])
        gdf["year"] = gdf["date"].apply(lambda x: x.year)
        gdf["month"] = gdf["date"].apply(lambda x: x.month)
        gdf["chip_id"] = gdf.apply(lambda x: "{}_{}".format(x.x_dim,x.y_dim),axis=1)
        chip_embeddings_path = f"output/{aoi_name}/embeddings/{processing_level}/{mgrs_grid_id}/"
        gdf["s3_location_cls_emb"] = gdf.apply(lambda x: f"s3://{bucket_name}/{chip_embeddings_path}{x.year}/{x.month}/{x.origin_tile}_{x.chip_size}_chip_cls_embeddings_{bands}_{x.chip_id}.npy" if x.missing_data_perc==0 else np.nan,axis=1)         
        gdf["s3_location_patch_emb"] = gdf.apply(lambda x: f"s3://{bucket_name}/{chip_embeddings_path}{x.year}/{x.month}/{x.origin_tile}_{x.chip_size}_chip_patch_embeddings_{bands}_{x.chip_id}.npy" if x.missing_data_perc==0 else np.nan,axis=1)

        # write to output path
        file_out_path=os.path.join(meta_path_output, file_name)
        gdf.to_parquet(file_out_path)
    
    logger.info("Meta files updated...")
    
    #create text files per each unique chip for which embedding have been generated (for sharding at later stage)
    def save_txt(path,content):
        # Open a file in write mode and save the string
        with open(path, "w") as file:
            file.write(content)
            
    for c in gdf["chip_id"].unique():
        content = str(c)
        file_name = f"{c}.txt"
        path = os.path.join(chip_id_output,file_name)
        save_txt(path,content)

    logger.info("Chip id text files saved...")
    logger.info("All done...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--BUCKET_NAME", type=str, default="")
    parser.add_argument("--AOI_NAME",type=str, default="")
    parser.add_argument("--BATCH_SIZE",type=int, default=64)
    parser.add_argument("--PROCESSING_LEVEL", type=str, default="l2a")
    parser.add_argument("--CHIP_SIZE",type=int, default=256)             
    parser.add_argument("--MAX_CLOUD_COVER_PERC", type=float, default=1.0)
    parser.add_argument("--S2_BANDS", type=str, default="red_green_blue_nir") #comma-seperated
    args = parser.parse_args()
    
    # Needed for GDAL CLI operations
    os.environ['PROJ_LIB'] = '/opt/conda/share/proj'
    os.environ['GDAL_DATA'] = '/opt/conda/share/gdal'
    os.environ['GDAL_CONFIG'] = '/opt/conda/bin/gdal-config'
    os.environ['GEOS_CONFIG'] = '/opt/conda/bin/geos-config'
    
    main(args)
