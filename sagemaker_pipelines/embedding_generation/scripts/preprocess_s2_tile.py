import argparse
import boto3
import os
import sys
import subprocess
import json
import time
import psutil
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import gc
import logging
from PIL import Image
from io import BytesIO
import xarray as xr
import numpy as np
import dask.array as da
from retrying import retry
from boto3.s3.transfer import TransferConfig
from joblib import Parallel, delayed, parallel_backend

import warnings
warnings.filterwarnings('ignore') # Ignore all warnings

import multiprocessing
num_cpus = multiprocessing.cpu_count() # Get the number of CPUs available on the system


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

def s2_scene_id_to_cog_path(scene_id):
    """
    Generate s3 URL for S2 L2A product (i.e., the cloud-optimized GeoTiff) from tile ID
    """
    parts = scene_id.split("_")
    s2_qualifier = "{}/{}/{}/{}/{}/{}".format(
        parts[1][0:2],
        parts[1][2],
        parts[1][3:5],
        parts[2][0:4],
        str(int(parts[2][4:6])),
        "_".join(parts)
    )
    
    return f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/{s2_qualifier}"


def download_from_s3(s3_obj_url, local_dir, local_file_fn=None, requester_pays=False):
    """
    Download Sentinel 2 L1C product (as identified by s3_obj_url)
    to local file share
    """
    os.makedirs(local_dir, exist_ok=True)
    if local_file_fn is None:
        local_file_path = os.path.join(local_dir, s3_obj_url.split("/")[-1])
    else:
        local_file_path = local_file_fn(local_dir, s3_obj_url)

    target_bucket_name = s3_obj_url.split("/")[2]
    target_bucket_ob_key = "/".join(s3_obj_url.split("/")[3:])

    s3_bucket = session.resource("s3").Bucket(target_bucket_name)
    if requester_pays:
        s3_bucket.download_file(
            target_bucket_ob_key, local_file_path, ExtraArgs={"RequestPayer": "requester"}
        )
    else:
        s3_bucket.download_file(target_bucket_ob_key, local_file_path)
    return local_file_path


def s2_tile_id_to_s2l1c_s3_url(tile_id):
    """
    Generate s3 URL for S2 L1C product from tile ID
    """
    parts = tile_id.split("_")
    s2l1c_qualifier = "{}/{}/{}/{}/{}/{}/0".format(
        parts[1][0:2],
        parts[1][2],
        parts[1][3:5],
        parts[2][0:4],
        str(int(parts[2][4:6])),
        str(int(parts[2][6:8])),
    )
    return f"s3://sentinel-s2-l1c/tiles/{s2l1c_qualifier}/"


def get_s2l1c_band_data_xarray(s2_tile_id, band_id):
    """
    Download S2 L1C data for given tile ID, optionally apply cloud mask and clip to AOI (clip geometry),
    reproject to target WGS84 (EPSG:4326) coordinate reference system (CRSs)
    """

    data_dir = "data/s2l1c/"
    s3_prefix = s2_tile_id_to_s2l1c_s3_url(s2_tile_id)

    local_tile_path = download_from_s3(
        f"{s3_prefix}{band_id}.jp2",
        data_dir,
        local_file_fn=lambda local_dir, s3_obj_url: os.path.join(
            data_dir, f"{s2_tile_id}_{band_id}.jp2"
        ),
        requester_pays=True,
    )

    band_data_epsg_32631 = rioxarray.open_rasterio(local_tile_path, masked=True,band_as_variable=True)
    kwargs = {"nodata": np.nan}
    band_data_epsg_4326 = band_data_epsg_32631.rio.reproject("EPSG:4326", **kwargs)

    os.remove(local_tile_path)

    return band_data_epsg_4326

def scene_id_to_datetime(scene_id):
    dt = pd.to_datetime(scene_id.split("_")[-3])
    return dt

def resample_raster(input_raster, target_width, target_height, resampling_method=Resampling.bilinear):
    """
    Resample a raster dataset using rioxarray.
    """
    # Perform the resampling
    resampled_raster = input_raster.rio.reproject(
        input_raster.rio.crs,
        shape=(target_height, target_width),
        resampling=resampling_method
    )
    return resampled_raster

def get_aoi_cloud_free_ratio(SCL_raster):
    #get cloud-free ratio
    SCL_mask_pixel_count = SCL_raster.SCL.data.size - np.count_nonzero(np.isnan(SCL_raster.SCL.data)) # get size of SCL mask in num pixels (excl. any nans)
    SCL_classes_cloud_free = [4,5,6] # see here: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
    SCL_cloud_free_pixel_count = np.isin(SCL_raster.SCL.data,SCL_classes_cloud_free).sum() #count pixels that are non-cloud class
    
    if SCL_mask_pixel_count>0:
        cloud_free_ratio = SCL_cloud_free_pixel_count/SCL_mask_pixel_count
    else:
        cloud_free_ratio=0 #set to 0 if no data in bound!

    return cloud_free_ratio

def normalize(band):
    min_val = band.min().compute()
    max_val = band.max().compute()
    return ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def create_and_save_rgb_image(s2_datacube,file_path):
    
    data_array = s2_datacube
    red_band = data_array.B04
    green_band = data_array.B03
    blue_band = data_array.B02
    
    red_normalized = normalize(red_band)
    green_normalized = normalize(green_band)
    blue_normalized = normalize(blue_band)
    
    rgb_array = da.stack([red_normalized, green_normalized, blue_normalized], axis=-1)
    
    img = Image.fromarray(rgb_array.compute(), 'RGB')
    
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG', optimize=True)
    img_byte_arr.seek(0)
    
    #save image locally to local_out_path_imgs
    
    # Save the image
    with open(file_path, 'wb') as f:
        f.write(img_byte_arr.getvalue()) 


def extract_bucket_and_key(s3_url):
    if s3_url.startswith('s3://'):
        url_parts = s3_url[5:].split('/', 1)
        bucket_name = url_parts[0]
        object_key = url_parts[1] if len(url_parts) > 1 else ''
        return bucket_name, object_key
    else:
        raise ValueError('Invalid S3 URL format')

def upload_to_s3(file_obj, bucket, key):
    s3_client = boto3.client('s3')
    config = TransferConfig(multipart_threshold=1024*25, max_concurrency=10)
    s3_client.upload_fileobj(file_obj, bucket, key, ExtraArgs={'ContentType': 'image/png'}, Config=config)

#MAIN
INSTANCE_CONFIG = {
    'total_cpus': num_cpus,
    'target_cpu_usage': 0.8,  # 80% target utilization
    'memory_limit': 0.9,      # 90% memory limit
    'scene_parallel_factor': 0.25,  # Use 25% of CPUs for scene-level parallelism
    'band_parallel_minimum': 2,     # Minimum CPUs per band processing
    'chip_parallel_factor': 0.5     # Use 50% of available CPUs per scene for chipping
}

def get_logger(level):
    """Configure logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    return logger

def calculate_parallel_jobs(n_scenes):
    """Calculate optimal number of parallel jobs for different processing levels"""
    config = INSTANCE_CONFIG
    max_scene_jobs = int(config['total_cpus'] * config['scene_parallel_factor'])
    scene_level_jobs = min(n_scenes, max_scene_jobs)
    
    cpus_per_scene = config['total_cpus'] // scene_level_jobs
    band_level_jobs = max(
        config['band_parallel_minimum'],
        int(cpus_per_scene * (1 - config['chip_parallel_factor']))
    )
    chip_level_jobs = max(
        2,
        int(cpus_per_scene * config['chip_parallel_factor'])
    )
    
    return scene_level_jobs, band_level_jobs, chip_level_jobs

def monitor_resources():
    """Monitor CPU and memory usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    logger.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%")

def check_memory_usage():
    """Check if memory usage is too high"""
    memory_percent = psutil.virtual_memory().percent
    return memory_percent > INSTANCE_CONFIG['memory_limit'] * 100

def process_single_cube(x, y, ds, cube_size, aoi_name, s2_scene_id, local_out_path_chips, 
                       local_out_path_imgs, s3_output_path_chips, s3_output_path_imgs, wgs84):
    """Process a single cube with improved error handling"""
    try:
        # Select a NxN-sized cube
        cube = ds.isel(x=slice(x, x + cube_size), y=slice(y, y + cube_size))
        
        if cube.sizes['x'] < cube_size or cube.sizes['y'] < cube_size:
            return None  # Skip incomplete cubes at edges
        
        grid_id = s2_scene_id.split("_")[1]
        date = scene_id_to_datetime(s2_scene_id)
        year = date.year
        month = date.month
        
        # Create directories
        directory_chips = f"{local_out_path_chips}{grid_id}/{year}/{month}/"
        directory_imgs = f"{local_out_path_imgs}{grid_id}/{year}/{month}/"
        os.makedirs(directory_chips, exist_ok=True)
        os.makedirs(directory_imgs, exist_ok=True)
        
        # Save netcdf with retry
        file_prefix_nc = f"{s2_scene_id}_{cube_size}_chipped_{x//cube_size}_{y//cube_size}.nc"
        full_path_nc = f"{directory_chips}{file_prefix_nc}"
        s3_path_nc = f"{s3_output_path_chips}{grid_id}/{year}/{month}/{file_prefix_nc}"
        
        retry_count = 0
        while retry_count < 2:
            try:
                cube.to_netcdf(full_path_nc)
                break
            except Exception as e:
                retry_count += 1
                if retry_count == 2:
                    raise
                time.sleep(1)
        
        # Save PNG with retry
        file_prefix_png = f"{s2_scene_id}_{cube_size}_rgb_thumbnail_{x//cube_size}_{y//cube_size}.png"
        full_path_png = f"{directory_imgs}{file_prefix_png}"
        s3_path_png = f"{s3_output_path_imgs}{grid_id}/{year}/{month}/{file_prefix_png}"
        
        retry_count = 0
        while retry_count < 2:
            try:
                create_and_save_rgb_image(s2_datacube=cube, file_path=full_path_png)
                break
            except Exception as e:
                retry_count += 1
                if retry_count == 2:
                    raise
                time.sleep(1)
        
        # Create cube metadata
        cube_dict = {
            "x_dim": x//cube_size,
            "y_dim": y//cube_size,
            "date": date,
            "aoi_name": aoi_name,
            "origin_tile": s2_scene_id,
            "chip_size": cube_size,
            "file_name": file_prefix_nc,
            "bbox": cube.rio.bounds(),
            "crs": wgs84,
            "cloud_cover_perc": 1-get_aoi_cloud_free_ratio(cube),
            "missing_data_perc": 1-int(cube.B02.count())/(cube_size*cube_size),
            "s3_location_netcdf": s3_path_nc,
            "s3_location_png_thumbnail": s3_path_png
        }
        
        return cube_dict
    
    except Exception as e:
        logger.error(f"Error processing cube at x={x}, y={y}: {str(e)}")
        return None

def generate_chips_from_xarray(ds, cube_size, aoi_name, s2_scene_id, 
                             local_out_path_chips, local_out_path_imgs,
                             s3_output_path_chips, s3_output_path_imgs):
    """Generate chips with optimized parallel processing"""
    try:
        # Calculate optimal number of jobs for chipping
        _, _, chip_level_jobs = calculate_parallel_jobs(1)  # 1 scene at a time
        logger.info(f"Using {chip_level_jobs} parallel jobs for chip generation")
        
        # Generate all x,y coordinates for processing
        coordinates = [(x, y) 
                      for x in range(0, ds.sizes['x'], cube_size)
                      for y in range(0, ds.sizes['y'], cube_size)]
        
        logger.info(f"Generating {len(coordinates)} chips for scene {s2_scene_id}")
        
        # Process cubes in parallel with memory protection
        with parallel_backend('threading', n_jobs=chip_level_jobs):
            results = Parallel(verbose=1)(
                delayed(process_single_cube)(
                    x, y, ds, cube_size, aoi_name, s2_scene_id,
                    local_out_path_chips, local_out_path_imgs,
                    s3_output_path_chips, s3_output_path_imgs, wgs84
                ) for x, y in coordinates
            )
        
        # Filter out None results and convert to DataFrame
        results = [r for r in results if r is not None]
        df = pd.DataFrame(results)
        
        logger.info(f"Successfully generated {len(results)} chips for scene {s2_scene_id}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in chip generation for scene {s2_scene_id}: {str(e)}")
        raise

def process_single_band(band_url, band_name):
    """Process a single band from COG URL"""
    try:
        band = rioxarray.open_rasterio(band_url, masked=True, band_as_variable=True)
        return band.rename(name_dict={"band_1": band_name})
    except Exception as e:
        logger.error(f"Error processing band {band_name}: {str(e)}")
        raise

def parallel_band_processing(s2_cog_prefix, band_configs, band_level_jobs):
    """Process multiple bands in parallel"""
    with parallel_backend('threading', n_jobs=band_level_jobs):
        results = Parallel(verbose=1)(
            delayed(process_single_band)(
                f"{s2_cog_prefix}/{band}.tif", 
                band
            ) for band, _ in band_configs
        )
    return results

def resample_raster(input_raster, target_width, target_height, resampling_method=Resampling.bilinear):
    """Resample a raster to target dimensions"""
    try:
        return input_raster.rio.reproject(
            input_raster.rio.crs,
            shape=(target_height, target_width),
            resampling=resampling_method
        )
    except Exception as e:
        logger.error(f"Error resampling raster: {str(e)}")
        raise

def parallel_resample(bands_to_resample, target_width, target_height, band_level_jobs):
    """Resample multiple bands in parallel"""
    with parallel_backend('threading', n_jobs=band_level_jobs):
        results = Parallel(verbose=1)(
            delayed(resample_raster)(
                band, 
                target_width, 
                target_height,
                Resampling.bilinear
            ) for band in bands_to_resample
        )
    return results

def bbox_to_polygon(bbox):
    minx, miny, maxx, maxy = tuple(bbox)
    return box(minx, miny, maxx, maxy)
    
@retry(stop_max_attempt_number=2, wait_fixed=2000)
def process_single_scene(item, args, output_path, band_level_jobs):
    """Process a single scene with retry"""
    scene_id = item["id"]
    try:
        if check_memory_usage():
            time.sleep(5)  # Wait for memory to free up
            
        start = time.time()
        monitor_resources()  # Monitor at start
        logger.info(f"Starting processing of scene: {scene_id}")
        
        grid_id = scene_id.split("_")[1]
        date = scene_id_to_datetime(scene_id)
        
        if args.S2_PROCESSING_LEVEL == "l2a":
            s2_cog_prefix = s2_scene_id_to_cog_path(scene_id)
            
            # Define band groups
            bands_10m = [("B02", "10m"), ("B03", "10m"), ("B04", "10m"), ("B08", "10m")]
            bands_20m = [("B05", "20m"), ("B06", "20m"), ("B07", "20m"), ("B8A", "20m"),
                        ("B11", "20m"), ("B12", "20m")]
            bands_60m = [("B01", "60m"), ("B09", "60m")]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Process bands in parallel
                logger.info(f"{scene_id}: Processing bands...")
                bands_10m_results = parallel_band_processing(s2_cog_prefix, bands_10m, band_level_jobs)
                bands_10m_data = dict(zip([b[0] for b in bands_10m], bands_10m_results))
                
                bands_20m_results = parallel_band_processing(s2_cog_prefix, bands_20m, band_level_jobs)
                bands_20m_data = dict(zip([b[0] for b in bands_20m], bands_20m_results))
                
                bands_60m_results = parallel_band_processing(s2_cog_prefix, bands_60m, band_level_jobs)
                bands_60m_data = dict(zip([b[0] for b in bands_60m], bands_60m_results))

                # Process SCL separately
                SCL = process_single_band(f"{s2_cog_prefix}/SCL.tif", "SCL")

                # Get target dimensions and resample
                target_width = int(bands_10m_data["B02"].rio.width)
                target_height = int(bands_10m_data["B02"].rio.height)
                
                logger.info(f"{scene_id}: Resampling bands...")
                resampled_20m = parallel_resample(bands_20m_data.values(), target_width, target_height, band_level_jobs)
                resampled_20m_dict = dict(zip(bands_20m_data.keys(), resampled_20m))
                
                resampled_60m = parallel_resample(bands_60m_data.values(), target_width, target_height, band_level_jobs)
                resampled_60m_dict = dict(zip(bands_60m_data.keys(), resampled_60m))
                
                SCL = resample_raster(SCL, target_width, target_height, resampling_method=Resampling.nearest)

                # Merge bands
                ds = xr.merge([
                    *bands_10m_data.values(),
                    *resampled_20m_dict.values(),
                    *resampled_60m_dict.values(),
                    SCL
                ])

        elif args.S2_PROCESSING_LEVEL == "l1c":
            bands_10m = ["B02", "B03", "B04", "B08"]
            bands_20m = ["B05", "B06", "B07", "B8A", "B11", "B12"]
            bands_60m = ["B01", "B09", "B10"]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with parallel_backend('threading', n_jobs=band_level_jobs):
                    logger.info(f"{scene_id}: Processing L1C bands...")
                    bands_10m_results = Parallel(verbose=1)(
                        delayed(get_s2l1c_band_data_xarray)(scene_id, band)
                        for band in bands_10m
                    )
                    bands_10m_data = dict(zip(bands_10m, bands_10m_results))

                    bands_20m_results = Parallel(verbose=1)(
                        delayed(get_s2l1c_band_data_xarray)(scene_id, band)
                        for band in bands_20m
                    )
                    bands_20m_data = dict(zip(bands_20m, bands_20m_results))

                    bands_60m_results = Parallel(verbose=1)(
                        delayed(get_s2l1c_band_data_xarray)(scene_id, band)
                        for band in bands_60m
                    )
                    bands_60m_data = dict(zip(bands_60m, bands_60m_results))

                # Get target dimensions and resample
                target_width = int(bands_10m_data["B02"].rio.width)
                target_height = int(bands_10m_data["B02"].rio.height)
                
                logger.info(f"{scene_id}: Resampling bands...")
                resampled_20m = parallel_resample(bands_20m_data.values(), target_width, target_height, band_level_jobs)
                resampled_60m = parallel_resample(bands_60m_data.values(), target_width, target_height, band_level_jobs)

                # Merge bands
                ds = xr.merge([
                    *bands_10m_data.values(),
                    *resampled_20m,
                    *resampled_60m
                ])

        else:
            raise ValueError("Invalid processing level")

        # Assign time dimension
        ds = ds.assign_coords(time=date)
        
        # Reproject to WGS84
        logger.info(f"{scene_id}: Reprojecting to WGS84...")
        kwargs = {"nodata": np.nan}
        ds = ds.rio.reproject(wgs84, **kwargs)
        cube_bbox = ds.rio.bounds()
        logger.info(f"Cube has bbox:{cube_bbox}")
        
        # Generate chips and metadata
        logger.info(f"{scene_id}: Generating chips...")
        cube_meta_df = generate_chips_from_xarray(
            ds=ds,
            cube_size=args.CHIP_SIZE,
            aoi_name=args.AOI_NAME,
            s2_scene_id=scene_id,
            local_out_path_chips=f"{output_path}chips/",
            local_out_path_imgs=f"{output_path}imgs/",
            s3_output_path_chips=args.S3_DESTINATION_PATH_CHIPS,
            s3_output_path_imgs=args.S3_DESTINATION_PATH_IMGS
        )
        
        # Convert bbox to polygon and save metadata
        cube_meta_df["bbox"] = cube_meta_df["bbox"].apply(bbox_to_polygon)
        cube_meta_gdf = gpd.GeoDataFrame(cube_meta_df, geometry='bbox', crs=wgs84)
        cube_meta_gdf.to_parquet(f"{output_path}meta/{scene_id}_chip_meta.parquet")
        
        elapsed_time = time.time() - start
        logger.info(f"Completed processing scene {scene_id} in {elapsed_time:.2f}s")
        
        monitor_resources()  # Monitor at end
        gc.collect()
        
        return {
            "status": "success",
            "scene_id": scene_id,
            "time": elapsed_time,
            "bbox": cube_bbox
        }
        
    except Exception as e:
        logger.error(f"Error processing scene {scene_id}: {str(e)}")
        monitor_resources()  # Monitor on failure
        gc.collect()
        raise

if __name__ == "__main__":
    session = boto3.Session() 
    logger = get_logger(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--S2_PROCESSING_LEVEL", type=str, default="l2a")
    parser.add_argument("--CHIP_SIZE", type=int, default=512)
    parser.add_argument("--AOI_NAME", type=str, default="tbd")
    parser.add_argument("--S3_DESTINATION_PATH_CHIPS", type=str, default="tbd")
    parser.add_argument("--S3_DESTINATION_PATH_IMGS", type=str, default="tbd")
    parser.add_argument("--S3_DESTINATION_PATH_META", type=str, default="tbd")
    wgs84 = "EPSG:4326"
    web_mercator = "EPSG:3857"

    args, _ = parser.parse_known_args()
    logger.info("Received arguments {}".format(args))
    
    logger.info("Starting processing")
    
    output_path = "/opt/ml/processing/output/"
    subprocess.check_call(["sudo","chown","-R","sagemaker-user", output_path])
    
    s2_data_path = '/opt/ml/processing/input/sentinel2_meta/'
    s2_items = []
    for current_path, sub_dirs, files in os.walk(s2_data_path):
        for file in files:
            if file.endswith(".json"):
                full_file_path = os.path.join(s2_data_path, current_path, file)
                with open(full_file_path, 'r') as f:
                    s2_items.append(json.load(f))
    
    item_count_total = len(s2_items) 
    logger.info("Received {} scenes to process".format(item_count_total))

    # Calculate optimal parallel jobs
    scene_level_jobs, band_level_jobs, chip_level_jobs = calculate_parallel_jobs(len(s2_items))
    logger.info(f"Using {scene_level_jobs} parallel scenes, {band_level_jobs} CPUs per band processing, "
                f"and {chip_level_jobs} CPUs per chip generation")
   
    # Create output directories
    os.makedirs(f"{output_path}meta/", exist_ok=True)
    os.makedirs(f"{output_path}chips/", exist_ok=True)
    os.makedirs(f"{output_path}imgs/", exist_ok=True)

    # Process all scenes in parallel
    logger.info("Starting parallel processing of all scenes...")
    with parallel_backend('threading', n_jobs=scene_level_jobs):
        results = Parallel(verbose=1)(
            delayed(process_single_scene)(item, args, output_path, band_level_jobs)
            for item in s2_items
        )

    # Process results
    successful_scenes = [r for r in results if r and r.get("status") == "success"]
    failed_scenes = [r for r in results if r and r.get("status") != "success"]
    
    # Log detailed results
    logger.info(f"Processing completed. Successfully processed {len(successful_scenes)}/{len(s2_items)} scenes.")
    if successful_scenes:
        avg_time = sum(r["time"] for r in successful_scenes) / len(successful_scenes)
        logger.info(f"Average processing time per successful scene: {avg_time:.2f}s")
    if failed_scenes:
        logger.info(f"Failed scenes: {[r['scene_id'] for r in failed_scenes]}")

    logger.info("Processing completed")