import os
import sys
import json
import leafmap
import solara
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from joblib import Parallel, delayed
from pathlib import Path
import ipyleaflet
from time import perf_counter
import boto3
import s3fs

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

s3 = boto3.client('s3')

# Initialize global state
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
DATA_DIR = Path(__file__).parent.parent / "data"
HERE = Path(__file__).parent.parent

from utils import load_config, image_from_s3
from vectordb_helper import VectorDBHelper

# Setup Solara theme
app_style = (HERE / "style.css").read_text()
solara.lab.theme.dark = False
solara.lab.theme.themes.light.primary = "#000000"

class Config:
    def __init__(self, config_data):
        for key, value in config_data["config"].items():
            setattr(self, key, value)

class MapState:
    def __init__(self):
        self.selected_chip = solara.reactive(None)
        self.similar_chips_dict = solara.reactive({}) 
        self.num_results = solara.reactive(5)
        self.min_score = solara.reactive(0.0)
        self.show_on_map = solara.reactive(False)
        self.similar_chips_layer = solara.reactive(None)
        self.selected_chip_layer = solara.reactive(None)


# Load configuration and initialize classes
demo_config = load_config()
vectordb_helper = VectorDBHelper(demo_config)
config = Config(demo_config)
map_state = MapState()

class MapLayerManager:
    def __init__(self, config, aoi_bounds):
        self.config = config
        self.aoi_bounds = aoi_bounds
        
    def create_base_layers(self):
        open_topo = ipyleaflet.TileLayer.element(
            max_zoom=24,
            name='OpenTopoMap',
            url='https://a.tile.opentopomap.org/{z}/{x}/{y}.png'
        )
        
        google_maps = ipyleaflet.TileLayer.element(
            max_zoom=24,
            name='Mosaic Satellite',
            url='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        )
        
        satellite = ipyleaflet.TileLayer.element(
            max_zoom=24,
            name='Sentinel-2',
            bounds=self.aoi_bounds,
            url=f"{self.config.tiles_backend_url}/{{z}}/{{x}}/{{y}}.png?url={self.config.sentinel2_cog_source_url}"
        )
        
        chip_grid = self._create_chip_grid()
        return (open_topo, google_maps, chip_grid, satellite)
    
    def _create_chip_grid(self):
                
        # Read the chip grid GeoJSON file from S3
        response = s3.get_object(Bucket=self.config.bucket_name, Key=self.config.chip_grid_geojson_s3_key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)

        return ipyleaflet.GeoJSON.element(
            data=data,
            name="ChipGrid",
            style={'color': 'yellow', 'fillOpacity': 0.0, 'weight': 0.3},
            hover_style={'color': 'white', 'dashArray': '0', 'fillOpacity': 0.2},
            info_mode="on_hover"
        )

    def create_base_controls(self):
        zoom = ipyleaflet.ZoomControl.element(position="topleft")
        fullscreen_control = ipyleaflet.FullScreenControl.element()
        layers_ctrl = ipyleaflet.LayersControl.element(position="topright")
        return (zoom, fullscreen_control, layers_ctrl)

class CustomMap(leafmap.Map):
    def __init__(self, **kwargs):
        kwargs["toolbar_control"] = False
        kwargs["draw_control"] = False
        super().__init__(**kwargs)
        self.add_layer_manager(opened=False)
        self.zoom_to_layer = False
        self._setup_click_handler()

    def _setup_click_handler(self):
        def handle_click(**kwargs):
            if kwargs.get("type") == "click":
                latlon = kwargs.get("coordinates")
                geometry = Point(latlon[::-1])
                selected = meta[meta.intersects(geometry)]

                if len(selected) > 0:
                    map_state.selected_chip.value = selected['chip_id'].unique()[0]
                else:
                    map_state.selected_chip.value = None

                logger.info(f"click at {latlon} geometry: {geometry} chip: {selected}")

                if map_state.selected_chip.value:
                    map_state.selected_chip_layer.value = gpd.GeoDataFrame(
                        data=meta[meta["chip_id"]==map_state.selected_chip.value].groupby('bbox').first().reset_index(),
                        geometry="bbox",
                        crs="EPSG:4326"
                    )
                
                map_state.show_on_map.value = False
                map_state.similar_chips_dict.value = {}
                map_state.similar_chips_layer.value = None

        self.on_interaction(handle_click)

class DataLoader:

    @staticmethod
    def load_meta_data(metadata_file_name, similarity_search_date):

        cols=['date', 'bbox', 'crs', 'cloud_cover_perc', 'missing_data_perc',
        's3_location_netcdf', config.png_thumbnail_key, 'year', 'month',
        'chip_id', 's3_location_cls_emb', 's3_location_patch_emb']
        
        if similarity_search_date != "":
            target_date = pd.Timestamp(similarity_search_date)
            meta = gpd.read_parquet(f"{DATA_DIR}/{metadata_file_name}", filters=[('date', '==', target_date)], columns=cols)
        else:
            meta = gpd.read_parquet(f"{DATA_DIR}/{metadata_file_name}", columns=cols)
        return meta
    
    @staticmethod
    def load_meta_data_from_s3(bucket_name, metadata_path, metadata_file_name, similarity_search_date):
        path = f"s3://{bucket_name}/{metadata_path}"
        s3 = s3fs.S3FileSystem()        
        
        cols=['date', 'bbox', 'crs', 'cloud_cover_perc', 'missing_data_perc',
        's3_location_netcdf', config.png_thumbnail_key, 'year', 'month',
        'chip_id', 's3_location_cls_emb', 's3_location_patch_emb']
        
        if similarity_search_date != "":
            target_date = pd.Timestamp(similarity_search_date)
            meta = gpd.read_parquet(f"{path}{metadata_file_name}", filters=[('date', '==', target_date)], columns=cols, filesystem=s3)
        else:
            meta = gpd.read_parquet(f"{path}{metadata_file_name}", columns=cols, filesystem=s3)
        return meta

    @staticmethod
    def load_aoi_gdf(data_dir, filename):
        path = f"{data_dir}/{filename}"
        aoi_gdf = gpd.read_file(path)
        aoi_gdf.crs = 'epsg:4326'
        bounds = aoi_gdf.total_bounds
        aoi_bounds = ((bounds[1], bounds[0]), (bounds[3], bounds[2]))
        return aoi_gdf, aoi_bounds
    
    @staticmethod
    def load_aoi_gdf_from_s3(bucket_name, object_key):
        s3 = s3fs.S3FileSystem()
        path = f"/vsis3/{bucket_name}/{object_key}"
        aoi_gdf = gpd.read_file(path)
        aoi_gdf.crs = 'epsg:4326'
        bounds = aoi_gdf.total_bounds
        aoi_bounds = ((bounds[1], bounds[0]), (bounds[3], bounds[2]))
        return aoi_gdf, aoi_bounds


# Load data
# aoi_gdf, aoi_bounds = DataLoader.load_aoi_gdf(DATA_DIR, config.aoi_geojson)
aoi_gdf, aoi_bounds = DataLoader.load_aoi_gdf_from_s3(config.bucket_name, config.aoi_geojson_s3_key)
zoom_default = 10
center_default = aoi_gdf.geometry.centroid[0].coords[0][::-1]
start = perf_counter()
meta = DataLoader.load_meta_data_from_s3(
    config.bucket_name,
    config.metadata_path,
    config.metadata_file_name,
    config.similarity_search_date
)
end = perf_counter()
logger.info(f"load_meta_data Execution time: {end - start:.6f} seconds")

# Initialize map components
layer_manager = MapLayerManager(config, aoi_bounds)
base_layers = layer_manager.create_base_layers()
base_controls = layer_manager.create_base_controls()

def update_layers(base_layers):
    
    if map_state.show_on_map.value==False and map_state.selected_chip.value is not None and map_state.selected_chip_layer.value is not None:
        selection_layer = ipyleaflet.GeoData.element(
            geo_dataframe=map_state.selected_chip_layer.value[["chip_id", "bbox"]],
            style={'color': 'yellow', 'fillOpacity': 0.0,'weight': 0.75},
            name='SelectedChip'
        )
        return base_layers + (selection_layer,)
    if map_state.show_on_map.value and map_state.selected_chip.value is not None and map_state.selected_chip_layer.value is not None and map_state.similar_chips_layer.value is not None:
        logger.info("Search result loop!")
        selection_layer = ipyleaflet.GeoData.element(
            geo_dataframe=map_state.selected_chip_layer.value[["chip_id", "bbox"]],
            style={'color': 'yellow', 'fillOpacity': 0.0,'weight': 0.75},
            name='SelectedChip'
        )
        search_result_layer = ipyleaflet.GeoData.element(
            geo_dataframe=map_state.similar_chips_layer.value[["chip_id", "bbox"]],
            style={'color': 'white', 'fillOpacity': 0.0,'weight': 0.75},
            name='SearchResults'
        )
        return base_layers + (search_result_layer,selection_layer,)
    else:
        return base_layers
    
@solara.component
def MapComponent(base_layers):

    zoom = solara.use_reactive(zoom_default)
    center = solara.use_reactive(center_default)

    # Memorize the layers calculation
    layers = solara.use_memo(
        lambda: update_layers(base_layers),
        [map_state.selected_chip_layer.value, map_state.show_on_map.value, map_state.similar_chips_layer.value]
    )

    return CustomMap.element(  # type: ignore
                    zoom=zoom.value,
                    on_zoom=zoom.set,
                    center=center.value,
                    on_center=center.set,
                    scroll_wheel_zoom=True,
                    toolbar_ctrl=False,
                    data_ctrl=False,
                    height="800px",
                    layers=layers,
                    controls=base_controls
                )

@solara.component
def SearchParamSection():

    def run_similarity_search():
        if  map_state.selected_chip.value:
            map_state.similar_chips_dict.value = {}

            # Update the selected_chip reactive state with the chip_id
            logger.info(f"finding similar images for chip_id: {map_state.selected_chip.value}")
            s3_cls_path = meta[meta['chip_id']==map_state.selected_chip.value]['s3_location_cls_emb'].to_list()[-1]
            search_cls_emb = vectordb_helper.get_cls_token_emb(s3_cls_path, map_state.selected_chip.value)['cls_emb']
            date_filter = None if config.similarity_search_date == "" else config.similarity_search_date 

            res = vectordb_helper.find_similar_items(
                    query_emb=search_cls_emb,
                    k=map_state.num_results.value,
                    num_results=map_state.num_results.value,
                    chip_ids = [map_state.selected_chip.value],
                    date_filter = date_filter
                )
            res_dict = [i for i in res]
            #exclude results below similarity score threshold
            res_dict = [i for i in res_dict if i['score'] >= map_state.min_score.value]
            
            # Parallel processing with threads
            images = Parallel(n_jobs=-1, prefer="threads", verbose=1)(
                delayed(image_from_s3)(itm['s3_location_chip_png']) 
                for itm in res_dict
            )
            #add back to res_dict
            for i, itm in enumerate(res_dict):
                itm['image'] = images[i]
            map_state.similar_chips_dict.value = res_dict

            #generate a gdf of the selected chips
            chip_ids = [i["chip_id"] for i in res_dict]
            map_state.similar_chips_layer.value = gpd.GeoDataFrame(data=meta[meta["chip_id"].isin(chip_ids)].groupby('bbox').first().reset_index(),geometry="bbox",crs="EPSG:4326")
        else:
            pass

    #package similarity search in handler function to avoid automatically running it when reactive components are updated
    def search_button_func():
        map_state.show_on_map.value=False #button
        run_similarity_search()
    
    try:
        s3_png_path = meta[(meta['chip_id']==map_state.selected_chip.value)][config.png_thumbnail_key].to_list()[-1]
        bbox = meta[(meta['chip_id']==map_state.selected_chip.value)]['bbox'].to_list()[-1]
        lat_lng = (round(bbox.centroid.y,3), round(bbox.centroid.x,3))
        date_sel = meta[(meta['chip_id']==map_state.selected_chip.value)]['date'].to_list()[-1]
    except:
        s3_png_path = None
    
    with solara.Card(title="Set Vector Search Parameters", 
                     subtitle="Review the reference image and set vector search parameters",
                     style={"border-radius": "15px","background": "#f5f5f5"}): #,style=app_style
        with solara.Columns([1, 2]):
            with solara.Card():
                if map_state.selected_chip.value is not None and s3_png_path is not None:
                    solara.Image(image_from_s3(s3_png_path), width="100%")
                    solara.Markdown(f"Chip ID: `{map_state.selected_chip.value}`")
                    solara.Markdown(f"Coords: `{lat_lng}`")
                    solara.Markdown(f"Date: `{date_sel.date()}`")
                else:
                    solara.Markdown("No chip selected")
            with solara.Card():
                solara.SliderInt(label="Number of similar items", value=map_state.num_results, min=1, max=100)
                solara.Markdown(f"**Value**: {map_state.num_results.value}")
                solara.SliderFloat(label="Similarity score threshold", value=map_state.min_score, min=0.0, max=1.0, step=0.05)
                solara.Markdown(f"**Value**: {map_state.min_score.value}")
                with solara.CardActions():
                    solara.Button("Run search", text=True,outlined=True,on_click=search_button_func)


@solara.component
def SearchResultSection(selected_chip, similar_chips):
    num_similar_chips = len(similar_chips)
    with solara.Card(title="Vector Search Results", subtitle=f"Number of similar items {num_similar_chips}",style={"border-radius": "15px","background": "#f5f5f5"}):
        #with solara.CardActions():
        solara.Switch(label="Show results on map", value=map_state.show_on_map)
        if selected_chip is not None:
            if len(similar_chips) > 0:
                with solara.VBox():
                    with solara.GridFixed(columns=3):
                        for i in similar_chips:
                            with solara.Card(style={"margin": "6px 0"}):
                                solara.Image(
                                    image=i['image'],width="100%"
                                    #style={"max-width": "70%"}
                                )
                                solara.Markdown(f"Similarity: `{round(i['score'],4)}`", style="font-style: bold; margin-top: 10px;")
        else:
            solara.Text("Select a chip on the map to view similar images and render them on the map")

@solara.component
def Page():
    solara.Style(app_style)
    solara.lab.theme.dark = False
    solara.lab.theme.themes.light.primary = "#000000" 
    solara.Title("Geospatial Similarity Search")
    
    with solara.v.Html(tag="a", attributes={"id": f"main-content"}):    
        with solara.Card(title="Run a Geospatial Similarity Search", subtitle="Select a reference chip from the map below to retrieve similar items",
                        style={"height": "3000px", "width": "100%","padding": "6px"}):

            with solara.Row(style={"height": "1000px", "width": "100%"}):
                
                # Left column with map
                with solara.Column(style={"flex": "1"}):
                    MapComponent(base_layers)


                # column For reference image 
                with solara.Column(style={"flex": "1"}):
                    
                    # Search Params
                    SearchParamSection()
                    
                    # Retrieved Results and Plot on Map
                    SearchResultSection(map_state.selected_chip.value, map_state.similar_chips_dict.value)        