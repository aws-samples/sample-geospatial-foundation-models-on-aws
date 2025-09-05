import os
import sys
import json
import leafmap
import solara
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import matplotlib.dates as mdates
import datetime
import ipyleaflet
from time import perf_counter
import boto3
import logging
import s3fs
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')
import plotly.express as px
import plotly.graph_objects as go

# Initialize global state
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
DATA_DIR = Path(__file__).parent.parent / "data"
HERE = Path(__file__).parent.parent

from utils import load_config, image_from_s3
from change_detection_utils import detect_outliers_by_year, fit_ols_regression, fit_harmonic_regression
from imagery_utils import IndexType, get_layer_url

# Setup Solara theme
app_style = (HERE / "style.css").read_text()
solara.lab.theme.dark = False
solara.lab.theme.themes.light.primary = "#000000"

class Config:
    def __init__(self, config_data):
        for key, value in config_data["config"].items():
            setattr(self, key, value)


# Load configuration
demo_config = load_config()
config = Config(demo_config)

# Initialize classes
class MapState:
    def __init__(self):
        self.selected_chip = solara.reactive(None)
        self.change_points = solara.reactive(None)
        self.y_axis = solara.reactive("cosine_sim")
        self.outliers_yn = solara.reactive(False)
        self.fit_curve_yn = solara.reactive(False)
        self.fit_trend_yn = solara.reactive(False)
        self.add_change_layer_yn = solara.reactive(False)
        self.click_data = solara.reactive(None)
        self.selected_chip_layer = solara.reactive(None)
        # SplitMap contorls
        self.selected_band = solara.reactive("True Color")
        self.left_year = solara.reactive(sorted(config.imagery_metadata.keys())[0])
        self.right_year = solara.reactive(sorted(config.imagery_metadata.keys())[-1])

map_state = MapState()


value_property_cat = 'change_intensity'
colormap_cat = {'High': '#b63679', 
                'Medium': '#fcfdbf',
                'Low': '#000004', }

def style_function_cat(feature):
    try:
        category = feature['properties'][value_property_cat]
        return {
            'fillColor': colormap_cat.get(category, '#GRAY'),  # Default to gray if category not found
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.3
        }
    except:
        return None

class MapLayerManager:
    def __init__(self, config, aoi_bounds):
        self.config = config
        self.aoi_bounds = aoi_bounds

    def get_change_intensity_layer(self):
        try:
            response = s3.get_object(Bucket=config.bucket_name, Key=config.chip_grid_change_intensity_geojson)
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)

            return ipyleaflet.GeoJSON.element(
                data=data,
                name='ChangeIntensity',
                style_callback=style_function_cat,
                hover_style={'fillOpacity': 0.9}
            )
        except Exception as e:
            logger.error(f"ERROR loading change_intensity_layer: {e}")            
            return None


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
            url=f"{os.environ.get('TILES_BACKEND_URL')}/{{z}}/{{x}}/{{y}}.png?url={self.config.sentinel2_cog_source_url}"
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

class DataLoader:
    @staticmethod
    def load_meta_data(metadata_file_name, similarity_search_date=""):
        if similarity_search_date != "":
            target_date = pd.Timestamp(similarity_search_date)
            meta = gpd.read_parquet(f"{DATA_DIR}/{metadata_file_name}", filters=[('date', '==', target_date)])
        else:
            meta = gpd.read_parquet(f"{DATA_DIR}/{metadata_file_name}")
        return meta

    @staticmethod
    def load_meta_data_from_s3(bucket_name, metadata_path, metadata_file_name, similarity_search_date=""):
        
        path = f"s3://{bucket_name}/{metadata_path}"
        s3 = s3fs.S3FileSystem()
        
        if similarity_search_date != "":
            target_date = pd.Timestamp(similarity_search_date)
            meta = gpd.read_parquet(f"{path}{metadata_file_name}", filters=[('date', '==', target_date)], filesystem=s3)            
            # meta = gpd.read_parquet(f"{DATA_DIR}/{metadata_file_name}", filters=[('date', '==', target_date)])
        else:
            # meta = gpd.read_parquet(f"{DATA_DIR}/{metadata_file_name}")
            meta = gpd.read_parquet(f"{path}{metadata_file_name}", filesystem=s3)
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
        # using s3:// is giving random errors on docker
        path = f"/vsis3/{bucket_name}/{object_key}"
        aoi_gdf = gpd.read_file(path)
        aoi_gdf.crs = 'epsg:4326'
        bounds = aoi_gdf.total_bounds
        aoi_bounds = ((bounds[1], bounds[0]), (bounds[3], bounds[2]))
        return aoi_gdf, aoi_bounds
    
# Load data
aoi_gdf, aoi_bounds = DataLoader.load_aoi_gdf_from_s3(config.bucket_name, config.aoi_geojson_s3_key)
zoom_default = 10
zoom_default_chip = 14
avg_days_yr = 365
center_default = aoi_gdf.geometry.centroid[0].coords[0][::-1]
start = perf_counter()
meta = DataLoader.load_meta_data_from_s3(
    config.bucket_name,
    config.metadata_path,
    config.metadata_file_name
)

end = perf_counter()
logger.debug(f"load_meta_data Execution time: {end - start:.6f} seconds")


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

                map_state.selected_chip_layer.value = gpd.GeoDataFrame(
                    data=meta[meta["chip_id"]==map_state.selected_chip.value].groupby('bbox').first().reset_index(),
                    geometry="bbox",
                    crs="EPSG:4326"
                )

        self.on_interaction(handle_click)

# Initialize base_layers, base_controls for MapComponent
layer_manager = MapLayerManager(config, aoi_bounds)
base_layers = layer_manager.create_base_layers()
base_controls = layer_manager.create_base_controls()

@solara.component
def MapComponent(base_layers):

    def update_layers(base_layers, base_controls):
        layers = base_layers
        controls = base_controls

        if map_state.add_change_layer_yn.value is True:
            
            change_layer = layer_manager.get_change_intensity_layer()
            if change_layer != None:
                layers = layers + (change_layer,)
            else:
                logger.error("Error loading change intensity layer")
            
            legend_dict = {
                    'High': '#b63679', 
                    'Medium': '#fcfdbf',
                    'Low': '#000004', 
                }
                
            legend = ipyleaflet.LegendControl(
                legend_dict,
                title="Change Intensity",
                position="bottomright"
            )

            controls = controls + (legend,)

        if map_state.selected_chip_layer.value is not None:
            selection_layer = ipyleaflet.GeoData.element(
                geo_dataframe = map_state.selected_chip_layer.value[["chip_id", "bbox"]],
                style={'color': 'yellow', 'fillOpacity': 0.2,'weight': 0.75},
                name='SelectedChip'
            )
            layers = layers + (selection_layer,)
        
        return layers, controls

    zoom = solara.use_reactive(zoom_default)
    center = solara.use_reactive(center_default)

    # Memorize the layers calculation
    layers, controls = solara.use_memo(
        lambda: update_layers(base_layers, base_controls),
        [map_state.selected_chip_layer.value, map_state.add_change_layer_yn.value]
    )
    logger.debug("Layers:", layers)
    return CustomMap.element(
                    zoom=zoom.value,
                    on_zoom=zoom.set,
                    center=center.value,
                    on_center=center.set,
                    scroll_wheel_zoom=True,
                    toolbar_ctrl=False,
                    data_ctrl=False,
                    height="600px",
                    layers=layers,
                    controls=controls,)


@solara.component
def ChipImageOnClick():
    with solara.Card():
        if map_state.selected_chip.value and map_state.click_data.value:
            date=pd.to_datetime(map_state.click_data.value["points"]["xs"][-1]) #.date
            try:
                s3_png_path = meta[(meta['chip_id']==map_state.selected_chip.value) &
                                   (meta['date']==date)][config.png_thumbnail_key].to_list()[-1]
                
                solara.Image(image_from_s3(s3_png_path), width="256px")
                solara.Markdown(f"**Date**: `{date.date()}`")
            except:
                logger.error(f"error loading PNG")
                solara.Markdown("No image available ")
        else:
            solara.Markdown("No data point selected")

@solara.component
def ChangeLikelihoodPane(thresholds=config.change_threshold_linear_reg):

    #TODO: APPLY THRESHOLD TO TREND OF RESIDUALS VS. HARMONIC REGRESSION

    def get_color(value: float):
        if value <thresholds[0]:
            return "#A7F1A8"  # Light green
        elif value <thresholds[1]:
            return "#FFFFC5"  # Light yellow
        else:
            return "#FF6C70"  # Light red

    if map_state.selected_chip.value is not None:
        #value = meta[meta['chip_id']==selected_chip.value]['cosine_sim'].mean()
        df = meta[meta["chip_id"]==map_state.selected_chip.value]
        logger.debug(f"df: {df}")

        if len(df) > 5: #TODO fix this number
            value = fit_ols_regression(df, y="patch_emb_pca_1")
            
            color = get_color(value)
            
            if value < thresholds[0]:
                text = "Low"
            elif value < thresholds[1]:
                text = "Medium"
            else:
                text = "High"
            
            card_style = {
                #"height": "40px",
                #"width":"175px",
                "background": color,
                "border-radius": "8px",
                #"padding": "0px",
                "text-align": "center",
                #"margin": "0px 0"
            }
        else:
            logger.warning(f"Not enough dataset to Fit only {len(df)} rows")
            text = "N/A"
            card_style={"border-radius": "15px","background": "#f5f5f5"}
            color="#f5f5f5"
    else:
        card_style={"border-radius": "15px","background": "#f5f5f5"}
        color="#f5f5f5"

    with solara.Card(style=card_style,margin=0):
        with solara.Column(align="center",style={"background":color}):
            if map_state.selected_chip.value is not None:
                solara.Markdown(f"**Change Severity**: `{text}`", style={"font-size": "14px", "align": "center", "background": color })
            else:
                solara.Text(f"No chip selected", style={"font-size": "14px", "text-align": "center", "background": color})


@solara.component
def TimeSeriesPane():
   
    with solara.Card(title="Image Embeddings Timeseries Plot", 
                     subtitle=f"Analyze embeddings over time to identify breaks and discontinuities that signify change",
                     style={"border-radius": "15px","background": "#f5f5f5"}): 
        
        with solara.Row(style={"background": "#f5f5f5"}):
            solara.Select(label="Y-axis", values=['cosine_sim','patch_emb_pca_1', 'patch_emb_pca_2','patch_emb_tsne_1', 'patch_emb_tsne_2'],value=map_state.y_axis)

        with solara.Row(justify="center",style={"background": "#f5f5f5"}):
        
            with solara.Columns([3,1]):
                
                with solara.Row(style={"background": "#f5f5f5"}):
                    solara.Switch(label="Exclude Outliers", value=map_state.outliers_yn)
                    solara.Switch(label="Fit Trend Line", value=map_state.fit_trend_yn)
                    solara.Switch(label="Fit Harmonic Regression", value=map_state.fit_curve_yn)  #TODO: Potentially make baseline variable
                
                ChangeLikelihoodPane()

        with solara.Row():

            #compute outliers
            if map_state.selected_chip.value is not None:
                outliers = detect_outliers_by_year(meta[meta["chip_id"]==map_state.selected_chip.value],map_state.y_axis.value,"year",2)
            else:
                pass
                
            #themes: ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
            if map_state.selected_chip.value is not None:
                if map_state.outliers_yn.value:
                    df =  meta[(meta['chip_id']==map_state.selected_chip.value) & (~meta["date"].isin(outliers))].sort_values(by='date', ascending=True)
                else:
                    df = meta[meta['chip_id']==map_state.selected_chip.value].sort_values(by='date', ascending=True)
                
                if map_state.fit_trend_yn.value:
                    fig = px.scatter(data_frame=df, x="date", y=map_state.y_axis.value,template="ggplot2",trendline="ols") #TODO: serialize the date before fitting OLS
                else:
                    fig = px.scatter(data_frame=df, x="date", y=map_state.y_axis.value,template="ggplot2")
                    
                if map_state.fit_curve_yn.value:
                    if len(df) > 2: #TODO fix this number
                        baseline_start_date = datetime.datetime.strptime(config.baseline_start_date, "%Y-%m-%d")
                        monitoring_start_date = datetime.datetime.strptime(config.monitoring_start_date, "%Y-%m-%d")
                        fitted_model, poly_transform, df = fit_harmonic_regression(data=df, outliers=outliers, date_col="date", y_col=map_state.y_axis.value, 
                                                baseline_start_date=baseline_start_date, monitoring_start_date=monitoring_start_date, 
                                                deg=2,reg=0.001)
                        
                        t_plot = np.arange(df["date_numerical"].min(), df["date_numerical"].max(),1)
                        t_plot_datetime = mdates.num2date(t_plot)
                        w = 2 * np.pi / avg_days_yr
                        X_plot = poly_transform.transform(np.column_stack((np.sin(w*t_plot), np.cos(w*t_plot))))
                        y_plot=fitted_model.predict(X_plot)
                        # Add the line trace to the existing figure
                        fig.add_trace(
                            go.Scatter(
                                x=t_plot_datetime, 
                                y=y_plot, 
                                mode='lines', 
                                name='Line Plot', 
                                legend=None
                            )
                        )
                    else:
                        logger.warning(f"Not enough dataset to Fit only {len(df)} rows")

                #add a title to the figure
                fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
                fig.update_layout({
                                    "plot_bgcolor": "rgba(0, 0, 0, 0)",
                                    "paper_bgcolor": "rgba(0, 0, 0, 0)",
                                    })
                
                fig.update_layout(title=f"Chip ID: {map_state.selected_chip.value}",showlegend=False)
                fig.update_layout(hovermode="closest")
                #render the figure

                with solara.Columns([3, 1],style={"background": "#f5f5f5"}):
                    solara.FigurePlotly(fig, on_click=map_state.click_data.set)
                    ChipImageOnClick()

class SplitMap(leafmap.Map):
    def __init__(self, **kwargs):
        kwargs["toolbar_control"] = False
        kwargs["zoom_to_layer"] = False
        kwargs["fit_bounds"] = False
        kwargs["auto_pan"] = False
        super().__init__(**kwargs)

# splitmap_base_layers for SplitMapComponent
osm_layer = ipyleaflet.TileLayer.element(
    max_zoom=19,
    name='OpenStreetMap',
    url='https://tile.openstreetmap.org/{z}/{x}/{y}.png'
)
splitmap_base_layers = (osm_layer, )

@solara.component
def SplitMapComponent(chip):
    
    def update_splitmap_layers(splitmap_base_layers):
        try:
            logger.debug("Updating splitmap layers")

            # Base controls
            zoom = ipyleaflet.ZoomControl(position="topleft")
            fullscreen_control = ipyleaflet.FullScreenControl()
            splitmap_base_controls = (zoom, fullscreen_control)

            if map_state.selected_chip_layer.value is not None:
                # FOR LOCAL TESTING
                left_url = "https://titiler.xyz/cog/tiles/WebMercatorQuad/{z}/{x}/{y}@1x?url=https%3A%2F%2Fsentinel-cogs.s3.us-west-2.amazonaws.com%2Fsentinel-s2-l2a-cogs%2F20%2FL%2FPP%2F2018%2F7%2FS2B_20LPP_20180702_0_L2A%2FTCI.tif&bidx=1&bidx=2&bidx=3&rescale=18.0%2C109.0&rescale=32.0%2C91.0&rescale=19.0%2C60.0"
                right_url = 'https://titiler.xyz/cog/tiles/WebMercatorQuad/{z}/{x}/{y}@1x?url=https%3A%2F%2Fsentinel-cogs.s3.us-west-2.amazonaws.com%2Fsentinel-s2-l2a-cogs%2F20%2FL%2FPP%2F2024%2F7%2FS2B_20LPP_20240730_0_L2A%2FTCI.tif&bidx=1&bidx=2&bidx=3&rescale=22.0%2C169.0&rescale=36.0%2C117.0&rescale=22.0%2C85.0'

                # Get URLs using the utility function with demo_config
                left_url = get_layer_url(
                    config,
                    map_state.left_year.value, 
                    map_state.selected_band.value
                )
                right_url = get_layer_url(
                    config,
                    map_state.right_year.value, 
                    map_state.selected_band.value
                )

                colormap_legend_dict = {
                    "0.00": "#d73027",
                    "0.10": "#f46d43",
                    "0.20": "#fdae61",
                    "0.30": "#fee08b",
                    "0.40": "#ffffbf",
                    "0.50": "#d9ef8b",
                    "0.60": "#a6d96a",
                    "0.70": "#66bd63",
                    "0.80": "#1a9850",
                    "0.90": "#006837"
                }

                # from above dict create a dict like this {"VALUE": "HEX"}
                legend = ipyleaflet.LegendControl(
                    colormap_legend_dict,
                    title="Legend",
                    position="bottomright"
                )

                pre_layer = ipyleaflet.TileLayer(bounds = aoi_bounds, url = left_url)
                post_layer = ipyleaflet.TileLayer(bounds = aoi_bounds, url = right_url)
                splitmap_control = ipyleaflet.SplitMapControl(
                        left_layer=pre_layer, 
                        right_layer=post_layer,
                        left_label=str(map_state.left_year.value),
                        right_label=str(map_state.right_year.value)
                )
                            
                # Add Legend for all except True Color
                controls = (splitmap_control,)
                if map_state.selected_band.value != IndexType.TRUE_COLOR.value:
                    controls = (splitmap_control, legend)
                
                return splitmap_base_layers, splitmap_base_controls + controls
                
            return splitmap_base_layers, splitmap_base_controls
        except Exception as e:
            logger.error(f"Error updating splitmap layers: {str(e)}")
            return None, None          

    zoom_split_map = solara.use_reactive(zoom_default_chip)
    center_split_map = solara.use_reactive(center_default)

    
    #TODO: change years, make dynamic
    #years = [2018,2019,2020,2021,2022,2023,2024]
    years = list(config.imagery_metadata.keys())
    #print(years)
    # bands = ["True Color", "Vegetation", "Burn Ratio", "Enhanced Vegetation"]
    bands = IndexType.get_all_display_names()

    # Memorize the layers calculation
    splitmap_layers, splitmap_controls = solara.use_memo(
        lambda: update_splitmap_layers(splitmap_base_layers),
        [map_state.selected_chip_layer.value, map_state.left_year.value, map_state.selected_band.value, map_state.right_year.value]
    )
    
    with solara.Card(title="Deep-Dive Visual Analysis", subtitle=f"Conduct an in-depth visual inspection of the selected chip by comparing the first and last observations",style={"border-radius": "15px","background": "#f5f5f5"}):
        if chip.value is not None:
        
            gdf=meta[meta["chip_id"]==chip.value].head(1).reset_index(drop=True)
            if len(gdf) > 0:
                centroid = gdf.geometry.centroid[0].coords[0][::-1]
                center_split_map.value = centroid

                with solara.Columns([3,1], style={"z-index": "2000 !important"}):
                    with solara.Row(style={"background": "#f5f5f5"}):
                        solara.Select(label="Left Map Year", value=map_state.left_year, values=years)
                        solara.Select(label="Index", value=map_state.selected_band, values=bands)
                        solara.Select(label="Right Map Year", value=map_state.right_year, values=years)
            
                SplitMap.element(  # type: ignore
                                zoom=zoom_split_map.value,
                                # on_zoom=zoom_split_map.set,
                                center=center_split_map.value,
                                # on_center=center_split_map.set,
                                scroll_wheel_zoom=True,
                                toolbar_ctrl=False,
                                data_ctrl=False,
                                height="440px",
                                # base_layers=splitmap_layers,
                                controls=splitmap_controls
                            )
            else:
                solara.Markdown("No data found for the selected chip, Select another chip")
        else:
            solara.Markdown("Select a chip to view a pre and post split map")


@solara.component
def Page():
    solara.Style(app_style)
    solara.lab.theme.dark = False
    solara.lab.theme.themes.light.primary = "#000000" 
    solara.Title("Ecosystem Change Detection")
    zoom = solara.use_reactive(zoom_default)
    center = solara.use_reactive(center_default)
    
    with solara.v.Html(tag="a", attributes={"id": f"main-content"}):  
        with solara.Card(title="Detect Change over Time",subtitle="Select a reference chip from the map below to run timeseries analyses and detect change",
                        style={"height": "1500px"}):
            with solara.GridFixed(columns=1):
                with solara.Columns([4,6]):
                     # Left column with map
                    with solara.Column(gap="0px"):
                            solara.Checkbox(label="Display Change Intensity", value=map_state.add_change_layer_yn)
                            MapComponent(base_layers)
                    # column For reference image 
                    with solara.Column(gap="5px"):
                        TimeSeriesPane()
            with solara.GridFixed(columns=1):
                with solara.Column(gap="5px",style={"flex": "6"}):
                    SplitMapComponent(map_state.selected_chip)

