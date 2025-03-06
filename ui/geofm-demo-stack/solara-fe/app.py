import os
import leafmap
import solara
import ipywidgets as widgets
import pandas as pd
import geopandas as gpd
import tempfile
import fsspec
import xarray as xr
import math
from shapely.geometry import Point

# Set defaults
scenario = "Tracking Deforestation - Brazil"
base_map_list = ["OpenTopoMap", "OpenStreetMap", "OpenStreetMap.HOT", "ESA WorldCover 2021"]
spectral_index_list = ["TCI", "NDVI", "EVI", "NBR"]

# Load data
aoi_name = "brazil-deforestation"
bucket_name = "gfm-demo-bucket"
s2_tile_id = "20LPP"
path = f"s3://{bucket_name}/chip-change-analysis/output/{aoi_name}/{s2_tile_id}/"

def load_meta_data(path, limit=1000):
    meta = gpd.read_parquet(path, columns=['x_dim', 'y_dim', 'aoi_name', 'chip_id', 'origin_tile', 'file_name', 'date', 'year', 'month',
                                           'bbox', 'crs', 'chip_size', 'cloud_cover_perc', 'missing_data_perc', 
                                           's3_location_netcdf', 's3_location_cls_emb', 's3_location_patch_emb'])
    
    # Limit the number of records to the specified limit
    meta = meta.head(limit)

    chip_grid = gpd.GeoDataFrame(geometry=meta["bbox"].unique())
    chip_grid.columns=['bbox']
    chip_grid.set_geometry('bbox', inplace=True)
    chip_grid.set_crs('epsg:4326', inplace=True)
    bbox_df = meta.groupby('bbox').first().reset_index()
    chip_grid = chip_grid.merge(bbox_df[['bbox', 'chip_id', 'aoi_name']], on='bbox', how='left')
    return meta, chip_grid

def load_aoi_gdf(path="../data/brazil_mato_grosso.geojson"):
    aoi_gdf = gpd.read_file(path)
    aoi_gdf.crs='epsg:4326'
    return aoi_gdf

def load_xarray(path):
    with fsspec.open(path, mode="rb") as f:
        scene = xr.open_dataset(f, decode_coords="all")
    scene['TCI'] = 1.2 * (scene.B05 - scene.B03) - 1.5 * (scene.B04 - scene.B03) * math.sqrt(scene.B05 / scene.B04)
    scene['NDVI'] = (scene.B08 - scene.B04) / (scene.B08 + scene.B04)
    scene['NBR'] = (scene.B08 - scene.B12) / (scene.B08 + scene.B12)
    scene['EVI'] = 2.5 * (scene.B08 - scene.B04) / ((scene.B08 + 6.0 * scene.B04 - 7.5 * scene.B02) + 1.0)
    return scene

def get_netcdf_paths(meta, year, month):
    month_year_meta = meta[(meta.year == int(year)) & (meta.month == int(month))]
    date = month_year_meta.date.value_counts().index[0]
    netcdf_list = month_year_meta[month_year_meta.date == date].s3_location_netcdf.values
    return netcdf_list

def add_widgets(m):
    style = {"description_width": "initial"}
    
    year_widget = widgets.Dropdown(options=list(range(2018, 2025)), description="Year:", style=style)
    month_widget = widgets.Dropdown(options=list(range(1, 13)), description="Month:", style=style)
    
    def update_map(*args):
        netcdf_list = get_netcdf_paths(meta, year_widget.value, month_widget.value)
        if netcdf_list:
            scene = load_xarray(netcdf_list[0])
            # Add code to update the map with the selected indices and layers
    
    update_button = widgets.Button(description="Add Satellite Layer")
    update_button.on_click(update_map)

    box = widgets.VBox([year_widget, month_widget, update_button])
    m.add_widget(box, position="topright")

    def handle_click(**kwargs):
        if kwargs.get("type") == "click":
            latlon = kwargs.get("coordinates")
            geometry = Point(latlon[::-1])
            selected = m.chip_grid[m.chip_grid.intersects(geometry)]
            setattr(m, "zoom_to_layer", False)
            print(f"click at {latlon} geometry: {geometry} chip: {selected}")
            # if len(selected) > 0:
            #     catalog_ids = selected["catalog_id"].values.tolist()

            #     if len(catalog_ids) > 1:
            #         image.options = catalog_ids
            #     image.value = catalog_ids[0]
            # else:
            #     image.value = None

    m.on_interaction(handle_click)


aoi_gdf = load_aoi_gdf()
meta, chip_grid = load_meta_data(path)

zoom_default = 9
center_default = aoi_gdf.geometry.centroid[0].coords[0][::-1]

class CustomMap(leafmap.Map):
    def __init__(self, **kwargs):
        kwargs["toolbar_control"] = False
        super().__init__(**kwargs)
        basemap = {
            "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Satellite",
        }
        self.add_tile_layer(**basemap, shown=False)
        self.add_layer_manager(opened=False)
        add_widgets(self)

        # self.add_gdf(aoi_gdf, layer_name=scenario)
        geojson_data_chip_grid = "chip_grid.geojson"
        self.add_geojson(geojson_data_chip_grid, layer_name="chip_grid", info_mode="on_hover")
        setattr(self, "chip_grid", gpd.read_file(geojson_data_chip_grid))

@solara.component
def Page():
    zoom = solara.reactive(zoom_default)
    center = solara.reactive(center_default)

    with solara.Column(style={"min-width": "500px"}):
        solara.Title("Explore Area of Observation")
        solara.Markdown(f"Explore the selected scenario (`{scenario}`) by retrieving raw satellite imagery.")
        
        m = CustomMap.element(  # type: ignore
            zoom=zoom.value,
            on_zoom=zoom.set,
            center=center.value,
            on_center=center.set,
            scroll_wheel_zoom=True,
            toolbar_ctrl=False,
            data_ctrl=False,
            height="780px",
        )
        
        # Define display options directly in the Page component
        display_aoi_widget = widgets.Checkbox(value=False, description="Display AOI")
        display_chip_grid_widget = widgets.Checkbox(value=False, description="Display Chip Grid")

        if display_aoi_widget.value:
            m.add_gdf(aoi_gdf, layer_name=scenario)

        if display_chip_grid_widget.value:
            geojson_data_chip_grid = chip_grid.__geo_interface__
            m.add_geojson(geojson_data_chip_grid)

        solara.Text(f"Center: {center.value}")
        solara.Text(f"Zoom: {zoom.value}")
