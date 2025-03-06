# imagery_utils.py
from enum import Enum
from typing import Optional, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class IndexType(Enum):
    """Supported spectral indices"""
    TRUE_COLOR = "True Color"
    VEGETATION = "Vegetation"
    BURN_RATIO = "Burn Ratio"
    ENHANCED_VEGETATION = "Enhanced Vegetation"

    @property
    def band_name(self) -> str:
        """Get the technical band name for the index"""
        return {
            self.TRUE_COLOR: "rgb",
            self.VEGETATION: "ndvi",
            self.BURN_RATIO: "nbr",
            self.ENHANCED_VEGETATION: "evi"
        }[self]

    @property
    def s2_band_name(self) -> str:
        """Get the technical band name for the index"""
        return {
            self.TRUE_COLOR: "TCI",
            self.VEGETATION: "ndvi", # TODO does not exit on raw s2
            self.BURN_RATIO: "nbr", # TODO does not exit on raw s2
            self.ENHANCED_VEGETATION: "evi" # TODO does not exit on raw s2
        }[self]

    @classmethod
    def get_all_display_names(cls) -> list[str]:
        """Get list of all display names for UI dropdowns"""
        return [index.value for index in cls]

    @classmethod
    def from_display_name(cls, name: str) -> 'IndexType':
        """Convert display name to enum"""
        try:
            return next(t for t in cls if t.value == name)
        except StopIteration:
            raise ValueError(f"Invalid index type: {name}")

class ProcessingLevel(Enum):
    """Sentinel-2 processing levels"""
    L1C = "L1C"
    L2A = "L2A"

class ImageryProvider(Enum):
    """Supported imagery providers"""
    SENTINEL2 = "sentinel2"
    PLANET = "planet"
    MAXAR = "maxar"

    @property
    def folder_structure(self) -> str:
        """Get provider-specific folder structure"""
        return {
            self.SENTINEL2: "s2_cubes",
            self.PLANET: "planet_cubes",
            self.MAXAR: "maxar_cubes"
        }[self]

@dataclass
class ImageMetadata:
    """Container for image metadata"""
    provider: ImageryProvider
    satellite: str
    date: str
    month: int
    tile_id: str
    processing_level: ProcessingLevel = ProcessingLevel.L2A

    def get_image_path(self) -> str:
        """Generate the image path based on provider's naming convention"""
        if self.provider == ImageryProvider.SENTINEL2:
            return f"{self.satellite}_{self.tile_id}_{self.date}_0_{self.processing_level.value}"
        elif self.provider == ImageryProvider.PLANET:
            return f"PSScene_{self.tile_id}_{self.date}"
        elif self.provider == ImageryProvider.MAXAR:
            return f"MaxarScene_{self.tile_id}_{self.date}"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

def get_visualization_params(index_type: IndexType, provider: ImageryProvider, 
                           processing_level: ProcessingLevel = ProcessingLevel.L2A) -> str:
    """Get visualization parameters for titiler"""
    if index_type == IndexType.TRUE_COLOR:
        if provider == ImageryProvider.SENTINEL2:
            rescale = "0,4000" if processing_level == ProcessingLevel.L1C else "0,2000"
            return f"bidx=1&bidx=2&bidx=3&rescale={rescale}"
        elif provider == ImageryProvider.PLANET:
            return "bidx=1&bidx=2&bidx=3&rescale=0,3000"
        else:
            return "bidx=1&bidx=2&bidx=3&rescale=0,2500"
    else:
        return "bidx=1&rescale=0,1&colormap_name=rdylgn"

def get_layer_url(config: 'Config', year: int, 
                 index: Union[str, IndexType]) -> str:
    """
    Generate tile URL for the specified year and index type.
    
    Args:
        config: Config class instance containing demo configuration
        year: Year of imagery
        index: Either IndexType enum or display name string ("True Color", "Vegetation", etc.)
    
    Returns:
        URL string for the tile layer
    """
    try:
        # Handle both string and enum inputs
        index_type = index if isinstance(index, IndexType) else IndexType.from_display_name(index)
        band_name = index_type.band_name
        s2_band_name = index_type.s2_band_name
        
        # Validate inputs and get image metadata
        year_str = str(year)
        if year_str not in config.imagery_metadata:  # Access as dictionary
            raise ValueError(f"No imagery metadata for year {year}")
        
        # Get image metadata for the year
        img_meta_dict = config.imagery_metadata[year_str]  # Access as dictionary
        
        # Create ImageMetadata instance
        img_meta = ImageMetadata(
            provider=ImageryProvider(img_meta_dict['provider']),  # Access dictionary values
            satellite=img_meta_dict['satellite'],
            date=img_meta_dict['date'],
            month=img_meta_dict['month'],
            tile_id=config.s2_tile_id,
            processing_level=ProcessingLevel(img_meta_dict.get('processing_level', 'L2A'))
        )

        def convert_sentinel_id(sentinel_id):
            return f"{sentinel_id[:2]}/{sentinel_id[2]}/{sentinel_id[3:]}"
        
        # Construct base URL
        # "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/11/S/LT/2025/1/S2A_11SLT_20250112_0_L2A/TCI.tif"
        if config.use_remote_s2_cogs:
            geotiff_url = (
                f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
                f"{convert_sentinel_id(config.s2_tile_id)}/{year}/{img_meta.month}/" 
                f"{img_meta.get_image_path()}/{s2_band_name}.tif"
            )
        else:
            geotiff_url = (
                f"{config.geotiff_bucket_url}/geotiff/{config.aoi_name}/"
                f"{img_meta.provider.folder_structure}/{year}/{img_meta.month}/"
                f"{img_meta.get_image_path()}_aoi_max_size_{band_name}.tif"
            )
        
        # Add visualization parameters
        viz_params = get_visualization_params(index_type, img_meta.provider, img_meta.processing_level)

        # no need of viz_params for remote sentinel_cog_url
        if not config.use_remote_s2_cogs:
            geotiff_url = f"{geotiff_url}&{viz_params}"
        
        return f"{config.tiles_backend_url}/{{z}}/{{x}}/{{y}}.png?url={geotiff_url}"
    
    except Exception as e:
        logger.error(f"Error generating layer URL: {str(e)}")
        raise
