import solara
from pathlib import Path
from solara.lab import theme
theme.themes.light.primary = "#000000"
import logging
logger = logging.getLogger(__name__)
import os, sys


# Get environment variable with a default value
telemetry_enabled = os.getenv("SOLARA_TELEMETRY_MIXPANEL_ENABLE", default="false")
logger.debug(f"telemetry_enabled: {telemetry_enabled}")
DATA_DIR = Path(__file__).parent.parent / "data"

# Load style
HERE = Path(__file__).parent.parent
page_style = (HERE / "style.css").read_text()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import load_config

@solara.component
def LinkedButton(path: str, label: str):
    with solara.Link(path) as link:
        solara.Button(label) #,color="green"
    return link

def scroll_to_next_section(): #TODO: not working
    solara.HTML("""
    <script>
        window.scrollBy(0, window.innerHeight);
    </script>
    """)


@solara.component
def LandingPage():
    
    #FOR LOCAL TESTING
    #video_path = f"{DATA_DIR}/amazon_forest_video_4.mp4"
    # video_path = f"{DATA_DIR}/amazon_forest_video_3.mp4"
    # with open(video_path, "rb") as video_file:
    #     video_data = video_file.read()
    #     video_base64 = base64.b64encode(video_data).decode("utf-8")
    # background_video_url = f"data:video/mp4;base64,{video_base64}"

    #FOR DEPLOYMENT
    demo_config = load_config()
    cloudfront_url = demo_config["config"]["cloudfront_url"]
    background_video_url = f"{cloudfront_url}/{demo_config['config']['background_video_file']}"

    with solara.Column(style={"height": "100vh", "width": "100vw", "position": "relative", "overflow": "hidden"}):
        # Embed the video element
        solara.HTML(unsafe_innerHTML=f"""
            <video autoplay loop muted playsinline 
                   style="position: absolute; top: 50%; left: 50%; min-width: 100%; min-height: 100%; transform: translateX(-50%) translateY(-50%);">  
                <source src="{background_video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        """)
        
        # Text overlay
        with solara.Column(align="center",style={
            "position": "absolute",
            "top": "50%",
            "left": "50%",
            "transform": "translate(-50%, -50%)",
            "text-align": "center",
            "color": "white",
            "z-index": "1",
            "background-color": "rgba(0, 0, 0, 0)",
            "padding": "0px",
            "border-radius": "0px",
            "width": "100%",
           
        }):
            solara.Markdown("# Ecosystem Monitoring with <br> Geospatial Foundation Models <br> (GeoFMs)", style={"font-size": "2rem", "margin-bottom": "1rem", "color": "#000000", "text-shadow": "2px 2px 4px rgba(0, 0, 0, 0.5);"})
            solara.Markdown("Discover how state-of-the-art geospatial vision models can help your organization <br> unlock the power of earth observation data for ecosystem monitoring.", style={"font-size": "1.5rem", "margin-bottom": "2rem","text-align": "center", "color": "#000000", "text-shadow": "2px 2px 4px rgba(0, 0, 0, 0.5);"})
            with solara.Column(align="center"):
                with solara.v.Html(tag="a", attributes={"href": f"#main-content"}):
                    solara.Button("Detect Deforestation with GeoFMs", color="primary", style={
                        "font-size": "14px",
                        "padding": "6px 12px",
                        "width": "auto"
                    })

    with solara.Columns([1,8,1]): #main column with content
 
        solara.Text("")
        
        with solara.Column(align="center"):
            with solara.v.Html(tag="a", attributes={"id": f"main-content"}):     
                solara.Markdown(r'''
                        #
                        #
                        ##
                        The Amazon rainforest is one of the [most biodiverse ecosystems in the world](https://www.copernicus.eu/en/media/image-day-gallery/deforestation-mato-grosso-brazil) and is considered critical in tackling climate change. 
                        Yet, there is evidence that the Amazon forest system [could soon reach a tipping point](https://www.nature.com/articles/s41586-023-06970-0), leading to large-scale collapse.
                        Generative vision models for geospatial data - so called **Geospatial Foundation Models (GeoFMs)** - offer a new and powerful technology for mapping the earth's surface at a continental scale, 
                        providing stakeholders with the tooling to detect and monitor ecosystem change like forest degradation.
                        
                        ''',style={"font-size": "1.25rem", "margin-bottom": "2rem","text-align": "center"})

                with solara.Columns([1,1,1],gutters_dense=False):
                    with solara.Column(align="end"):
                        with solara.Card(title="Map Similar Vegetation",
                                        subtitle="Quickly map any surface type with a RAG-inspired geospatial embedding search",
                                        style={"border-radius": "5px","background": "#f5f5f5"}):
                            with solara.Column(align="center",style={"background": "#f5f5f5"}):
                                solara.Image(f"{DATA_DIR}/similarity_fade_transition_v3.gif",width="400px",classes=["rounded-image"])
                            with solara.CardActions():
                                LinkedButton(path="/run-similarity-search", label="Try it Out")

                    with solara.Column(align="end"):
                        with solara.Card(title="Detect Ecosystem Change",
                                            subtitle="Analyze time series of geospatial embeddings to identify surface disruptions over time",
                                            style={"border-radius": "5px","background": "#f5f5f5"}):
                            with solara.Column(align="center",style={"background": "#f5f5f5"}):
                                solara.Image(f"{DATA_DIR}/fading_animation_change_simple.gif",width="400px",classes=["rounded-image"])
                            with solara.CardActions():
                                LinkedButton(path="/detect-change", label="Try it Out")
                    with solara.Column(align="end"):
                        with solara.Card(title="Fine-Tune a Surface Model",
                                            subtitle="Deploy a fine-tuned segmentation model for classifying surface types at pixel level",
                                            style={"border-radius": "5px","background": "#f5f5f5"}):
                            with solara.Column(align="center",style={"background": "#f5f5f5"}):
                                solara.Image(f"{DATA_DIR}/custom_model_fade_transition_v2.gif",width="400px",classes=["rounded-image"])
                            with solara.CardActions():
                                LinkedButton(path="/apply-custom-model", label="Try it Out")
        solara.Text("")    

@solara.component
def Page():
    solara.Style(page_style)
    solara.lab.theme.dark = False
    solara.lab.theme.themes.light.primary = "#000000" 
    return LandingPage()
