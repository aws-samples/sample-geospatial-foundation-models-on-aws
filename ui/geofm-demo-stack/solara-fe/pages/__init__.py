import solara
from pathlib import Path
# Load data
HERE = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent.parent / "data"
app_style = (HERE / "style.css").read_text()

# Change the primary color for the light theme
solara.lab.theme.dark = False
solara.lab.theme.themes.light.primary = "#000000" 

@solara.component
def LinkedImage(src: str, path: str):
    with solara.Link(path) as link:
        solara.Image(src, width="60px",format="svg")
    return link

@solara.component
def Layout(children=[]):
    route_current, routes_all = solara.use_route()

    label_mapper = {
        "Home": "Home",
        "Run Similarity Search":"Geospatial Similarity Search",
        "Detect Change":"Ecosystem Change Detection",
        "Apply Custom Model": "Geospatial ML with a Fine-Tuned Model",
        "Technology":"How it Works",
        "Use Cases":"Use Cases",
        "Team":"Team"
    }

    with solara.AppLayout(sidebar_open=False):
        solara.Style(app_style)
        with solara.AppBar():
            with solara.Row(justify="space-between",style={"background":"#000000"}):
                solara.Text("")
        with solara.Sidebar():
            with solara.Column(style={"width":"150px"}):
                for route in routes_all:
                    with solara.Link(route):
                        solara.Button(
                            label = label_mapper[route.label],
                            color="primary" if route_current == route else None,
                            text=True,
                            full_width=True
                        )
        
        solara.Column(children=children)