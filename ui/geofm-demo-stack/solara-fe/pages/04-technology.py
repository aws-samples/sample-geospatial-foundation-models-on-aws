import solara
from pathlib import Path
solara.lab.theme.dark = False
solara.lab.theme.themes.light.primary = "#000000" 

DATA_DIR = Path(__file__).parent.parent / "data"

# Load style
HERE = Path(__file__).parent.parent
app_style = (HERE / "style.css").read_text()

@solara.component
def LinkedButton(path: str, label: str):
    with solara.Link(path) as link:
        solara.Button(label)
    return link

@solara.component
def CenterAlignedImage(path: str, width: str):
    with solara.Column(align="center") as img:
        solara.Image(image=path,width=width)
    return img


@solara.component
def Page():
    solara.Style(app_style)
    solara.lab.theme.dark = False
    solara.lab.theme.themes.light.primary = "#000000"   

    with solara.Columns([1, 6, 1]):
        
        solara.Text("")  # Empty column for left margin
        
        with solara.Column(style={"width": "100%"}): #main column with content

            solara.Markdown(r'''
                            # Technical Deep Dive
                            ''',style={"font-size": "16px"})

            solara.Markdown(r'''
                            ## How it works
                            Geospatial Foundation Models (GeoFMs) are [Vision Transformers (ViTs)]() tailored for geospatial data. 
                            These large, generative ML models are pre-trained on massive geospatial imagery datasets. 
                            This makes them general purpose models that can be leveraged for a wide array of geospatial applications.
                            ''',style={"font-size": "16px"})
            
            solara.Image(f"{DATA_DIR}/GeoFM_demo_architecture-Flow-chart.png")

            solara.Markdown(r'''
                                **This demonstrator showcases the power of GeoFMs for ecosystem monitoring. It is powered by [**Clay-v01**](https://clay-foundation.github.io/model/index.html), a state-of-the-art GFM available on [**HuggingFace**](https://huggingface.co/made-with-clay/Clay). Satellite imagery used in this demo is from ESA's Sentinel-2 mission and is hosted on the [AWS Registry of Open Data](https://registry.opendata.aws/sentinel-2/).**
                            ''',style={"font-size": "16px"})
            
            solara.Markdown(r'''
                            ## Application Architecture
                            The application follows a three-tiered architecture comprising a Frontend, an AI/ML and Analytics Backend, and a Data Layer. 
                            Additionally, it incorporates two asynchronous [SageMaker](https://aws.amazon.com/sagemaker/) pipelines for embedding generation and GeoFM fine-tuning. 
                            The main components are:

                            1. **Solara Frontend** - A reactive containerized web application built with [Solara](https://solara.dev/documentation), hosted on [AWS Fargate](https://aws.amazon.com/fargate/), and served through [CloudFront](https://aws.amazon.com/cloudfront/) CDN. Authentication is handled by [Amazon Cognito](https://aws.amazon.com/pm/cognito/?gclid=CjwKCAiAudG5BhAREiwAWMlSjIMQlGJruxjBD8L18Z4S84V0GeBmiqRpt9dZcvaRquyQZakBXj51JhoCRXUQAvD_BwE&trk=3e612152-ae90-4f91-abda-680eead5127a&sc_channel=ps&ef_id=CjwKCAiAudG5BhAREiwAWMlSjIMQlGJruxjBD8L18Z4S84V0GeBmiqRpt9dZcvaRquyQZakBXj51JhoCRXUQAvD_BwE:G:s&s_kwcid=AL!4422!3!651541907485!e!!g!!amazon%20cognito!19835790380!146491699385).
                            2. **Serve Geospatial Imagery Lambda** - Provides a tile server using [Amazon ElastiCache](https://aws.amazon.com/elasticache/) for low-latency geospatial imagery serving.
                            3. **Search Similar Regions Lambda** - Performs similarity searches based on a chip ID and vector search parameters, utilizing [Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/).
                            4. **Detect Change Lambda** - Conducts change detection by analyzing time series of embedding vectors retrieved from Amazon OpenSearch Service.
                            5. **Apply Custom Model Lambda** - Generates predictions for given chip IDs using a SageMaker endpoint hosting a fine-tuned GeoFM (see Step 6).
                            6. **GeoFM Fine-Tuning Pipeline** - A [SageMaker Pipeline](https://aws.amazon.com/sagemaker/pipelines/) for data preparation and fine-tuning of a [HuggingFace](https://huggingface.co/)-hosted GeoFM model on custom datasets.
                            7. **Geo Embedding Generation Pipeline** - A SageMaker Pipeline for retrieving, preprocessing [Sentinel-2](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2) data, and generating geospatial embeddings using a pre-trained GeoFM from HuggingFace Hub.
                            
                            ''',style={"font-size": "16px"})
            
            summary_text = "View the Architecture Diagram"
            img = solara.Image(f"{DATA_DIR}/GeoFM_demo_architecture_detailed.png")
            additional_content = [img]
            solara.Details(
                summary=summary_text,
                children=additional_content,
                expand=False
            )

            
            solara.Markdown(r'''
                            ## Geospatial Foundation Models (GeoFMs)

                            GeoFMs are a type of pre-trained Vision Transformer (ViT) models that are specifically adapted to geospatial data sources and that have been pre-trained on earth-scaler unlabeled geospatial data using a technique called Self-Supervised Learning (SSL). 
                            
                            GeoFMs offer immediate value without training. They excel as embedding models for geospatial similarity search and ecosystem change detection. With minimal labeled data, GeoFMs can be fine-tuned for custom tasks like classification, semantic segmentation, or pixel-level regression.                                 
                            ''',style={"font-size": "16px"})
            
            summary_text = "Dive deep into GeoFM architectures and the process of fine-tuning GeoFMs"

            arch_details = solara.Markdown(r'''
                            
                            ### Model Architecture Deep Dive
                            Architecturally, GeoFMs build on the vision transformer (ViT) architecture first introduced in the seminal 2022 research paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). 
                            To account for the specific properties of geospatial data (multiple channels, varying electromagnetic spectrum coverage, spatio-temporal ), GeoFMs incorporate several architectural innovations primarily related to varying input size and positional embeddings that capture spatio-temporal aspects.
                            The pre-training is conducted on unlabeled geospatial data sampled from across the globe to ensure all ecosystems and surface types are represented appropriately in the trainings set.
                            The image below (taken from the paper [SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery](https://arxiv.org/abs/2207.08051)) illustrates the self-supervised learning (SSL) pre-training process. 
                            During SSL, a certain amount of patches from an image are withheld (up to 75%) and the model learns to reconstruct the image.
                            The decoder used for reconstruction is later discarded and only the encoder is retained, and subsequently referred to as the GeoFM.
                            ''',style={"font-size": "16px"})
            
            model_arch_img = CenterAlignedImage(path="https://sustainlab-group.github.io/SatMAE/static/images/teaser.png", width="95%")

            open_weights = solara.Markdown(r'''
                                                 
                            ### Open Weight GeoFMs (Selection)
                            GeoFMs represent an emerging research field, with state-of-the-art models coming primarily from leading research institutions. Many of these labs provide model weights under very permissive licenses.
                            Below is a selection of open-source GeoFM models and links to model weights.
                            
                            * [SatVision-Base](https://huggingface.co/nasa-cisto-data-science-group/satvision-base), NASA GSFC CISTO Data Science Group (Available on HuggingFace)
                            * [Prithvi-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M), NASA & IBM (Available on HuggingFace)
                            * [DOFA](https://huggingface.co/XShadow/DOFA), Technical University of Munich (Available on HuggingFace)
                            * [SatMAE](https://zenodo.org/records/7338613), Stanford University (Available on Zenodo)
                            * [Clay-v1](https://huggingface.co/made-with-clay/Clay), Clay Foundation (Available on HuggingFace)
                                           
                            This demonstrator uses the Clay-v1 model.
                                
                            ''',style={"font-size": "16px"})
            
            finetune_details = solara.Markdown(r'''
                   
                            ### Fine-Tuning GeoFMs
                            Fine-tuning involves adding a custom decoder atop a frozen encoder, which generates a latent representation of key geospatial image features. 
                            This process produces a reduced-size embedding tensor for the class token and each 16x16 patch. In the Clay model (used in this demo), each embedding vector is 768-dimensional, 
                            with added positional encodings, resulting in a nx64x768 patch embedding vector plus a single-dimensional class token vector (n is the number of bands).
                            
                            For classification tasks, such as determining forest presence, a linear layer or multiple layers can be added to reproject the embeddings to the desired number of classes. 
                            Typically, this utilizes the class token embeddings, projecting from 768 dimensions to the target class count using a simple linear layer or a multi perceptron (MLP). 
                            Even simpler, an out-of-the-box model like an XGBoost classifiers can be trained on top of the embeddings.

                            Pixel-wise regression or classification (segmentation) tasks require a more complex decoder architecture to reconstruct the embedding tensor to original image dimensions. 
                            For instance, a forest vs. non-forest binary segmentation mask would require reconstructing the embedding tensor to a 256x256x2 image (2 channels because we have 2 classes). 
                            Various methods can achieve this, including MLPs, U-Net-type approaches, or SegFormer, which adapts Feature Pyramid Networks for use with transformer backbones.                                
                            ''',style={"font-size": "16px"})

            finetune_img = CenterAlignedImage(path=f"{DATA_DIR}/fine-tune-decoders.png", width="95%")

            solara.Details(
                summary=summary_text,
                children=[arch_details,model_arch_img,open_weights,finetune_details,finetune_img],
                expand=False
            )

        solara.Text("")  # Empty column for right margin