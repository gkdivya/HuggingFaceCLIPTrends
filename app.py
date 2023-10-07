import gradio as gr
import torch
import clip
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Define apparel categories and attributes
categories = ["t-shirt", "jeans", "jacket", "dress", "shorts", "sweater", "skirt"]
attributes = ["striped", "plain", "floral", "polka dot", "denim", "leather", "wool"]

# Pre-compute embeddings for categories and attributes
with torch.no_grad():
    category_embeddings = model.encode_text(clip.tokenize(categories).to(device))
    attribute_embeddings = model.encode_text(clip.tokenize(attributes).to(device))

def predict_apparel_and_attributes(image):
    # Process image and compute its embedding
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image_input)
    
    # Calculate similarity scores
    category_similarities = (image_embedding @ category_embeddings.T).squeeze(0)
    attribute_similarities = (image_embedding @ attribute_embeddings.T).squeeze(0)
    
    # Get top category and attributes
    top_category = categories[category_similarities.argmax().item()]
    top_attributes = [attributes[i] for i in attribute_similarities.argsort(descending=True)[:3]]  # top 3 attributes

    return top_category, ", ".join(top_attributes)

# Define Gradio interface
iface = gr.Interface(
    fn=predict_apparel_and_attributes, 
    inputs=gr.inputs.Image(label="Upload an apparel image"), 
    outputs=[gr.outputs.Textbox(label="Apparel Category"), gr.outputs.Textbox(label="Apparel Attributes")]
)
iface.launch()
