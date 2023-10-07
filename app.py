import gradio as gr
import torch
import clip
from PIL import Image
import numpy as np
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import os

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

def plot_trends(dataframe):
    plt.figure(figsize=(12,6))
    for column in dataframe.columns:
        if column != 'isPartial':
            plt.plot(dataframe.index, dataframe[column], label=column)
    plt.legend()
    plt.title("Google Trends Over Time")
    plt.xlabel("Time")
    plt.ylabel("Interest")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a temporary file and return its path
    path = "trends_plot.png"
    plt.savefig(path)
    plt.close()
    return path
    
def predict_apparel_and_attributes(image):
    #pil_image = Image.fromarray((image * 255).astype(np.uint8))
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_embedding = model.encode_image(image_input)
    
    # Calculate similarity scores
    category_similarities = (image_embedding @ category_embeddings.T).squeeze(0)
    attribute_similarities = (image_embedding @ attribute_embeddings.T).squeeze(0)
    
    # Get top category and attributes
    top_category = categories[category_similarities.argmax().item()]
    top_attributes = [attributes[i] for i in attribute_similarities.argsort(descending=True)[:3]]  # top 3 attributes
    print(f"results:{top_category, ','.join(top_attributes)}")

    # Fetch trends for the top apparel category and attributes
    pytrend = TrendReq()
    keywords = [top_category] + top_attributes
    pytrend.build_payload(kw_list=keywords, timeframe='now 1-H', geo='', gprop='')
    interest_over_time_df = pytrend.interest_over_time()

    # Plot the trends and get the path to the saved plot
    plot_path = plot_trends(interest_over_time_df)

    #trends_text = interest_over_time_df.to_string()

    return top_category, ", ".join(top_attributes), plot_path

demo = gr.Interface(
    predict_apparel_and_attributes,
    gr.Image(type="pil"),
    outputs=[ gr.Textbox(label="Apparel Category"), 
              gr.Textbox(label="Apparel Attributes"),
              gr.Image(label="Google Trends Plot")],  # Output types
    examples=[
        os.path.join(os.path.abspath(''), "images/jeans.jpeg")
    ],
)

if __name__ == "__main__":
    demo.launch()
