import requests
import random
import matplotlib.pyplot as plt
from PIL import Image
from annoy import AnnoyIndex
import gradio as gr


# Charger les données depuis le fichier Annoy
index = AnnoyIndex(576, 'angular')
index.load('annoy_db_1.ann')

def process_image(image):
    
    response = requests.post("http://annoy-db:5000/reco", json={"image" :image.tolist()})
    paths = [f"MLP-20M/MLP-20M/{i}.jpg" for i in response.json()["closest_indices"]]
    fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
    for i, path in enumerate(paths):
        img = Image.open(path)
        axs[i].imshow(img)
        axs[i].axis('off')
    return fig 

iface = gr.Interface(fn=process_image, inputs="image", outputs="plot")
iface.launch(server_name="0.0.0.0")  
