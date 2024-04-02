import requests
import random
import matplotlib.pyplot as plt
from PIL import Image
from annoy import AnnoyIndex
import gradio as gr
import os
import logging
import pandas as pd
from flask import Flask
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)


# Charger les données depuis le fichier Annoy
index = AnnoyIndex(576, 'angular')
index.load('annoy_db_1.ann')
base_path = "MLP-20M/MLP-20M"
file_names = pd.read_csv('path.csv')
#annoy_to_file = {i: file_name for i, file_name in enumerate(file_names.iterrows())}
annoy_to_file = file_names.values.tolist()

def process_image(image):
    
    response = requests.post("http://annoy-db:5000/reco", json={"image" :image.tolist()})
    logging.info('gradioapi')
    logging.info(os.getcwd())
    logging.info(os.listdir())
    logging.info(response.json())
    #paths = [f"MLP-20M/MLP-20M/{i}.jpg" for i in response.json()["closest_indices"]]
    base_path = "MLP-20M/MLP-20M"

    paths = [element for sous_liste in [annoy_to_file[idx] for idx in response.json()] for element in sous_liste]

    logging.info('Le chemin : %s', paths)
    #print(paths)
    fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
    for i, path in enumerate(paths):
        img = Image.open(path)
        axs[i].imshow(img) 
        axs[i].axis('off')
    return fig 

iface = gr.Interface(fn=process_image, inputs="image", outputs="plot")
iface.launch(server_name="0.0.0.0")  
