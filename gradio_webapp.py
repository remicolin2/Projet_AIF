import gradio as gr
import requests
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# A fake dataframe with paths to images
df = pd.DataFrame({
    'index': [0, 1],
    'path': ['images/duck1.jpg', 'images/duck2.jpg']
})

def process_image(image):
    # Here you would extract the vector from the image for example using a mobile net
    # For the example I just generate a random vector
    vector = [random.random(), random.random()]

    # Now we send the vector to the API
    # Replace 'annoy-db:5000' with your Flask server address if different (see docker-compose.yml)
    response = requests.post('http://annoy-db:5000/reco', json={'vector': vector})
    if response.status_code == 200:
        indices = response.json()

        # Retrieve paths for the indices
        paths = df[df['index'].isin(indices)]['path'].tolist()

        # Plot the images
        fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis('off')
        return fig
    else:
        return "Error in API request"

iface = gr.Interface(fn=process_image, inputs="image", outputs="plot")
iface.launch(server_name="0.0.0.0") # the server will be accessible externally under this address

