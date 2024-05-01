from flask import Flask, request, jsonify
from annoy import AnnoyIndex
from model import MobileNet, process
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import logging
from nlp_model import recommend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# Load the Annoy database
annoy_db = AnnoyIndex(576, metric='angular')  # Here 2 is the dimension of the vectors in the database in my example
                                            # you would replace it with the dimension of your vectors
annoy_db.load('annoy_db_1.ann')  # Replace 'annoy_db.ann' with the path to your Annoy database


@app.route('/') # This is the home route, it just returns 'Hello world!'
def index():    # I use it to check that the server is running and accessible it's not necessary
    return 'Hello world!'

@app.route('/reco', methods=['POST']) # This route is used to get recommendations
def reco():
    vector = np.array(request.json['image'],dtype=np.uint8) # Get the vector from the 
    model = MobileNet()
    model.eval()
    #to_pil = ToPILImage()
    vector=Image.fromarray(vector).convert('RGB')
    image = process(1, vector)
    logging.info('La variable est ')
    with torch.no_grad():
        vector = model(image.unsqueeze(0))
        #print(vector)
    print(vector[0].numpy().shape)
    closest_indices = annoy_db.get_nns_by_vector(vector[0].numpy(), 5) # Get the 2 closest elements indices  
    logging.info('La variable est : %s', closest_indices)
    return jsonify(closest_indices) # Return the reco as a JSON

@app.route('/prompt_text', methods=['POST'])
def prompt():
    # data = request.json
    # query = data.get('description')
    # recommendations = recommend(query)
    # return jsonify(recommendations)
    query = request.form.get('prompt')
    recommendations = recommend(query)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Run the server on port 5000 and make it accessible externally
