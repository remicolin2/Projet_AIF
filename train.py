import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from model import MobileNet
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.image as mpimg
import argparse
import numpy as np
from annoy import AnnoyIndex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import subprocess
# download_command = "kaggle datasets download ghrzarea/movielens-20m-posters-for-machine-learning"
# subprocess.run(download_command.split(), check=True)

# unzip_command = "unzip movielens-20m-posters-for-machine-learning.zip"
# subprocess.run(unzip_command.split(), check=True)

class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _= super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path
    

def train(net):
    model = net.eval()

def test(model, dataloader):
    features_list = []
    paths_list = []
    for x, paths in tqdm(dataloader):
        with torch.no_grad():
            embeddings = model(x.cuda())
            features_list.extend(embeddings.cpu().numpy())
            paths_list.extend(paths)

            df = pd.DataFrame({
                'features': features_list,
                'path': paths_list
            })
    return df

def create_annoy_vec(features_list):
    dimension = 576 
    n_trees = 10
    index = AnnoyIndex(dimension, 'angular')
    for i, vecteur in enumerate(features_list):
        index.add_item(i, vecteur)
    index.build(n_trees)
    index.save('annoy_db_1.ann')

def recom(df, idx):
    features = df['features']
    features = np.vstack(features)
    cosine_sim = cosine_distances(features, features)
    recos = cosine_sim[idx].argsort()[1:6]
    reco_posters = df.iloc[recos]['path'].tolist()
    return reco_posters

if __name__=='__main__':
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    normalize = transforms.Normalize(mean, std)
    inv_normalize = transforms.Normalize(
    mean= [-m/s for m, s in zip(mean, std)],
    std= [1/s for s in std]
    )   
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    dataset = ImageAndPathsDataset('MLP-20M', transform)

    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)
    net = MobileNet().to(device)
    train(net)

    # Dans API ?
    features = test(net, dataloader)
    create_annoy_vec(features['features'])
    #parser = argparse.ArgumentParser()
    #parser.add_argument('movie', type=int, default = 10)
    #args = parser.parse_args()
    #movie = args.movie
    #recoms_movies = recom(features, movie)
    #print(f"The recommended movies are :{recoms_movies}")

