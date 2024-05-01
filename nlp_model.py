import torch
import pandas as pd 
from sentence_transformers import SentenceTransformer, util




def recommend(query):
    #Compute cosine-similarities with all embeddings 
    data = pd.read_csv('movies_metadata.csv')
    data.dropna(subset=['title'], inplace=True)
    data.dropna(subset=['overview'], inplace=True)
    data = data[['title', 'overview']]

    model = SentenceTransformer('sentence-transformers/paraphrase-albert-base-v2')
    resumes = data['overview'].to_list()
    compare_emb = []

    for resume in resumes : 
        compare_emb.append(model.encode(resume))
    query_emb = model.encode(query)
    cosine_scores = util.pytorch_cos_sim(query_emb, compare_emb)
    top5_matches = torch.argsort(cosine_scores, dim=-1, descending=True).tolist()[0][1:6]
    return top5_matches


query1 = "a lion with a witch"
recommend(query1)
