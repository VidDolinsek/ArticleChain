import numpy as np
from sentence_transformers import SentenceTransformer
from read_yaml_data import read_yaml_exclude_urls

sentences = ["This is an example sentence", "Each sentence is converted"]

yaml_extraction = read_yaml_exclude_urls("mmc-100.yaml")  # Change to mmc-100.yaml or mmc.yaml as needed

titles = yaml_extraction['lead']  # Assuming 'lead' contains the titles or main text to be embedded


model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


embeddings = model.encode(titles)
print(np.shape(embeddings))
