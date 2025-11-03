import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import torch
from load_embeddings import load_embedding_model

#Kmeans

random_state = 42
n_clusters = 10

model, tokenizer = load_embedding_model(151936, 1536, "embeddings_qwen.pth", "unembeddings_qwen.pth", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
all_tokens = list(tokenizer.get_vocab().keys())
emb_weight = torch.load("embeddings_qwen.pth")["weight"]
all_embeddings = emb_weight.detach().cpu().numpy()

print(f"Embeddings shape: {all_embeddings.shape}")

kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(all_embeddings)
token_clusters = dict(zip(all_tokens, kmeans.labels_))
print(kmeans.labels_)

for token, label in list(token_clusters.items())[:10]:
    print(f"Token: {token}, Cluster: {label}")


#PCA
pca = PCA(n_components=2)
pca.fit(all_embeddings)
