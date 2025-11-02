import torch
import torch.nn as nn
from transformers import AutoTokenizer
from load_embeddings import load_embedding_model
from utils import find_similar_embeddings, prompt_to_embeddings

model, tokenizer = load_embedding_model(151936, 1536, "embeddings_qwen.pth", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

token_id_list_cat, embedding_cat, token_str_cat = prompt_to_embeddings(model, tokenizer, "cat")
token_id_list_dog, embedding_dog, token_str_dog = prompt_to_embeddings(model, tokenizer, "dog")
token_id_list_mouse, embedding_mouse, token_str_mouse = prompt_to_embeddings(model, tokenizer, "mouse")

embedding_cat = embedding_cat[0, -1, :]
embedding_dog = embedding_dog[0, -1, :]
embedding_mouse = embedding_mouse[0, -1, :]

weighted_embedding = 0.33*embedding_cat + 0.33*embedding_dog + 0.34*embedding_mouse

print("Embedding cat: ", embedding_cat)
print("Embedding dog: ", embedding_dog)
print("Embedding mouse: ", embedding_mouse)
print("Cat + Dog + Mouse embedding: ", weighted_embedding)

most_similar_words = find_similar_embeddings(model, tokenizer, weighted_embedding, n=20)
print(most_similar_words)