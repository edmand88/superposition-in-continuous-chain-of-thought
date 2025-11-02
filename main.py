import torch
import torch.nn as nn
from transformers import AutoTokenizer
from load_embeddings import load_embedding_model
from utils import find_similar_embeddings, prompt_to_embeddings, find_similar_logits

# model, tokenizer = load_embedding_model(151936, 1536, "embeddings_qwen.pth", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
vocab_size = 151936
dimensions = 1536
project_dir = "/Users/wuyuchen/Desktop/SCCOT/" # adjust this path to yours
model, tokenizer = load_embedding_model(
    vocab_size,
    dimensions,
    emb_path=f"{project_dir}/embeddings_qwen.pth",
    unemb_path=f"{project_dir}/unembedding_qwen.pth",
    tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# token_id_list_cat, embedding_cat, token_str_cat = prompt_to_embeddings(model, tokenizer, "cat")
# token_id_list_dog, embedding_dog, token_str_dog = prompt_to_embeddings(model, tokenizer, "dog")
# token_id_list_mouse, embedding_mouse, token_str_mouse = prompt_to_embeddings(model, tokenizer, "mouse")

# embedding_cat = embedding_cat[0, -1, :]
# embedding_dog = embedding_dog[0, -1, :]
# embedding_mouse = embedding_mouse[0, -1, :]

# weighted_embedding = 0.33*embedding_cat + 0.33*embedding_dog + 0.34*embedding_mouse

# print("Embedding cat: ", embedding_cat)
# print("Embedding dog: ", embedding_dog)
# print("Embedding mouse: ", embedding_mouse)
# print("Cat + Dog + Mouse embedding: ", weighted_embedding)

# most_similar_words = find_similar_embeddings(model, tokenizer, weighted_embedding, n=20)
# print(most_similar_words)

def get_token_embedding(word):
    tok = tokenizer(word, return_tensors="pt")["input_ids"]
    emb, _ = model(tok)
    return emb[0, -1, :]

cat = get_token_embedding("cat")
dog = get_token_embedding("dog")
mouse = get_token_embedding("mouse")

superpos = 0.33 * cat + 0.33 * dog + 0.34 * mouse

print("Superposition vector ready. Computing top logits ...")
top_words = find_similar_logits(model, tokenizer, superpos, n=20)
for w, s in top_words:
    print(f"{w:15s}  {s:.4f}")
