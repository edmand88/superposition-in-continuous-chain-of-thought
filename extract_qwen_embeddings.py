import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

project_dir = "/Users/wuyuchen/Desktop/SCCOT/"
os.makedirs(project_dir, exist_ok=True)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# try other maybe larger models later
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name)

# extract embedding
embeddings = model.get_input_embeddings()
emb_path = os.path.join(project_dir, "embeddings_qwen.pth")
torch.save(embeddings.state_dict(), emb_path)
print(f"Saved embeddings to {emb_path}")

# extract unembedding
unembedding = model.lm_head.weight
unemb_path = os.path.join(project_dir, "unembedding_qwen.pth")
torch.save(unembedding.detach().cpu(), unemb_path)
print(f"Saved unembedding to {unemb_path}")
