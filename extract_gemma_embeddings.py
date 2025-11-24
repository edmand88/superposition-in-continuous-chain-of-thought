import torch
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, Gemma3ForCausalLM

access_token = "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"

#should try other maybe larger models
model_id = "google/gemma-3-1b-it"
filename_safe = model_id.replace("/", "_")
model_name = model_id

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

vocab_size = tokenizer.vocab_size
print("Vocab size:", vocab_size)

# Load the pre-trained model
model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, token=access_token
).eval()

embedding_dim = model.model.embed_tokens.weight.shape
print("Embedding dimension:", embedding_dim)

# extract embedding
embeddings = model.model.embed_tokens.weight
print(f"Extracted Embeddings Layer for {model_name}: {embeddings}")
torch.save(embeddings.detach(), os.path.join(f"{filename_safe}_embeddings_gemma.pth"))

# extract unembedding
# weight tied
unembedding = model.model.embed_tokens.weight
print(f"Extracted Unembeddings Layer for {model_name}: {unembedding}")
torch.save(unembedding.detach(), os.path.join(f"{filename_safe}_unembeddings_gemma.pth"))