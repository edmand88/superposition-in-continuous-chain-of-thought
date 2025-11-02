import torch
import torch.nn as nn
from transformers import AutoTokenizer

# vocab_size = 151936
# dimensions = 1536
# embeddings_filename = r"embeddings_qwen.pth"
# tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def load_embedding_model(vocab_size, dimensions, embeddings_filename, tokenizer_name):

    class EmbeddingModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(EmbeddingModel, self).__init__()
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        def forward(self, input_ids):
            return self.embedding(input_ids)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Initialize the custom embedding model
    model = EmbeddingModel(vocab_size, dimensions)

    # Load the saved embeddings from the file
    saved_embeddings = torch.load(embeddings_filename)

    # Ensure the 'weight' key exists in the saved embeddings dictionary
    if 'weight' not in saved_embeddings:
        raise KeyError("The saved embeddings file does not contain 'weight' key.")

    embeddings_tensor = saved_embeddings['weight']

    # Check if the dimensions match
    if embeddings_tensor.size() != (vocab_size, dimensions):
        raise ValueError(f"The dimensions of the loaded embeddings do not match the model's expected dimensions ({vocab_size}, {dimensions}).")

    # Assign the extracted embeddings tensor to the model's embedding layer
    model.embedding.weight.data = embeddings_tensor

    # put the model in eval mode
    model.eval()

    return model, tokenizer