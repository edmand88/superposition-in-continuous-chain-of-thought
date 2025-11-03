import torch

def find_similar_embeddings(model, tokenizer, target_embedding, n=10):
    """
    Find the n most similar embeddings to the target embedding using cosine similarity
    Args:
        target_embedding: The embedding vector to compare against
        n: Number of similar embeddings to return (default 3)
    Returns:
        List of tuples containing (word, similarity_score) sorted by similarity
    """
    # Convert target to tensor if not already
    if not isinstance(target_embedding, torch.Tensor):
        target_embedding = torch.tensor(target_embedding)

    # Get all embeddings from the model
    all_embeddings = model.embedding.weight

    # Compute cosine similarity between target and all embeddings
    similarities = torch.nn.functional.cosine_similarity(
        target_embedding.unsqueeze(0),
        all_embeddings
    )

    # Get top n similar embeddings
    top_n_similarities, top_n_indices = torch.topk(similarities, n)

    # Convert to word-similarity pairs
    results = []
    for idx, score in zip(top_n_indices, top_n_similarities):
        word = tokenizer.decode(idx)
        results.append((word, score.item()))

    return results
  
def prompt_to_embeddings(model, tokenizer, prompt:str):
    # tokenize the input text
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']

    # make a forward pass
    embeddings, _ = model(input_ids)

    # directly use the embeddings layer to get embeddings for the input_ids
    embeddings = embeddings

    # print each token
    token_id_list = tokenizer.encode(prompt, add_special_tokens=True)
    token_str = [tokenizer.decode(t_id, skip_special_tokens=True) for t_id in token_id_list]

    return token_id_list, embeddings, token_str

def find_similar_logits(model, tokenizer, embedding_vector, n=10):
    """
    Use unembedding (lm_head) to convert embedding_vector to logit and find top-k
    """
    if not isinstance(embedding_vector, torch.Tensor):
        embedding_vector = torch.tensor(embedding_vector)
    logits = model.unembedding(embedding_vector.unsqueeze(0))  # [1, vocab]
    topk = torch.topk(logits, n)
    results = []
    for idx, val in zip(topk.indices[0], topk.values[0]):
        word = tokenizer.decode(idx)
        results.append((word, val.item()))
    return results

def get_token_embedding(model, tokenizer, word):
    _, embedding_word, _ = prompt_to_embeddings(model, tokenizer, word)
    return embedding_word[0, -1, :]

def get_all_tokens(tokenizer):
    tokens = list(tokenizer.get_vocab().keys())
    return tokens