import torch
from attention import SELEXCrossAttentionModel

def test_selex_cross_attention_model_shapes():
    embed_dim = 16
    num_heads = 4
    batch_size_context = 8
    batch_size_query = 5
    seq_len = 10

    model = SELEXCrossAttentionModel(embed_dim=embed_dim, num_heads=num_heads)

    # Random context and query embeddings
    context_embeddings = torch.randn(batch_size_context, seq_len, embed_dim)
    query_embeddings = torch.randn(batch_size_query, seq_len, embed_dim)

    output = model(context_embeddings, query_embeddings)
    # Output should be [batch_size_query]
    assert output.shape == (batch_size_query,), f"Expected output shape {(batch_size_query,)}, got {output.shape}"

def test_selex_cross_attention_model_forward_pass():
    embed_dim = 8
    num_heads = 2
    N = 3
    m = 2
    L = 6

    model = SELEXCrossAttentionModel(embed_dim=embed_dim, num_heads=num_heads)
    context_embeddings = torch.ones(N, L, embed_dim)
    query_embeddings = torch.zeros(m, L, embed_dim)

    output = model(context_embeddings, query_embeddings)
    # Output should be [m]
    assert output.shape == (m,)
    # Output should be finite
    assert torch.isfinite(output).all()

if __name__ == "__main__":
    test_selex_cross_attention_model_shapes()
    test_selex_cross_attention_model_forward_pass()
    print("All tests passed.")