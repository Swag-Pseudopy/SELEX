import torch
import torch.nn as nn
import torch.nn.functional as F

from args import get_arg_parser

parser = get_arg_parser()
args = parser.parse_args()

class SELEXCrossAttentionModel(nn.Module):
    """
    SELEXCrossAttentionModel implements a cross-attention neural network architecture for modeling relationships between context
    and query sequences, inspired by SELEX experiments. It encodes sequences via self-attention, injects relative abundance
    information into the class token embeddings, and applies cross-attention before predicting log abundance changes.
    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads for both self-attention and cross-attention modules.
    Inputs:
        context_embeddings (Tensor): [N, L, d] embeddings of N context sequences.
        context_abundances (Tensor): [N] current relative abundances (rho) of context sequences.
        query_embeddings (Tensor): [m, L, d] embeddings of m query sequences.
        query_abundances (Tensor): [m] current relative abundances (rho) of query sequences.
    Returns:
        Tensor: Predicted log abundance ratios ln(rho_next / rho_current) of shape [m].
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super(SELEXCrossAttentionModel, self).__init__()
        self.embed_dim = embed_dim

        # In-sequence self-attention (per sequence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.self_attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # MLP to project log-abundance scalar -> d-dimensional vector
        self.abundance_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Predict log abundance ratio ln(rho_next / rho_current)
        num_layers = args.num_layers
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            *[
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU()
            ) for _ in range(num_layers - 1)
            ],
            nn.Linear(embed_dim, 1)
        )

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_abundances: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_abundances: torch.Tensor
    ) -> torch.Tensor:
        # In-sequence self-attention
        context_encoded = self.self_attn_encoder(context_embeddings)  # [N, L, d]
        query_encoded = self.self_attn_encoder(query_embeddings)      # [m, L, d]

        # Extract CLASS token (index 0)
        context_class = context_encoded[:, 0, :]  # [N, d]
        query_class = query_encoded[:, 0, :]      # [m, d]

        # Mid-layer injection of abundance information
        # Compute log-abundance features and add to class tokens
        eps = 1e-8
        # Shape log_rho: [N, 1]
        log_rho_ctx = torch.log(context_abundances.unsqueeze(-1) + eps)
        ctx_r = self.abundance_mlp(log_rho_ctx)           # [N, d]
        context_class = context_class + ctx_r

        # Shape log_rho: [m, 1]
        log_rho_qry = torch.log(query_abundances.unsqueeze(-1) + eps)
        qry_r = self.abundance_mlp(log_rho_qry)           # [m, d]
        query_class = query_class + qry_r

        # Cross-attention: queries attend to contexts
        # Prepare inputs for MultiheadAttention
        query_input = query_class.unsqueeze(1)  # [m, 1, d]
        key = context_class.unsqueeze(0).expand(
            query_class.size(0), -1, -1
        )  # [m, N, d]
        value = key

        # Iteratively apply cross-attention num_cross_att times
        num_cross_att = args.num_cross_att
        cross_output = query_input
        for _ in range(num_cross_att):
            cross_output, _ = self.cross_attn(cross_output, key, value)  # [m, 1, d]
        cross_output = cross_output.squeeze(1)  # [m, d]

        # Predict log abundance ratio
        prediction = self.output_head(cross_output)  # [m, 1]
        return prediction.squeeze(-1)  # [m]
    
