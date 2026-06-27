"""Auxiliary model for NextLat latent prediction"""
import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass

@dataclass
class NL_Aux_Config:
    input_hidden_size: int # Input size (Concatenation of hidden and next token embedding vectors)
    hidden_size: int = -1 # Residual dimension (Projected down from concatenated input), if set to -1 defaults to input_hidden_size
    intermediate_size: int = 640 # MLP intermediate dimension
    mlp_count: int = 2 # MLP layer count

class MLP(nn.Module): # Coped MLP definition from model, not re-using to allow for flexibility
    def __init__(self, config: NL_Aux_Config):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())

class NL_Aux_Model(nn.Module):
    # Takes in the normalized hidden states of the main model along with the embeddings of the next token, and predicts the hidden states of the next token.
    def __init__(self, config: NL_Aux_Config):
        super().__init__()
        self.config = config

        if config.hidden_size == -1:
            config.hidden_size = config.input_hidden_size

        self.input_proj = nn.Linear(config.input_hidden_size * 2, config.hidden_size, bias=False)
        if config.input_hidden_size != config.hidden_size: # Allow for aux models hidden size to be different from main model. This could be useful to create a bottleneck.
            self.out_proj = nn.Linear(config.hidden_size, config.input_hidden_size, bias=False)
        else:
            self.out_proj = nn.Identity()
        self.mlps = nn.Sequential(*[MLP(config) for _ in range(config.mlp_count)])

    def forward(self, hidden, embeddings, segment_ids): 
        # Inputs expected in shape [batch, seq_len, hidden_size], except segment IDs which are [batch, seq_len].
        # Returns loss only
        # We use detach for x+1 and h+1 to ensure gradients only flow through the previous tokens.

        # Prevent cross-document and padding token prediction.
        valid_transition = (segment_ids[:, :-1] >= 0) & (segment_ids[:, :-1] == segment_ids[:, 1:])

        x = torch.cat([hidden[:, :-1], embeddings[:, 1:].detach()], dim=-1) # Concatenate hidden with next token's embedding
        x = self.input_proj(x)
        x = self.mlps(x)
        x = self.out_proj(x) + hidden[:, :-1] # Residual connection

        pred = x[valid_transition]
        target = hidden[:, 1:][valid_transition].detach()
        if pred.numel() == 0:  # Return zero loss if no valid transitions
            return x.sum() * 0.0

        return F.smooth_l1_loss(pred, target)
