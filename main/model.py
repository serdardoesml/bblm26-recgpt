from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class RecGPTConfig(PretrainedConfig):
    model_type = "recgpt"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 640,
        embedding_size: int = 192, # Allows for factorized embeddings, only makes sense at babylm scale.
        head_dim: int = 64,
        intermediate_size: int = 10240,
        recursive_depth: int = 16,
        max_position_embeddings: int = 256,
        pad_token_id: int = 0, # Padding is determined by segment_ids, this is only used for embeddings/HF metadata.
        tie_word_embeddings: bool = False, # Tied embeddings greatly hurt performance for recursive models.
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if hidden_size % head_dim != 0:
            raise ValueError("hidden_size must be divisible by head_dim.")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.intermediate_size = intermediate_size
        self.recursive_depth = recursive_depth
        self.max_position_embeddings = max_position_embeddings
        self.is_decoder = True
        self.use_cache = False


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, use_bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if use_bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
        if self.bias is not None:
            x = x + self.bias
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        
        # We register these as buffers to ensure they get moved to device together with the model.
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cos = self.cos[position_ids].unsqueeze(2).to(dtype=x.dtype)
        sin = self.sin[position_ids].unsqueeze(2).to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_odd * cos + x_even * sin
        return out


def _select_flex_backend(device: torch.device) -> str:
    if device.type != "cuda":
        raise RuntimeError("RecGPT attention requires a CUDA/ROCm accelerator.")
    if torch.version.hip is None:
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 9:
            return "FLASH"
    return "TRITON"

ROPE_THETA = 10000.0

class SelfAttention(nn.Module):
    def __init__(self, config: RecGPTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # One fused projection for QKV.
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Zero init (Idea from modded-nanogpt speedrun, empirically seems to work well).
        nn.init.zeros_(self.out.weight)
        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings, ROPE_THETA)

        # Gated Attention (https://arxiv.org/pdf/2505.06708)
        # SDPAHeadwiseGate: per-head sigmoid gate applied to attention output.
        # Empirically, attention gating should benefit us since we don't use any <|BOS|> token during training,
        # meaning the model has no attention sinks. GA should reduce the need for attention sinks.
        self.gate = nn.Linear(config.hidden_size, self.num_heads, bias=True)
        nn.init.zeros_(self.gate.weight)

        # Since the gated attention paper finds that the model converges toward a more sparse gate,
        # we initialize the gate bias with 0.0 (so the sigmoid of the bias is 0.5).
        # 0.5 is right in the middle, not too high to start with default behavior, not too low to enforce sparsity early on.
        # TODO: Rewrite this comment more clearly
        # TODO: Re-consider if bias is even necessary
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
        backend: str,
    ) -> torch.Tensor:
        from torch.nn.attention.flex_attention import flex_attention

        batch_size, seq_len, hidden_size = x.shape
        # Project to QKV and reshape to [B, T, 3, H, D].
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Unparameterized QK norm. We used parameterized QK norms in the old repo,
        # but this lean version keeps them fixed for now.
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        # Pick cos/sin for each token position, then broadcast over heads.
        q = self.rope(q, position_ids).transpose(1, 2)
        k = self.rope(k, position_ids).transpose(1, 2)
        v = v.transpose(1, 2)

        y = flex_attention(q, k, v, block_mask=block_mask, kernel_options={"BACKEND": backend})
        gate = torch.sigmoid(self.gate(x)).view(batch_size, seq_len, self.num_heads, 1)
        y = y.transpose(1, 2) * gate
        y = y.contiguous().view(batch_size, seq_len, hidden_size)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, config: RecGPTConfig):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # Standard dense ReLU^2 MLP with zero init output
        # (Idea from modded-nanogpt speedrun, empirically seems to work well).
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())


class RecursiveBlock(nn.Module):
    def __init__(self, config: RecGPTConfig):
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attn_norm: RMSNorm,
        mlp_norm: RMSNorm,
        position_ids: torch.Tensor,
        block_mask,
        backend: str,
    ) -> torch.Tensor:
        # We do pre-norm and QK norm.
        # We used to do a Gemma 3 style post-norm, but removed it to improve stability
        # and keep the residual stream norm in check. Seems to work fine.
        # Update: Tried KEEL norm paper with residual scaling, it hurt performance.
        x = x + self.attn(attn_norm(x), position_ids, block_mask, backend)
        x = x + self.mlp(mlp_norm(x))
        return x


class RecGPTForCausalLM(PreTrainedModel):
    config_class = RecGPTConfig
    base_model_prefix = "model"

    def __init__(self, config: RecGPTConfig):
        super().__init__(config)
        self.use_factorized = config.embedding_size != config.hidden_size

        # Factorized Embeddings (https://arxiv.org/pdf/1909.11942)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        if self.use_factorized:
            self.e_to_h = nn.Linear(config.embedding_size, config.hidden_size, bias=False)
            self.h_to_e = nn.Linear(config.hidden_size, config.embedding_size, bias=False)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        # This is the main recursive model idea: a single block is reused at every depth.
        # The norms are depth-specific, but the attention and MLP weights are shared.
        self.block = RecursiveBlock(config)
        self.attn_norms = nn.ModuleList([RMSNorm(config.hidden_size, use_bias=True) for _ in range(config.recursive_depth)])
        self.mlp_norms = nn.ModuleList([RMSNorm(config.hidden_size, use_bias=True) for _ in range(config.recursive_depth)])
        self.final_norm = RMSNorm(config.hidden_size)

        self.post_init()
        if config.tie_word_embeddings:
            self.tie_weights()

    def init_weights(self):
        return

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return getattr(self, "lm_head", None)

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        position_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutput | tuple[torch.Tensor, ...]:
        # Training uses one packed, block-masked sequence per microbatch, so B is always 1.
        # We keep the batch dimension because HF expects it.
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")
        if segment_ids.shape != input_ids.shape:
            raise ValueError("segment_ids must match input_ids shape.")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids shape.")

        from torch.nn.attention.flex_attention import create_block_mask

        batch_size, seq_len = input_ids.shape

        def mask_mod(b, h, q_idx, kv_idx):
            """
            FlexAttention calls mask_mod with scalar/block index tensors and uses the
            result to build a block-sparse attention mask. This is the only place where
            document boundaries are enforced.
            
            segment_ids[b, t] >= 0 means a real token. segment_ids[b, t] == -1 means
            padding. Tokens can only attend causally within the same segment, so packed
            documents in the same row still have hard attention boundaries.
            """
            
            valid = segment_ids[b, q_idx] >= 0
            same_segment = segment_ids[b, q_idx] == segment_ids[b, kv_idx]
            causal = kv_idx <= q_idx
            return valid & same_segment & causal

        # BlockMask is built once per forward and reused at every recursive depth.
        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=self.config.num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=input_ids.device,
        )
        backend = _select_flex_backend(input_ids.device)

        x = self.embed_tokens(input_ids)
        if self.use_factorized:
            x = self.e_to_h(x)
        for attn_norm, mlp_norm in zip(self.attn_norms, self.mlp_norms):
            x = self.block(x, attn_norm, mlp_norm, position_ids, block_mask, backend)

        x = self.final_norm(x)
        if self.use_factorized:
            x = self.h_to_e(x)
        if hasattr(self, "lm_head"):
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        use_return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if use_return_dict:
            return CausalLMOutput(loss=loss, logits=logits)
        if loss is None:
            return (logits,)
        return (loss, logits)
