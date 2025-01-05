# Implementation of the T5XXL architecture

# Model Layers
# T5_CONFIG = {
#     "d_ff": 10240,
#     "d_model": 4096,
#     "num_heads": 64,
#     "num_layers": 24,
#     "vocab_size": 32128,
# }

import math
import torch
from torch import nn
from torch.nn import functional as F
from utils import Tokenizer
from transformers import T5TokenizerFast
from clip import ClipModel

class SimplifiedLayerNorm(nn.Module):
    def __init__(self, in_channels: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_channels)) # We load weights in the future
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight.to(device=x.device, dtype=x.dtype) * x # Move both tensors on the same device and perform calculation.

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, relative_attention_bias: bool):
        super().__init__()
        self.num_heads = num_heads

        self.SelfAttention = nn.Module() # Merged T5Attention together with T5SelfAttention class
        self.SelfAttention.q = nn.Linear(in_channels, in_channels, bias=False)
        self.SelfAttention.k = nn.Linear(in_channels, in_channels, bias=False)
        self.SelfAttention.v = nn.Linear(in_channels, in_channels, bias=False)
        self.SelfAttention.o = nn.Linear(in_channels, in_channels, bias=False)

        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)

        self.layer_norm = SimplifiedLayerNorm(in_channels)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ): 
        relative_buckets = 0 # [0, inf)
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # Half the buckets are used for exact position increments
        # Other half used logarithmically bigger bins for distances up to max_distance
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / (math.log(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).to(torch.long)

        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length)[:, None]
        memory_position = torch.arange(key_length)[None: ]
        relative_position = (memory_position - context_position)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )
        values = self.relative_attention_bias(relative_position_bucket) # (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0) # (1, num_heads, query_length, key_length)
        return values

    def forward(self, x: torch.Tensor, past_bias=None):
        residual = x
        
        x = self.layer_norm(x)
        q = self.SelfAttention.q(x)
        k = self.SelfAttention.k(x)
        v = self.SelfAttention.v(x)
        
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1])

        k = k * ((k.shape[-1] / self.num_heads) ** 0.5)
        
        batch_size, _, dim = q.shape
        head_dim = dim // self.num_heads
        
        q = q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, attention_mask=past_bias, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(batch_size, -1, dim)
        
        out = self.o(out)
        out = out + residual
        
        return out, past_bias

class DenseGeluDense(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.wi_0 = nn.Linear(in_channels, hidden_dim, bias=False)
        self.wi_1 = nn.Linear(in_channels, hidden_dim, bias=False)
        self.wo = nn.Linear(hidden_dim, in_channels, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu = self.wi_0(x)
        gelu = F.gelu(gelu)
        
        x = self.wi_1(x)
        x = x * gelu
        x = self.wo(x)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, in_channels: int, ff_dim: int):
        super().__init__()
        self.DenseReluDense = DenseGeluDense(in_channels, ff_dim) # Orignially named as DenseReluDense, going to have to rename the weights if we want to change the incorrect variable name
        self.layer_norm = SimplifiedLayerNorm(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.DenseReluDense(x)
        x = x + residual
        return x
    
class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ff_dim: int,
        num_heads: int,
        relative_attention_bias: bool,
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            SelfAttention(in_channels, num_heads, relative_attention_bias),
            FeedForward(in_channels, ff_dim),
        ])
        
    # Implemented it very weird in the original code. I might refactor the weights in the future
    def forward(self, x: torch.Tensor, past_bias=None):
        x, past_bias = self.layer[0](x, past_bias)
        x = self.layer[1](x)
        return x, past_bias
    
class T5(nn.Module):
    def __init__(self): # Preset values for T5XXL
        super().__init__()
        self.ff_dim = 10240
        self.model_dim = 4096
        self.num_heads = 64
        self.num_layers = 24
        self.vocab_size = 32128

        self.encoder.embed_tokens = nn.Embedding(self.vocab_size, self.model_dim)
        self.encoder.block = nn.ModuleList([
            Block(
                self.model_dim,
                self.ff_dim,
                self.num_heads,
                relative_attention_bias=(i == 0)
            )
        ] for i in range(self.num_layers))
        self.encoder.final_layer_norm = SimplifiedLayerNorm(self.model_dim)
        
    def get_input_embeddings(self):
        return self.encoder.embed_tokens
    
    def set_input_embeddings(self, embeddings):
        self.encoder.embed_tokens = embeddings
        
    def forward(self, tokens, intermediate_output=None, final_layer_norm_intermediate=True):
        intermediate = None
        x = self.encoder.embed_tokens(tokens)
        past_bias = None
        for i, layer in enumerate(self.block):
            x, past_bias = layer(x, past_bias)
            if i == intermediate_output:
                intermediate = x.clone()
        
        x = self.encoder.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        
        return x, intermediate
    
class T5Tokenizer(Tokenizer):
    def __init__(self):
        super().__init__(
            tokenizer=T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl"),
            max_length=99999999,
            pad_with_end=False,
            has_start_token=False,
            pad_to_max_length=False,
            min_length=77,
        )
        
class T5Model(ClipModel):
    """Wrapper for the T5 model to fit the ClipModel interface"""
    
    def __init__(
        self,
        config,
        layer="last",
        layer_idx=None,
    ):
        super().__init__(
            layer=layer,
            layer_idx=layer_idx,
            # textmodel_json_config=config,
            special_tokens={"end": 1, "pad": 0},
            model_class=T5,
        )