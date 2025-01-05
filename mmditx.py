import math

import torch
from typing import Dict, List, Optional, Any
from torch import nn
from torch.nn import functional as F
from einops import repeat, rearrange

from clip import MLP
from utils import modulate

class PatchEmbedder(nn.Module): # Previously implemented as PatchEmbed
    def __init__(
        self,
        # img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        # strict_img_size: bool = True,
        # dynamic_img_pad: bool = False,
    ):
        super().__init__()
        # self.patch_size = patch_size
        
        # self.patch_size = (patch_size, patch_size)
        # if img_size is not None:
        #     self.img_size = (img_size, img_size)
        #     self.grid_size = tuple(
        #         [s // p for s, p in zip(self.img_size, self.patch_size)]
        #     )
        #     self.num_patches = self.grid_size[0] * self.grid_size[1]
        # else:
        #     self.img_size = None
        #     self.grid_size = None
        #     self.num_patches = None
        
        # self.strict_img_size = strict_img_size
        # self.dynamic_img_pad = dynamic_img_pad
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        
        return x
    
class TimestepEmbedder(nn.Module): 
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        linear_1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        activation = nn.SiLU()
        linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.mlp = nn.Sequential(linear_1, activation, linear_2)
        self.frequency_embedding_size = frequency_embedding_size
        
    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, out_channels: int, max_period=10000):
        half = out_channels // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32)
            / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if out_channels % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        if torch.is_floating_point(timesteps):
            embedding = embedding.to(timesteps.dtype)
            
        return embedding
    
    def forward(self, timesteps):
        frequencies = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        embeddings = self.mlp(frequencies)
        
        return embeddings
    
class VectorEmbedder(nn.Module):
    """ Embeds a flat vector of dimension input_dim """
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        
        linear_1 = nn.Linear(in_channels, hidden_channels, bias=True)
        activation = nn.SiLU()
        linear_2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        
        self.mlp = nn.Sequential(linear_1, activation, linear_2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
class RMSNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter("weight", None)
            
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._norm(x)
        
        if self.learnable_scale:
            x = x * self.weight.to(x.device, x.dtype)

        return x
    
# class SwiGLUFeedForward(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: int,
#         multiple_of: int, # To ensure that the hidden channels are multiples of multiple_of
#         feed_forward_multiplier: int,
#     ):
#         super().__init__()
#         hidden_channels = int(2 * hidden_channels / 3)
        
#         if feed_forward_multiplier is not None:
#             hidden_channels = int(feed_forward_multiplier * hidden_channels)
            
#         hidden_channels = multiple_of * ((hidden_channels + multiple_of - 1) // multiple_of)
        
#         self.w1 = nn.Linear(in_channels, hidden_channels, bias=False)
#         self.w2 = nn.Linear(hidden_channels, in_channels, bias=False)
#         self.w3 = nn.Linear(in_channels, hidden_channels, bias=False)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x1 = self.w1(x)
#         x1 = F.silu(x1)
        
#         x2 = x1 * self.w3(x)
#         x2 = self.w2(x2)
        
#         return x2
    
class SelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int,
            qkv_bias: bool = True,
            pre_only: bool = False,
            qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.pre_only = pre_only

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        if not pre_only:
            self.proj = nn.Linear(in_channels, in_channels)

        if qk_norm == "rms":
            self.ln_q = RMSNorm(in_channels, elementwise_affine=True)
            self.ln_k = RMSNorm(in_channels, elementwise_affine=True)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(in_channels, elementwise_affine=True)
            self.ln_k = nn.LayerNorm(in_channels, elementwise_affine=True)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()

    def pre_attention(self, x : torch.Tensor):
        qkv = self.qkv(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, self.head_dim)
        qkv = qkv.movedim(2, 0)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.pre_attention(x)

        batch_size = q.shape[0]

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.in_channels)

        if not self.pre_only:
            out = self.post_attention(out)

        return out
        
class DiTBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_ratio: int = 4,
            qkv_bias: bool = False,
            pre_only: bool = False,
            scale_mod_only: bool = False,
            swiglu: bool = False,
            qk_norm: Optional[str] = None,
            x_block_self_attn: bool = False,
    ):
        
        self.x_block_self_attn = x_block_self_attn
        self.scale_mod_only = scale_mod_only
        self.pre_only = pre_only


        self.norm1 = RMSNorm(hidden_size, elementwise_affine=False)
        self.attn = SelfAttention(hidden_size, num_heads, qkv_bias, pre_only, qk_norm)

        if self.x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only

            self.attn2 = SelfAttention(hidden_size, num_heads, qkv_bias, False, qk_norm)

        if not pre_only:
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            self.norm2 = RMSNorm(hidden_size, elementwise_affine=False)
            self.mlp = MLP(hidden_size, hidden_size, mlp_hidden_dim, nn.GELU(approximate="tanh"))
            # if not swiglu:
            #     self.mlp = MLP(hidden_size, hidden_size, mlp_hidden_dim, nn.GELU(approximate="tanh"))
            # else:
            #     self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim, multiple_of=256)
        
        if self.x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, n_mods * hidden_size, bias=True),
        )

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        secondary = None

        assert x is not None, "pre_attention called with empty X"
        if self.pre_only:
            if self.scale_mod_only:
                shift_msa, scale_msa = None, self.adaLN_modulation(c)
            else:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
        else:
            if self.scale_mod_only:
                shift_msa, shift_mlp = None, None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
                secondary = (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                secondary = (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

        qkv = x 
        qkv = self.norm1(qkv)
        qkv = modulate(qkv, shift_msa, scale_msa)
        qkv = self.attn.pre_attention(qkv)

        return qkv, secondary or None
    
    def pre_attention_x(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert self.x_block_self_attn
        residual = x
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = self.adaLN_modulation(c).chunk(9, dim=1)

        x = self.norm1(x)
        x_mod1 = modulate(x, shift_msa, scale_msa)
        x_mod2 = modulate(x, shift_msa2, scale_msa2)
        qkv1 = self.attn.pre_attention(x_mod1)
        qkv2 = self.attn2.pre_attention(x_mod2)

        return qkv1, qkv2, (residual, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2)
    
    def post_attention(self, attn: torch.Tensor, x: torch.Tensor, gate_msa: torch.Tensor, shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor) -> torch.Tensor:
        # assert not self.pre_only # A bit redundant
        attn = self.attn.post_attention(attn)
        attn = attn * gate_msa.unsqueeze(1)
        attn += x

        x = attn
        attn = self.norm2(attn)
        attn = modulate(attn, shift_mlp, scale_mlp)
        attn = self.mlp(attn)
        attn = attn * gate_mlp.unsqueeze(1)
        attn += x
        
        return attn

    def post_attention_x(
            self,
            attn1: torch.Tensor,
            attn2: torch.Tensor,
            x: torch.Tensor,
            gate_msa: torch.Tensor,
            shift_mlp: torch.Tensor,
            scale_mlp: torch.Tensor,
            gate_mlp: torch.Tensor,
            gate_msa2: torch.Tensor,
            attn1_dropout: float = 0.0,
    ):
        # assert not self.pre_only
        attn1 = self.attn.post_attention(attn1)
        attn1 = gate_msa.unsqueeze(1) * attn1

        if attn1_dropout > 0.0:
            dropout = torch.bernoulli(
                torch.full((attn1.size(0), 1, 1), 1 - attn1_dropout, device=attn1.device)
            )
            attn1 = attn1 * dropout

        x += attn1

        attn2 = self.attn2.post_attention(attn2)
        attn2 = gate_msa2.unsqueeze(1) * attn2

        x += attn2
        residual = x

        x = self.norm2(x)   
        x = modulate(x, shift_mlp, scale_mlp)
        x = self.mlp(x)
        x = gate_mlp.unsqueeze(1) * x

        x += residual

        return x
    
    
    def attention(q, k, v, heads, mask=None): # Fuck it
        """Convenience wrapper around a basic attention operation"""
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only

        if self.x_block_self_attn:
            (q1, k1, v1), (q2, k2, v2), intermediates = self.pre_attention_x(x, c)
            attn1 = self.attention(q1, k1, v1, self.attn.num_heads)
            attn2 = self.attention(q2, k2, v2, self.attn2.num_heads)
            return self.post_attention_x(attn1, attn2, *intermediates)
        else:
            (q, k, v), intermediates = self.pre_attention(x, c)
            attn = self.attention(q, k, v, self.attn.num_heads)
            return self.post_attention(attn)
        
def block_mixing(context: torch.Tensor, x: torch.Tensor, context_block: DiTBlock, x_block: DiTBlock, c: torch.Tensor):
    assert context is not None, "block_mixing called with context set to None"
    context_qkv, context_intermediates = context_block.pre_attention(context, c)

    if x_block.x_block_self_attn:
        x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
    else:
        x_qkv, x_intermediates = x_block.pre_attention(x, c)

    q, k, v = tuple(
        torch.cat(tuple(qkv[i] for qkv in [context_qkv, x_qkv]), dim=1)
        for i in range(3)
    )

    attn = x_block.attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = (
        attn[:, : context_qkv[0].shape[1]],
        attn[:, context_qkv[0].shape[1] :],
    )

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None

    if x_block.x_block_self_attn:
        x_q2, x_k2, x_v2 = x_qkv2
        attn2 = x_block.attention(x_q2, x_k2, x_v2, x_block.attn2.num_heads)
        x = x_block.post_attention_x(x_attn, attn2, x, *x_intermediates)
    else:
        x = x_block.post_attention(x_attn, *x_intermediates)

    return context, x

class JointBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        qk_norm = kwargs.pop("qk_norm", None)
        x_block_self_attn = kwargs.pop("x_block_self_attn", False)

        self.context_block = DiTBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs) 
        self.x_block = DiTBlock(*args, pre_only=False, qk_norm=qk_norm, x_block_self_attn=x_block_self_attn, **kwargs)

        def forward(self, *args, **kwargs):
            return block_mixing(*args, self.context_block, self.x_block, **kwargs)

class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        total_out_channels: Optional[int] = None,
    ):
        super().__init__()
        
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = (
            nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
            ) if total_out_channels is None else nn.Linear(
                hidden_size,
                total_out_channels,
                bias=True,
            )
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)

        return x

class MMDiTX(nn.Module):
    def __init__(
        self,
        patch_size: int = 2, # Ok
        in_channels: int = 4, # Ok
        depth: int = 28, # Ok
        mlp_ratio: float = 4.0, 
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None, # Ok
        context_embedder_config: Optional[Dict] = None, # Ok
        register_length: int = 0,
        rmsnorm: bool = False, 
        scale_mod_only: bool = False,
        swiglu: bool = False, 
        out_channels: Optional[int] = None,
        pos_embed_max_size: Optional[int] = None, # Ok
        num_patches=None, # Ok
        qk_norm: Optional[str] = None, # Ok
        x_block_self_attn_layers: Optional[List[int]] = [], # Ok
        qkv_bias: bool = True,
        dtype = None,
    ):
        super().__init__()

        hidden_size = 64 * depth
        num_heads = depth

        self.register_length = register_length
        self.context_embedder = nn.Identity()
        self.x_embedder = PatchEmbedder(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size)

        if context_embedder_config is not None:
            if context_embedder_config["target"] == "torch.nn.Linear":
                self.context_embedder = nn.Linear(**context_embedder_config["params"], dtype=dtype)

        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size, dtype=dtype))

        if num_patches is not None:
            self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size, dtype=dtype))
        else:
            self.pos_embed = None
        
        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    pre_only = i == depth - 1,
                    scale_mod_only=scale_mod_only,
                    swiglu=swiglu,
                    qk_norm=qk_norm,
                    x_block_self_attn= (i in self.x_block_self_attn_layers),
                ) for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
    
    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        # patched size
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed
    
    def unpatchify(self, x, hw=None):
        patch_size = self.x_embedder.patch_size[0]
        channels = self.out_channels
        
        if hw is None:
            height = width = int(x.shape[1] ** 0.5)
        else:
            height, width = hw
            height = height // patch_size
            width = width // patch_size
        assert height * width == x.shape[1]

        x = x.reshape(shape=(x.shape[0], height, width, patch_size, patch_size, channels))
        x = torch.einsum("bhwpqc->bchpwq", x)
        images = x.reshape(shape=(x.shape[0], channels, height * patch_size, width * patch_size))

        return images
    
    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            controlnet_hidden_states: Optional[torch.Tensor] = None,
            skip_layers: Optional[List] = [],
    ) -> torch.Tensor:
        hw = x.shape[-2:]
        pos_embed = self.cropped_pos_embed(hw)
        x = self.x_embedder(x) + pos_embed
        c= self.t_embedder(t).to(x.dtype)

        if y is not None:
            y = self.y_embedder(y)
            c = c + y

        context = self.context_embedder(context)
        
        if self.register_length > 0:
            context = torch.cat(
                (
                    repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                    context if context is not None else torch.Tensor([]).type_as(x),
                ),
                1
            )

        for i, block in enumerate(self.joint_blocks):
            if i in skip_layers:
                continue
            context, x = block(context, x, c=c)    
            if controlnet_hidden_states is not None:
                controlnet_block_interval = len(self.joint_blocks) // len(controlnet_hidden_states)
            
            x = x + controlnet_hidden_states[i // controlnet_block_interval]

        x = self.final_layer(x, c)
        x = self.unpatchify(x, hw)

        return x