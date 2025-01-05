import torch
from torch import nn
from typing import Dict
from torch.nn import functional as F
from utils import Tokenizer

# CLIPG_CONFIG = {
#     "hidden_act": "gelu",
#     "hidden_size": 1280,
#     "intermediate_size": 5120,
#     "num_attention_heads": 20,
#     "num_hidden_layers": 32,
# }

# CLIPL_CONFIG = {
#     "hidden_act": "quick_gelu",
#     "hidden_size": 768,
#     "intermediate_size": 3072,
#     "num_attention_heads": 12,
#     "num_hidden_layers": 12,
# }

ACTIVATIONS = {
    'quick_gelu': lambda x: x * torch.sigmoid(1.702 * x),
    'gelu': F.gelu,
}

class SDXLClipGTokenizer(Tokenizer):
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)
    
class SDXLClipLTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(in_channels, in_channels, bias=True)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=True)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=True)
        self.out_proj = nn.Linear(in_channels, in_channels, bias=True)
        
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        batch_size, sequence_length, embed_dim = q.shape
        head_dim = embed_dim // self.num_heads
        
        q = q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k , v, attention_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(batch_size, -1, embed_dim)
        
        out = self.out_proj(out)
        return out
        
class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, act_func: str):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.act = act_func
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x
        
class Layer(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, hidden_size: int, act_func: None):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.self_attn = SelfAttention(in_channels, num_heads)
        self.layer_norm2 = nn.LayerNorm(in_channels)
        self.mlp = MLP(in_channels, in_channels, hidden_size, act_func)
        
    def forward(self, x: torch.Tensor, mask: None) -> torch.Tensor:
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x, mask=mask)
        x += residual
        
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x += residual
        
        return x
       
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        num_layers: int,
        hidden_size: int,
        act_func: None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(in_channels, num_heads, hidden_size, act_func) for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: None, intermediate_output=None) -> torch.Tensor:
        if intermediate_output is not None and intermediate_output < 0:
            intermediate_output += len(self.layers)
        
        intermediate = None
        for idx, layer in enumerate(self.layers):
            x = layer(x, mask)     
            if intermediate_output is not None and idx == intermediate_output:
                intermediate = x.clone()
                
        return x, intermediate
        
class ClipEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int = 49408, max_sequence_length: int = 77):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x += self.position_embedding.weight
        return x
    
class ClipTextModel(nn.Module): 
    """
    Clip model that defaults to clipL configuration
    """
    def __init__(self, config_dict: Dict):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"] 
        embed_dim = config_dict["hidden_size"]
        num_heads = config_dict["num_attention_heads"]
        hidden_size = config_dict["intermediate_size"]
        act_func = ACTIVATIONS[config_dict["hidden_act"]]
        
        self.text_model = nn.Module()
        self.text_model.embeddings = ClipEmbeddings(embed_dim)
        self.text_model.encoder = Encoder(
            in_channels=embed_dim,
            num_heads=num_heads,
            num_layers=self.num_layers,
            hidden_size=hidden_size,
            act_func=act_func,
        )
        self.text_model.final_layer_norm = nn.LayerNorm(embed_dim)
        
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        
    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding
    
    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings
    
    def forward(
        self,
        tokens,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
    ):
        x = self.text_model.embeddings(tokens)
        causal_mask = (torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
                       .fill_(float("-inf"))
                       .triu(1))
        
        x, intermediate = self.text_model.encoder(x, mask=causal_mask, intermediate_output=intermediate_output)
        x = self.text_model.final_layer_norm(x)

        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.text_model.final_layer_norm(intermediate)
            
        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),
        ]
        
        out = self.text_projection(pooled_output)
        return (x, intermediate, out, pooled_output)
    
class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        out, pooled = self([tokens])
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled
    
class ClipModel(nn.Module, ClipTokenWeightEncoder):
    """
    Generic interface for all our text encoders
    """
    
    LAYERS = ["last", "pooled", "hidden"]
    
    def __init__(
        self,
        max_length: int = 77,
        layer: str = "last",
        layer_idx = None,
        textmodel_json_config: Dict = None,
        model_class=ClipTextModel,
        special_tokens={"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state=True,
        return_projected_pool=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = layer_idx
        self.special_tokens = special_tokens
        self.logit_scale = nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pool = return_projected_pool
        
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
            
        self.options_default = (
            self.layer,
            self.layer_idx,
            self.return_projected_pool,
        )
        
    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pool = options.get("projected_pool", self.return_projected_pool)
        
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx
       
    def process_tokens(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        tokens = torch.LongTensor(tokens).to(backup_embeds.weight.device)
        outputs = self.transfomer(
            tokens,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        self.transformer.set_input_embeddings(backup_embeds)
        return outputs

    def extract_outputs(self, outputs):
        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]
            
    def extract_pooled_output(self, outputs):
        pooled_output = None
        if len(outputs) >= 3:
            if (
                not self.return_projected_pool
                and len(outputs) >= 4
                and outputs[3] is not None
            ):
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        return pooled_output
            
    def forward(self, tokens):
        outputs = self.process_tokens(tokens)
        x = self.extract_outputs(outputs).float()
        pooled_output = self.extract_pooled_output(outputs)
        
        return x, pooled_output
    
class SDXLClipGModel(ClipModel):
    """Wrapper for SDXLClipG to fit the ClipModel interface"""
    def __init__(
        self,
        config,
        layer="penultimate",
        layer_idx=None,
    ):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2
        
        super().__init__(
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=config,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
            layer_norm_hidden_state=False,
        )
        
