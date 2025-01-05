import torch
import math
import re

from torch import nn
from torch.nn import functional as F
from PIL import Image
from transformers import CLIPTokenizer

from clip import SDXLClipLTokenizer, SDXLClipGTokenizer
from t5 import T5Tokenizer
from mmditx import MMDiTX
from controlnet import ControlNetEmbedder

class DiscreteFlowSampling(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image
    
class BaseModel(nn.Module):
    """Wrapper around the core MM-DiT model"""
    def __init__(
            self,
            shift=1.0,
            file=None,
            prefix="",
            control_model_ckpt=None,
    ):
        super().__init__()
        patch_size = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[2]
        depth = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[0] // 64
        num_patches = file.get_tensor(f"{prefix}pos_embed").shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = file.get_tensor(f"{prefix}y_embedder.mlp.0.weight").shape[1]
        context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape
       
        qk_norm = (
            "rms"
            if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in file.keys()
            else None
        )

        x_block_self_attn_layers = sorted(
            [
                int(key.split(".x_block.attn2.ln_k.weight")[0].split(".")[-1])
                for key in list(
                    filter(
                        re.compile(".*.x_block.attn2.ln_k.weight").match, file.keys()
                    )
                )
            ]
        )

        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0],
            },
        }

        self.diffusion_model = MMDiTX(
            patch_size = patch_size,
            in_channels=16,
            depth=depth,
            adm_in_channels=adm_in_channels,
            context_embedder_config=context_embedder_config,
            pos_embed_max_size=pos_embed_max_size,
            num_patches=num_patches,
            qk_norm=qk_norm,
            x_block_self_attn_layers=x_block_self_attn_layers,
        ).to(device="cpu", dtype=torch.float32)

        self.model_sampling = DiscreteFlowSampling(shift=shift)
        self.control_model = None
        
        if control_model_ckpt is not None:
            n_controlnet_layers = len(
                list(
                    filter(
                        re.compile(".*.attn.proj.weight").match,
                        control_model_ckpt.keys(),
                    )
                )
            )

            hidden_size = depth * 64
            num_heads = depth
            head_dim = hidden_size // num_heads
            pooled_projection_size = control_model_ckpt.get_tensor('time_text_embed.text_embedder.linear_1.weight').shape[1]

            print(f"Initializing ControlNetEmbedder with {n_controlnet_layers} layers")

            self.control_model = ControlNetEmbedder(
                patch_size=patch_size,
                in_channels=16,
                num_layers=n_controlnet_layers,
                attention_head_dim=num_heads,
                pooled_projection_size=pooled_projection_size,
            )

    def apply_model(self, x, sigma, c_crossattn=None, y = None, skip_layers = [], controlnet_cond = None):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        controlnet_hidden_states = None

        if controlnet_cond is not None:
            y_cond = y.to(dtype)
            controlnet_cond = controlnet_cond.to(x.device, x.dtype)
            cotrolnet_cond = controlnet_cond.repeat(x.shape[0], 1, 1, 1)

            if not self.control_model.using_8b_controlnet:
                hw = x.shape[-2:]
                x_controlnet = self.diffusion_model.x_embedder(x) + self.diffusion_model.cropped_pos_embed(hw)
                controlnet_hidden_states = self.control_model(x_controlnet, controlnet_cond, y_cond, 1, sigma.to(torch.float32))

        model_output = self.diffusion_model(
            x.to(dtype),
            timestep,
            context=c_crossattn.to(dtype),
            controlnet_hidden_states=controlnet_hidden_states,
            skip_layers=skip_layers,
        ).float()

        return self.model_sampling.calculate_denoised(sigma, model_output, x)
    
    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)   
    
    def get_dtype(self):
        return self.diffusion_model.dtype
    
class CFGDenoiser(nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, *args):
        super().__init__()
        self.model = model
    
    def forward(
            self,
            x,
            timestep,
            cond,
            uncond,
            cond_scale,
            **kwargs
    ):
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            **kwargs
        )

        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled
    
class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, *args):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        **kwargs,
    ):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled


class SkipLayerCFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, steps, skip_layer_config):
        super().__init__()
        self.model = model
        self.steps = steps
        self.slg = skip_layer_config["scale"]
        self.skip_start = skip_layer_config["start"]
        self.skip_end = skip_layer_config["end"]
        self.skip_layers = skip_layer_config["layers"]
        self.step = 0

    def forward(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        **kwargs,
    ):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        # Then run with skip layer
        if (
            self.slg > 0
            and self.step > (self.skip_start * self.steps)
            and self.step < (self.skip_end * self.steps)
        ):
            skip_layer_out = self.model.apply_model(
                x,
                timestep,
                c_crossattn=cond["c_crossattn"],
                y=cond["y"],
                skip_layers=self.skip_layers,
            )
            # Then scale acc to skip layer guidance
            scaled = scaled + (pos_out - skip_layer_out) * self.slg
        self.step += 1
        return scaled

class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

    def decode_latent_to_preview(self, x0):
        """Quick RGB approximate preview of sd3 latents"""
        factors = torch.tensor(
            [
                [-0.0645, 0.0177, 0.1052],
                [0.0028, 0.0312, 0.0650],
                [0.1848, 0.0762, 0.0360],
                [0.0944, 0.0360, 0.0889],
                [0.0897, 0.0506, -0.0364],
                [-0.0020, 0.1203, 0.0284],
                [0.0855, 0.0118, 0.0283],
                [-0.0539, 0.0658, 0.1047],
                [-0.0057, 0.0116, 0.0700],
                [-0.0412, 0.0281, -0.0039],
                [0.1106, 0.1171, 0.1220],
                [-0.0248, 0.0682, -0.0481],
                [0.0815, 0.0846, 0.1207],
                [-0.0120, -0.0055, -0.0867],
                [-0.0749, -0.0634, -0.0456],
                [-0.1418, -0.1457, -0.1259],
            ],
            device="cpu",
        )
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

        latents_ubyte = (
            ((latent_image + 1) / 2)
            .clamp(0, 1)  # change scale from -1..1 to 0..1
            .mul(0xFF)  # to 0..255
            .byte()
        ).cpu()

        return Image.fromarray(latents_ubyte.numpy())
    
class SD3Tokenizer:
    def __init__(self):
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_l = SDXLClipLTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        self.t5xxl = T5Tokenizer()

    def tokenize_with_weights(self, text: str):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text)
        out["g"] = self.clip_g.tokenize_with_weights(text)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text[:226])
        return out