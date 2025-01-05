from json import load
import torch
from safetensors import safe_open

from t5 import T5Model
from clip import ClipModel, SDXLClipGModel
from sd3 import BaseModel
from vae import VAE

def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e
            
class T5:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(
            f"{model_folder}/t5xxl.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = T5Model()
            load_into(f, self.model.transformer, "", device="cpu", dtype=torch.float32)

CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

class ClipL:
    def __init__(self, model_folder: str) -> None:
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = ClipModel(
                layer="hidden",
                layer_idx=-2,
                layer_norm_hidden_state=False,
                return_projected_pool=False,
                textmodel_json_config=CLIPL_CONFIG,
            ).to("cpu", torch.float32)

CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

class ClipG:
    def __init__(self, model_folder: str) -> None:
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipGModel(CLIPG_CONFIG).to("cpu", torch.float32)
            load_into(f, self.model.transformer, "", device="cpu", dtype=torch.float32)

class VAE:
    def __init__(self, model_folder: str):
        with safe_open(
            f"{model_folder}/vae.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = VAE().eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", torch.float32)

CONTROLNET_MAP = {
    "time_text_embed.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "pos_embed.proj.bias": "x_embedder.proj.bias",
    "pos_embed.proj.weight": "x_embedder.proj.weight",
    "time_text_embed.text_embedder.linear_1.bias": "y_embedder.mlp.0.bias",
    "time_text_embed.text_embedder.linear_1.weight": "y_embedder.mlp.0.weight",
    "time_text_embed.text_embedder.linear_2.bias": "y_embedder.mlp.2.bias",
    "time_text_embed.text_embedder.linear_2.weight": "y_embedder.mlp.2.weight",
}

class SD3:
    def __init__(
            self,
            model_folder: str,
            shift: float,
            control_model_file = None,
            device: str = "cpu",
    ):
        self.using_8b_controlnet = False

        with safe_open(
            f"{model_folder}/sd3.5_medium.safetensors", framework="pt", device=device
        ) as f:
            control_model_ckpt = None
            if control_model_file is not None:
                control_model_ckpt = safe_open(control_model_file, framework="pt", device=device)
            
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                control_model_ckpt=control_model_ckpt,
            ).eval().to("cpu", torch.float32)
            load_into(f, self.model, "model.", "cuda", torch.float16)

        if control_model_file is not None:
            control_model_ckpt = safe_open(control_model_file, framework="pt", device=device)
            self.model.control_model = self.model.control_model.to(device)
            load_into(control_model_ckpt, self.model.control_model, "", device, torch.float16, remap=CONTROLNET_MAP)

            self.using_8b_controlnet = self.model.control_model.y_embedder.mlp[0].in_features == 2048
            self.model.control_model.using_8b_controlnet = self.using_8b_controlnet

        control_model_ckpt = None