import torch
from torch import nn
from torch.nn import functional as F

# Referenced Papers:
# [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.nin_shortcut = nn.Identity()
        else:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        self.swish = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.swish(x)
        x = self.conv2(x)

        return x + self.nin_shortcut(residual)
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init()
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.norm(x)

            q = self.q(x)
            k = self.k(x)
            v = self.v(x)

            b, c, h, w = q.shape
            interim_shape = (b, c, h * w)
            q = q.view(interim_shape).transpose(-1, -2).unsqueeze(1)
            k = k.view(interim_shape).transpose(-1, -2).unsqueeze(1)
            v = v.view(interim_shape).transpose(-1, -2).unsqueeze(1)

            # Original code
            # q, k, v = map(
            #     lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(),
            #     (q, k, v),
            # )

            x = F.scaled_dot_product_attention(q, k, v)
            x = x.squeeze(dim=1).transpose(-1, -2)
            x = x.view(b, c, h, w)
            x = self.proj_out(x)

            return x + residual

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, weight)
        
        pad = (0, 1, 0, 1) # (left, right, up, down)
        x = F.pad(x, pad, mode="constant", value=0) # (batch_size, channels, height, width) -> (batch_size, channels, height / 2, width / 2)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels : int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class Encoder(nn.Module): # in_channels = 3, out_channels = 2 * 16
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.ch_mult=(1, 2, 4, 4)
        self.in_ch_mult = (1,) + tuple(self.ch_mult)
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = 2

        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1) # (batch_size, 3, height, width) -> (batch-size, 128, height, width)
        self.down = nn.ModuleList()
        for resolution in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            block_in_channels = 128 * self.in_ch_mult[resolution] # (1, 1, 2, 4, 4) * 128
            block_out_channels = 128 * self.ch_mult[resolution] # (1, 2, 4, 4) * 128

            for b in range(self.num_res_blocks): 
                block.append(ResidualBlock(block_in_channels, block_out_channels))
                block_in_channels = block_out_channels
            down = nn.Module()
            down.block = block
            if resolution != self.num_resolutions - 1:
                down.downsample = Downsample(block_in_channels)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResidualBlock(block_in_channels, block_in_channels)
        self.mid.attn_1 = AttentionBlock(block_in_channels)
        self.mid.block_2 = ResidualBlock(block_in_channels, block_in_channels)
        
        self.norm_out = nn.GroupNorm(32, block_in_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(in_channels=block_in_channels, out_channels=(2 * out_channels), kernel_size=3, padding=1)

        self.swish = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, 3, height, width)

        # Downsampling
        x = self.conv_in(x) # (batch_size, 3, height, width) -> [(batch_size, 128, height, width)]
        
        for resolution in range(self.num_resolutions):
            for block in range(self.num_res_blocks):
                x = self.down[resolution].block[block](x)
            if resolution != self.num_resolutions - 1:
                x = self.down[resolution].downsample(x)
        
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        x = self.norm_out(x)
        x = self.swish(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module): # in_channels = 16, out_channels = 3
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.ch_mult += (1, 2, 4, 4)
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = 2
        self.resolution = 256
        
        block_in_channels = 128 * self.ch_mult[self.num_resolutions - 1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(in_channels, block_in_channels, kernel_size=3, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResidualBlock(block_in_channels, block_in_channels)
        self.mid.attn_1 = AttentionBlock(block_in_channels)
        self.mid.block_2 = ResidualBlock(block_in_channels, block_in_channels)

        self.up = nn.ModuleList()
        for resolution in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out_channels = 128 * self.ch_mult[resolution]
            
            for b in range(self.num_res_blocks + 1):
                block.append(ResidualBlock(block_in_channels, block_out_channels))
                block_in_channels = block_out_channels
            up = nn.Module()
            up.block = up

            if resolution != 0:
                up.upsample = Upsample(block_in_channels)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        
        self.norm_out = nn.GroupNorm(32, block_in_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in_channels, out_channels, kernel_size=3, padding=1)
        self.swish = nn.SiLU(inplace=True)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        for resolution in reversed(range(self.num_resolutions)):
            for block in range(self.num_res_blocks + 1):
                x = self.up[resolution].block[block](x)
            if resolution != 0:
                x = self.up[resolution].upsample(x)

        x = self.norm_out(x)
        x = self.swish(x)
        x = self.conv_out(x)

        return x
    
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(3, 16)
        self.decoder = Decoder(16, 3)

    @torch.autocast("cuda", dtype=torch.float16)
    def decode(self, latent):
        return self.decoder(latent)

    @torch.autocast("cuda", dtype=torch.float16)
    def encode(self, image):
        # image: (batch_size, 3, height, width)
        
        x = self.encoder(image)
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        var = torch.exp(log_var)
        st_dev = torch.sqrt(var)

        return mean + st_dev * torch.randn_like(mean)

