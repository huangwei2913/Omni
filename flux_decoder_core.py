import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput # 尝试保留基础组件
# 如果上面依然报错，说明 diffusers 彻底不可用，我再给你写纯 nn.Module 版本

class FluxSmallDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 强制根据 FLUX.2 [klein] 的参数进行硬编码
        self.config = config
        # 这里的核心是直接调用 FLUX 专用的 Decoder 结构
        # 由于 0.27.2 没有这个类，我们需要手动构建
        from diffusers.models.autoencoders.autoencoder_kl import Decoder 
        
        self.decoder = Decoder(
            in_channels=config.get("latent_channels", 32),
            out_channels=config.get("out_channels", 3),
            up_block_types=config.get("decoder_up_block_types", ["UpDecoderBlock2D"] * 4),
            block_out_channels=config.get("decoder_block_out_channels", [96, 192, 384, 384]),
            layers_per_block=config.get("decoder_layers_per_block", 2),
            norm_num_groups=config.get("decoder_norm_num_groups", 32),
            act_fn=config.get("decoder_act_fn", "silu"),
        )

    def decode(self, z):
        # FLUX.2 可能会对 latent 进行缩放
        z = z / self.config.get("scaling_factor", 0.3611)
        dec = self.decoder(z)
        return dec