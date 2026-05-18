import re
import math
from torch import nn
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SELayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}



#也就说我们可以在这里强制让视觉编码器，直接输出IdentityMap，特征向量
def build_vision_projector(config, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')
    input_dim = config.mm_hidden_size  # 现在这是同步后的，也就是视觉编码器的维度
    output_dim = config.hidden_size    # LLM 的维度，如 2048 或 4096
    if projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(input_dim, output_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(output_dim, output_dim))
            return nn.Sequential(*modules)
    else:
        return IdentityMap()



