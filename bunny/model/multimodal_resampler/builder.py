import torch
from .fovea_sampler import FoveaIntentResampler
class IdentityMap(torch.nn.Module):
    def __init__(self, *args, **kwargs): # 加上这个，吃掉所有传进来的指令
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}

def build_vision_resampler(model_args, **kwargs):
    resampler_type = getattr(model_args, 'mm_resampler_type', None)
    if resampler_type is None:
        return IdentityMap(**kwargs)     
    else:
        return FoveaIntentResampler(model_args,**kwargs)
