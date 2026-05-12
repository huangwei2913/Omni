import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Model, LayoutLMv3Config
from .base_encoder import BaseVisionTower, ProcessorWrapper
import math
import os

class LayoutLMv3VisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False, **kwargs):
        super(LayoutLMv3VisionTower, self).__init__(vision_tower, args, delay_load, **kwargs)
        
        # 1. 基础维度参数
        self._hidden_size = 768
        self._image_size = 224
        self._patch_size = 16
        
        # 2. 混合编码器对齐参数 (24x24 = 576)
        self.target_grid_size = getattr(args, "mm_vision_grid_size", 24)
        self.target_N = self.target_grid_size * self.target_grid_size
        
        # 3. 选取的交互层索引 (LayoutLMv3-base 共 12 层)
        # 建议选取 [2, 5, 8, 11] 以匹配你 DINOv3 的语义高度
        self.interaction_indexes = [2, 5, 8, 11]

        if not self.delay_load:
            self.load_model()
        else:
            # 延迟加载时，先构建一个虚拟 config 供基类属性使用
            self.cfg_only = LayoutLMv3Config(
                hidden_size=self._hidden_size, 
                image_size=self._image_size,
                patch_size=self._patch_size
            )

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        print(f"🛒 [LayoutLM 执行层] 身份: {self.training_stage} -> 加载: {self.vision_tower_name}")
        
        # 仅加载视觉部分，忽略文本和 OCR[cite: 1, 2]
        self.vision_tower = LayoutLMv3Model.from_pretrained(self.vision_tower_name)
        
        # 精度与设备对齐
        self.vision_tower.to(device=self.device, dtype=torch.float16)
        
        # 根据 BaseVisionTower 的 unfreeze 参数决定梯度
        if not self.unfreeze_mm_vision_tower:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
        else:
            self.vision_tower.train()

        self.is_loaded = True

    def _forward(self, images):
        """
        核心前向传播：提取多层特征并进行空间对齐[cite: 7]
        """
        # 确保输入精度
        images = images.to(device=self.device, dtype=self.dtype)
        
        # 获取所有隐藏层输出
        outputs = self.vision_tower(
            pixel_values=images, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        # 提取选定的 4 层特征[cite: 7]
        aligned_layers = []
        # hidden_states[0] 是 embedding 层，所以 1-12 才是 transformer 层
        for idx in self.interaction_indexes:
            # LayoutLMv3 索引 +1 对应 Transformer 层输出
            feat_full = outputs.hidden_states[idx + 1] # [B, 197, 768]
            
            # 分离 CLS 和 Patch
            cls_token = feat_full[:, 0:1, :]       # [B, 1, 768]
            patch_feat = feat_full[:, 1:, :]       # [B, 196, 768]
            
            # 空间插值：14x14 (196) -> 24x24 (576)[cite: 7]
            B, T, C = patch_feat.shape
            orig_hw = int(math.sqrt(T)) # 14
            target_hw = self.target_grid_size # 24
            
            # [B, T, C] -> [B, C, H, W]
            patch_feat = patch_feat.view(B, orig_hw, orig_hw, C).permute(0, 3, 1, 2)
            patch_feat = F.interpolate(
                patch_feat, 
                size=(target_hw, target_hw), 
                mode="bilinear", 
                align_corners=False
            )
            # [B, C, H, W] -> [B, T_target, C]
            patch_feat = patch_feat.permute(0, 2, 3, 1).view(B, -1, C).contiguous()
            
            # 重新拼合含 CLS 的序列 [B, 577, 768]
            feat_with_cls = torch.cat([cls_token, patch_feat], dim=1)
            aligned_layers.append(feat_with_cls)
                
        # 1. 最后一层作为“主输出”
        # 2. 所有 4 层拼接作为“画廊特征”供后续深度交互[cite: 7]
        all_intermediate_features = torch.cat(aligned_layers, dim=1) # [B, 4*577, 768]
        
        return aligned_layers[-1], all_intermediate_features

    @property
    def hidden_size(self):
        """
        返回 768。
        混合编码器会根据此值建立 MLP 映射：nn.Linear(768, 1024)。
        """
        return self._hidden_size

    @property
    def num_patches(self):
        """
        非常关键！必须返回插值对齐后的数量 576 (即 24*24)。
        基类默认通过 config 计算会得到 196 (14*14)，这会导致采样器维度崩溃[cite: 7, 8]。
        """
        return self.target_N

    @property
    def image_size(self):
        """返回 224[cite: 2]。"""
        return self._image_size

    @property
    def patch_size(self):
        """返回 16[cite: 2]。"""
        # LayoutLMv3 默认 Patch 大小为 16
        return self._patch_size

    @property
    def layer_count(self):
        """
        返回 4。
        这决定了混合塔在解析 'all_intermediate_features' 时如何切分 Tensor。
        """
        return len(self.interaction_indexes)

    @property
    def num_patches_per_side(self):
        """返回 24。"""
        return int(self.num_patches ** 0.5)

    @property
    def config(self):
        """
        优先返回物理模型的 config。
        如果处于 delay_load 阶段，则返回预设的 cfg_only，确保基类属性读取安全[cite: 8]。
        """
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only