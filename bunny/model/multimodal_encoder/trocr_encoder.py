import torch
import torch.nn.functional as F
from typing import Union, List, Tuple
import sys, os
from .base_encoder import BaseVisionTower
import math
from transformers import (
    TrOCRConfig,
    TrOCRProcessor,
    TrOCRForCausalLM,
    ViTConfig,
    ViTModel,
    VisionEncoderDecoderModel,
)

class TrOCRVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args,**kwargs):
        super(TrOCRVisionTower, self).__init__(vision_tower, args,**kwargs)
        # 1. 基础参数定义
        self.vision_tower_name = vision_tower
        self._image_size = 384    
        self._patch_size = 16 
        self._num_patches_cached = None 
        self.is_loaded  = False
        self.interaction_indexes = [2, 5, 8, 11]
        self.training_stage = kwargs.get('training_stage', getattr(args, 'training_stage', 'inference'))  
        print(f"🎨 [TrOCRVisionTower] 成功识别training_stage: {self.training_stage}")
        self.delay_load = kwargs.get('delay_load', False) 
        print(f"🎨 [TrOCRVisionTower] 成功识别delay_load: {self.delay_load}")   
        self.target_grid_size = getattr(args, "mm_vision_grid_size", 24)
        self.target_N = self.target_grid_size * self.target_grid_size
        self._hidden_size = 768  # 这里的 hidden_size 指 Backbone 输出维度
        self.target_embed_dim = self._hidden_size # 内部投影目标维度
        self.pretrained_path = getattr(args, "trocr_pretrained_path", "/data/WorkSpace/models/trocr-base-str")
        # 2. Config 信息预加载（确保 delay_load 模式下属性依然可用）
        self.cfg_only = TrOCRConfig()
        if not self.delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"✅ [TrOCRVisionTower] 状态已锁定，无需重复加载，保护当前显存权重。")
            return
        # 2. 核心决策逻辑：搭架子还是装权重？
        if self.training_stage  in ["finetune", "inference"]:
            # 【搭架子模式】：微调和推理时，我们只需要物理架构
            # 权重会由 BunnyMetaModel 后续通过全量 Checkpoint 注入
            print(f"🏗️ [TrOCRVisionTower 执行层] 身份: {self.training_stage } -> 模式: 仅构建物理架构 (Skeleton Only)")
            encoder = ViTModel(ViTConfig())
            decoder = TrOCRForCausalLM(TrOCRConfig())
            model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
            # 使用工厂函数直接创建架构，pretrained 设为 False
            self.vision_tower = encoder
        else:  #第一个阶段
            # 【官方加载模式】：预训练第一阶段（Stage 1）
            # 此时没有全量 Checkpoint，必须加载官方原始权重
            print(f"🛒 [TrOCRVisionTower 执行层] 身份: {self.training_stage} -> 模式: 加载官方预训练权重 {self.pretrained_path}")
            model = VisionEncoderDecoderModel.from_pretrained(self.pretrained_path)
            self.vision_tower = model.encoder
        
        self.vision_tower.to(device=self.device)
        processor = TrOCRProcessor.from_pretrained(self.pretrained_path)
        self.image_processor = processor
        self.is_loaded = True
        print(f"✅ [TrOCRVisionTower] 装载任务执行完毕。")
       
    def _forward(self, images):
        # 1. 确保输入精度和设备一致
        images = images.to(device=self.device, dtype=self.dtype)
        
        # 2. 调用模型并获取所有隐藏层
        # ViTModel 的输出对象中 hidden_states 是一个元组：(embedding_output, layer1, layer2, ...)
        outputs = self.vision_tower(images, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        aligned_layers = []
        
        # 计算当前的 grid size (基于输入图像计算，通常是 24)
        # B: Batch, L: Sequence Length, C: Hidden Size
        B, L, C = hidden_states[0].shape
        current_num_patches = L - 1 # 减去 CLS token
        current_grid_size = int(math.sqrt(current_num_patches))

        for idx in self.interaction_indexes:
            # 提取指定索引的层 [B, L, C]
            feat = hidden_states[idx] 
            
            # 分离 CLS token [B, 1, C] 和 Patch tokens [B, L-1, C]
            cls_token = feat[:, 0:1, :]
            patch_tokens = feat[:, 1:, :]
            
            # 3. 插值逻辑：如果当前 patch 数量不等于目标 target_N，进行空间缩放
            if patch_tokens.shape[1] != self.target_N:
                # [B, L-1, C] -> [B, C, H, W]
                # 先转置为卷积格式以进行插值
                patch_tokens = patch_tokens.transpose(1, 2).reshape(
                    B, C, current_grid_size, current_grid_size
                )
                
                # 执行双线性插值到 target_grid_size (例如 24x24)
                patch_tokens = F.interpolate(
                    patch_tokens.to(torch.float32), # 建议在 float32 下插值保证精度
                    size=(self.target_grid_size, self.target_grid_size),
                    mode='bilinear',
                    align_corners=False
                ).to(self.dtype)
                
                # [B, C, H_target, W_target] -> [B, target_N, C]
                patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
            
            # 4. 重新拼接 CLS 和处理后的 Patch
            combined = torch.cat([cls_token, patch_tokens], dim=1)
            aligned_layers.append(combined)

        # 5. 拼接所有选定的中间层
        # 结果维度: [B, (1 + target_N) * len(interaction_indexes), C]
        all_intermediate_features = torch.cat(aligned_layers, dim=1)

        # 更新缓存信息
        if self._num_patches_cached is None:
            self._num_patches_cached = self.target_N

        return images, all_intermediate_features

    @property
    def hidden_size(self):
        """返回 Backbone 的原始维度，混合编码器会根据此值建立 MLP 映射到 1024"""
        return self._hidden_size

    @property
    def num_patches(self):
        """返回插值对齐后的 Patch 数量 (不含 CLS)"""
        return self.target_N

    @property
    def image_size(self):
        return self._image_size

    @property
    def patch_size(self):
        # 优先从加载后的模型获取真实 patch_size
        return getattr(self.vision_tower, "patch_size", self._patch_size)

    @property
    def layer_count(self):
        """返回提取的中间层数量"""
        return len(self.interaction_indexes)

    @property
    def num_patches_per_side(self):
        return int(self.num_patches ** 0.5)

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only