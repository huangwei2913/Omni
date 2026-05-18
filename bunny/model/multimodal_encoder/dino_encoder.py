import torch
import torch.nn.functional as F
from modelscope import AutoConfig, AutoImageProcessor
from typing import Union, List, Tuple
import sys
import os
from .base_encoder import BaseVisionTower
from bunny.util.merge import bipartite_soft_matching_merge
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16
from safetensors.torch import load_file  
import math

DINOv3_MODEL_FACTORIES = {
    "dinounet_s": dinov3_vits16,
    "dinounet_b": dinov3_vitb16,
    "dinounet_l": dinov3_vitl16,
    "dinounet_7b": dinov3_vit7b16,
}
import os
import torch
from safetensors.torch import load_file

def load_dinov3_model(model_name, pretrained_path):
    # 1. 架构准备
    factory = DINOv3_MODEL_FACTORIES.get(model_name)
    if factory is None:
        raise ValueError(f"Unknown DINOv3 model: {model_name}")
    model = factory(pretrained=False)

    # 2. 路径/文件名自动补全逻辑 (融合您的经验)
    st_path = None
    # 如果传入的是目录，自动拼接 model.safetensors
    if os.path.isdir(pretrained_path):
        potential_path = os.path.join(pretrained_path, "model.safetensors")
        if os.path.exists(potential_path):
            st_path = potential_path
    # 如果传入的已经是文件路径
    elif os.path.isfile(pretrained_path):
        st_path = pretrained_path

    # 3. 物理加载
    if st_path and st_path.endswith('.safetensors'):
        print(f"✅ Found weights at: {st_path}")
        sd = load_file(st_path)
    elif st_path: # 兼容旧版 .bin 或 .pth
        sd = torch.load(st_path, map_location='cpu')
    else:
        print(f"⚠️ Warning: Weights not found in {pretrained_path}, falling back to online weights.")
        return factory(pretrained=True)

    # 4. 执行映射逻辑 (HF -> Timm/Native) - 确保 173 个 Key 完美对齐
    new_sd = {}
    try:
        # 基础组件
        new_sd["cls_token"] = sd["embeddings.cls_token"]
        if "embeddings.mask_token" in sd:
            mask_token= sd["embeddings.mask_token"]
            if mask_token.dim() == 3 and mask_token.shape[1] == 1:
                mask_token = mask_token.squeeze(1) # [1, 1, 768] -> [1, 768]
            new_sd["mask_token"] = mask_token    
            
        new_sd["patch_embed.proj.weight"] = sd["embeddings.patch_embeddings.weight"]
        new_sd["patch_embed.proj.bias"] = sd["embeddings.patch_embeddings.bias"]
        new_sd["norm.weight"] = sd["norm.weight"]
        new_sd["norm.bias"] = sd["norm.bias"]
        
        # 逐层处理 (ViT-B 为 12 层)
        for i in range(12):
            prefix = f"layer.{i}."
            target = f"blocks.{i}."
            
            # Norms
            new_sd[f"{target}norm1.weight"] = sd[f"{prefix}norm1.weight"]
            new_sd[f"{target}norm1.bias"] = sd[f"{prefix}norm1.bias"]
            new_sd[f"{target}norm2.weight"] = sd[f"{prefix}norm2.weight"]
            new_sd[f"{target}norm2.bias"] = sd[f"{prefix}norm2.bias"]
            
            # MLP
            new_sd[f"{target}mlp.fc1.weight"] = sd[f"{prefix}mlp.up_proj.weight"]
            new_sd[f"{target}mlp.fc1.bias"] = sd[f"{prefix}mlp.up_proj.bias"]
            new_sd[f"{target}mlp.fc2.weight"] = sd[f"{prefix}mlp.down_proj.weight"]
            new_sd[f"{target}mlp.fc2.bias"] = sd[f"{prefix}mlp.down_proj.bias"]
            
            # LayerScale
            if f"{prefix}layer_scale1.lambda1" in sd:
                new_sd[f"{target}ls1.gamma"] = sd[f"{prefix}layer_scale1.lambda1"]
                new_sd[f"{target}ls2.gamma"] = sd[f"{prefix}layer_scale2.lambda1"]

            # QKV 拼接 (核心)
            qw = sd[f"{prefix}attention.q_proj.weight"]
            kw = sd[f"{prefix}attention.k_proj.weight"]
            vw = sd[f"{prefix}attention.v_proj.weight"]
            new_sd[f"{target}attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
            
            qb = sd[f"{prefix}attention.q_proj.bias"]
            kb = sd[f"{prefix}attention.k_proj.bias"] if f"{prefix}attention.k_proj.bias" in sd else torch.zeros_like(qb)
            vb = sd[f"{prefix}attention.v_proj.bias"]
            new_sd[f"{target}attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
            
            # Out Projection
            new_sd[f"{target}attn.proj.weight"] = sd[f"{prefix}attention.o_proj.weight"]
            new_sd[f"{target}attn.proj.bias"] = sd[f"{prefix}attention.o_proj.bias"]

        # 5. 加载转换后的权重
        model.load_state_dict(new_sd, strict=False)
        print("🚀 DINOv3 HuggingFace weights remapped and loaded successfully!")

    except Exception as e:
        print(f"❌ Error remapping DINOv3 weights: {e}")
        raise e

    return model


def load_manual_processor(config_path):
    import json
    from torchvision import transforms
    # 读取你 cat 出来的那个 json 文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 提取关键参数
    mean = config.get("image_mean", [0.485, 0.456, 0.406]) #
    std = config.get("image_std", [0.229, 0.224, 0.225])   #
    rescale = config.get("rescale_factor", 1/255.0)       #
    size = config.get("size", {"height": 384, "width": 384}) #

    # 构建 torchvision 流程
    # 注意：在 CoBunny 实验中，你之前测试 337*337 成功过，可以根据需求调整 size
    preprocess = transforms.Compose([
        transforms.Resize((size["height"], size["width"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return preprocess



import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from modelscope import AutoConfig, AutoImageProcessor
from .base_encoder import BaseVisionTower
from safetensors.torch import load_file

# 假设这些常量在外部定义或已导入
# DINOv3_MODEL_INFO, DINOv3_INTERACTION_INDEXES, load_dinov3_model 需确保在上下文中可用

class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args,**kwargs):
        super(DinoVisionTower, self).__init__(vision_tower, args,**kwargs)
        
        # 1. 基础参数定义
        self.vision_tower_name = vision_tower
        self._image_size = 384    # DinoV3 默认常用尺寸
        self._patch_size = 16 
        self._num_patches_cached = None 
        self.is_loaded  = False
        self.model_name = "dinounet_b"
        self.interaction_indexes = [2, 5, 8, 11]
        self.training_stage = kwargs.get('training_stage', getattr(args, 'training_stage', 'inference'))  
        print(f"🎨 [DinoVisionTower] 成功识别training_stage: {self.training_stage}")
        self.delay_load = kwargs.get('delay_load', False) 
        print(f"🎨 [DinoVisionTower] 成功识别delay_load: {self.delay_load}")   
        self.target_grid_size = getattr(args, "mm_vision_grid_size", 24)
        self.target_N = self.target_grid_size * self.target_grid_size
        self._hidden_size = 768  # 这里的 hidden_size 指 Backbone 输出维度
        self.target_embed_dim = self._hidden_size # 内部投影目标维度
        self.pretrained_path = getattr(args, "dinov3_pretrained_path", "/data/WorkSpace/models/dinov3-vitb16-pretrain-lvd1689m")

        # 2. Config 信息预加载（确保 delay_load 模式下属性依然可用）
        try:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)
        except Exception as e:
            from transformers import PretrainedConfig
            self.cfg_only = PretrainedConfig(hidden_size=self._hidden_size, image_size=self._image_size)

        if not self.delay_load:
            self.load_model()


    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"✅ [DinoVisionTower] 状态已锁定，无需重复加载，保护当前显存权重。")
            return
        # 2. 核心决策逻辑：搭架子还是装权重？
        if self.training_stage  in ["finetune", "inference"]:
            # 【搭架子模式】：微调和推理时，我们只需要物理架构
            # 权重会由 BunnyMetaModel 后续通过全量 Checkpoint 注入
            print(f"🏗️ [DINO 执行层] 身份: {self.training_stage } -> 模式: 仅构建物理架构 (Skeleton Only)")
            
            # 使用工厂函数直接创建架构，pretrained 设为 False
            model_factory = DINOv3_MODEL_FACTORIES[self.model_name]
            self.vision_tower = model_factory(pretrained=False)
        else:  #第一个阶段
            # 【官方加载模式】：预训练第一阶段（Stage 1）
            # 此时没有全量 Checkpoint，必须加载官方原始权重
            print(f"🛒 [DINO 执行层] 身份: {self.training_stage} -> 模式: 加载官方预训练权重 {self.pretrained_path}")
            self.vision_tower = load_dinov3_model(self.model_name, self.pretrained_path)
        
        self.vision_tower.to(device=self.device, dtype=torch.float16)
  
        CONFIG_FILE = os.path.join(self.pretrained_path, 'preprocessor_config.json')
        self.image_processor = load_manual_processor(CONFIG_FILE)
        #self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.is_loaded = True
        print(f"✅ [DinoVisionTower] 装载任务执行完毕。")
       
    def _forward(self, images):
        # 确保输入精度一致
        images = images.to(device=self.device, dtype=self.dtype)
        # 获取 4 层中间层特征 (List of Tuple: (feat, cls))
        all_layers = self.vision_tower.get_intermediate_layers(
            images, n=self.interaction_indexes, return_class_token=True
        )
        aligned_layers = []
        for feat, cls in all_layers:

            feat = feat.to(images.dtype)
            cls = cls.to(images.dtype)

            # 3. 空间特征对齐 [B, C, H, W] -> [B, T, C]
            if feat.dim() == 4:
                B, C, H, W = feat.shape
                feat = feat.view(B, C, H * W).permute(0, 2, 1)
   
            # 空间插值对齐到 target_N (如 24x24=576)
            if feat.shape[1] != self.target_N:
                # [B, T, C] -> [B, C, T] -> [B, C, target_N] -> [B, target_N, C]
                B, T, C = feat.shape
                hw = int(math.sqrt(T)) # 算出原始的宽高，比如 24
                target_hw = int(math.sqrt(self.target_N)) # 目标宽高，比如 24
                feat = feat.view(B, hw, hw, C).permute(0, 3, 1, 2)
                feat = F.interpolate(
                    feat, 
                    size=(target_hw, target_hw), 
                    mode="bilinear", 
                    align_corners=False
                )
                feat = feat.permute(0, 2, 3, 1).view(B, -1, C).contiguous()

            # 5. 直接拼接真正的 CLS Token
            # [B, 1, C] + [B, target_N, C] -> [B, 577, C]
            cls_tokens = cls.unsqueeze(1)
            feat_with_cls = torch.cat([cls_tokens, feat], dim=1)
            aligned_layers.append(feat_with_cls)
                
        # 拼接所有选定层的特征
        all_intermediate_features = torch.cat(aligned_layers, dim=1)
        
        # 更新缓存的 Patch 数量 (不含 CLS)
        if self._num_patches_cached is None:
            self._num_patches_cached = self.target_N

        #返回的影像张量是 [B*6, 3, 378, 378] all_intermediate_features的张量是[B,577*N, C]
        #注意这里的N是层数，C
        return images, all_intermediate_features

    # --- 必须保留的核心属性 ---
    
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