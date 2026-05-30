import torch
import torch.nn as nn
import torch.nn.functional as F
from .dino_encoder import DinoVisionTower
from .trocr_encoder import TrOCRVisionTower  # 引入你写的 TrOCR 塔
from bunny.util.utils import CrossAttentionBlock
from PIL import Image
from typing import List
import math

class SharedFeatureFusionAligner(nn.Module):
    def __init__(self, dim=768, intermediate_dim=1024): # 1024 或者是 1536
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        
        # 🌟 闭眼享受你提出的“升维 -> 降维”优雅沙漏架构
        self.mlp = nn.Sequential(
            nn.Linear(dim, intermediate_dim, bias=False),
            nn.GELU(),
            nn.Linear(intermediate_dim, dim, bias=False)
        )
        
    def forward(self, x):
        # x 形状: [B*6, Tokens, 768]
        # 采用最稳健的 Pre-LN 残差流形设计：x + MLP(LN(x))
        # 保证初始化时几乎是恒等映射，随着训练推进，隐式多任务梯度会完美注入到 MLP 的权重中
        return x + self.mlp(self.ln(x))


class ImageProcessorMultipleEncoders:
    def __init__(self, target_size: int = 384):
        # 统一到 384，因为 TrOCR 和 DinoV3 在 384 下表现最稳
        self.target_size = 384 
        self.dino_transform = None
        self.trocr_transform = None

    def preprocess(self, images, **kwargs):
        if not isinstance(images, list): images = [images]
        if self.dino_transform is None:
            from torchvision import transforms
            # DINO: ImageNet 归一化
            self.dino_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            # TrOCR: 0.5 均值归一化 (ViT 标准)
            self.trocr_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])

        stacked = []
        for img in images:
            if not isinstance(img, Image.Image): 
                img = Image.open(img).convert('RGB')
            img_res = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            
            # Index 0: Dino, Index 1: TrOCR
            dual_tower_tensor = torch.stack([
                self.dino_transform(img_res), 
                self.trocr_transform(img_res)
            ], dim=0)
            stacked.append(dual_tower_tensor)
        return {"pixel_values": torch.stack(stacked).contiguous()}

class AdaptiveConcatenationVisionTower(nn.Module):
    def __init__(self, vision_tower, args, **kwargs):
        super().__init__()

        self.is_loaded = False
        self.training_stage = kwargs.get('training_stage', getattr(args, 'training_stage', 'inference'))  
        print(f"🎨 [MixedEncoder] 成功识别training_stage: {self.training_stage}")
        self.delay_load = kwargs.get('delay_load', False) 
        print(f"🎨 [MixedEncoder] 成功识别delay_load: {self.delay_load}")    
        self.args = args
        # 1. 加载双塔
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.dino_vision_tower = DinoVisionTower(args.vision_tower_dino, args, **kwargs)
        self.trocr_vision_tower = TrOCRVisionTower(args.vision_tower_trocr, args, **kwargs)
        self.image_processor = ImageProcessorMultipleEncoders(target_size=384)
        #self.current_raw_images = None  #为了后面的重构损失
        #self.combined_features =None #为了后面的重构损失
        self._hidden_size = 768  #明确输出这个塔
        self.shared_aligner = SharedFeatureFusionAligner(dim=768, intermediate_dim=1024) # 选项：1024 或 1536
        if not kwargs.get('delay_load', False):
            self.load_model()

    def forward(self, images):
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = 0 # 如果不是分布式训练，默认为 0
        # 只有 Rank 0 负责“发声”
        if rank == 0:
            if not hasattr(self, "has_printed_shape"):
                print("\n" + "👁️" * 15 + " RANK 0 独家质检 " + "👁️" * 15)
                print(f"🚀 [VISION TOWER ENTRY]")
                print(f"   - Images Shape: {images.shape}")
                # 顺便检查一下数据在哪张卡上
                print(f"   - Device: {images.device}") 
        # 检查一下数值范围，确保没有溢出或全零
                print(f"   - Mean Value: {images.mean().item():.4f}") 
                print("👁️" * 40 + "\n")
            self.has_printed_shape = True

        device = images.device
        b, num_crops, num_towers, c, h, w = images.shape  #[B, 6, 2, 3, 384, 384])
        # --- 1. 双塔特征提取 ---
        dino_input = images[:, :, 0].view(-1, c, h, w)
        trocr_input = images[:, :, 1].view(-1, c, h, w)

        # 获取中间层特征 [B*6, Layers * 577, 768]
        _, dino_gallery = self.dino_vision_tower(dino_input) #这个返回的是torch.Size([B*6, 2308, 768])
        _, trocr_gallery = self.trocr_vision_tower(trocr_input) #这个返回的是torch.Size([B*6, 2308, 768])
        combined_per_crop = torch.cat([dino_gallery, trocr_gallery], dim=1)  #这个返回的是torch.Size([B*6, 2308*2, 768])
        
        #self.current_raw_images = images  # 这个用于解码器的重构损失计算，形状是[B, 6, 2, 3, 384, 384])
        current_raw_images = images[:, :, 0].view(-1, c, h, w) #这个用于解码器的重构损失计算,可以只选dino的部分 [B*6, 3, 384, 384] 
        shared_features = self.shared_aligner(combined_per_crop)
        self.combined_features = shared_features  #torch.Size([B*6, 2308*2, 768])
        return shared_features,(current_raw_images, shared_features)  #返回的是 torch.Size([B*6, 2308*2, 768])

    def load_model(self):
        if self.is_loaded:
            return
        self.dino_vision_tower.load_model()
        self.trocr_vision_tower.load_model()
        self._set_subtower_grad_state()
        self.is_loaded = True

    def _set_subtower_grad_state(self):

        mode_desc = "🚀 [全量微调/全参数模式]" if self.unfreeze_mm_vision_tower else "🔒 [冻结模式/只读推理模式]"
        print(f"🛠️  [MixedEncoder 属性设定] 业务意图: {mode_desc}")
        is_actually_unfreezing = (self.training_stage == 'finetune') and self.unfreeze_mm_vision_tower
        for sub_tower in [self.trocr_vision_tower, self.dino_vision_tower]:
            if sub_tower is not None:
                # 注入属性
                if hasattr(sub_tower, 'config'):
                    sub_tower.config.unfreeze_mm_vision_tower = is_actually_unfreezing
                sub_tower.unfreeze_mm_vision_tower = is_actually_unfreezing
                # 获取名字，如果是 DINO 或 SigLIP 应该能看出来
                t_name = getattr(sub_tower, "vision_tower_name", "Sub-Tower")
                if is_actually_unfreezing:
                    sub_tower.requires_grad_(True)
                    sub_tower.train()
                    print(f"   💡 子塔 {t_name}: 已解锁权重。它将随主模型一起更新（微调必备）。")
                else:
                    sub_tower.requires_grad_(False)
                    sub_tower.eval()
                    print(f"   💡 子塔 {t_name}: 已锁定权重。它将作为纯特征提取器使用（预训练/推理必备）。")

    @property
    def hidden_size(self): 
        return self._hidden_size