import torch
import torch.nn as nn
import torch.nn.functional as F

class FluxProjectorGrid(nn.Module):
    def __init__(self, input_dim=768, output_channels=32, target_grid=48):
        super().__init__()
        self.target_grid = target_grid  # 48
        
        # 🌟 彻底抛弃 Conv2d 和 Norm，使用大模型界最稳健的 MLP 架构
        # 将双塔 8 层的通道 (768 * 8 = 6144) 直接压缩到 32 通道
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 8, 256),
            nn.GELU(),
            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        """
        输入 x 形状: [B*6, 4616, 768]
        """
        B_times_6 = x.shape[0]
        
        # 1. 拆解双塔与层数维度
        x = x.view(B_times_6, 8, 577, 768)
        
        # 2. 剔除 CLS Token，只保留 576 个视觉 Patch
        x = x[:, :, 1:, :]  # [B*6, 8, 576, 768]
        
        # =================================================================
        # 🚨 NPU 救命核心：必须加 .contiguous() 重新分配物理内存
        # =================================================================
        # 将 8 个层聚合到特征维度: [B*6, 576, 8, 768]
        x = x.permute(0, 2, 1, 3).contiguous() 
        # 展平特征: [B*6, 576, 6144]
        x = x.view(B_times_6, 576, -1) 
        
        # 3. 运行极度稳健的全连接层投影 (在 bfloat16 下依然稳如泰山)
        # [B*6, 576, 6144] -> [B*6, 576, 32]
        x = self.proj(x)
        
        # 4. 将一维序列完美复原为二维物理地图
        # [B*6, 32, 576]
        x = x.permute(0, 2, 1).contiguous() 
        # [B*6, 32, 24, 24]
        x = x.view(B_times_6, 32, 24, 24)
        
        # 5. 空间几何放大 (24x24 -> 48x48)
        # 仅仅在这唯一的一步数学插值上，采用 float32 保障边缘平滑度
        orig_dtype = x.dtype
        x = F.interpolate(
            x.float(), 
            size=(self.target_grid, self.target_grid), 
            mode='bilinear', 
            align_corners=False
        )
        
        return x.to(dtype=orig_dtype)