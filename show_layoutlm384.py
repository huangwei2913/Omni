import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt # 新增：用于绘图
import numpy as np              # 新增：用于处理矩阵


import torch
import torch_npu
from PIL import Image
import torchvision.transforms as T
from transformers import LayoutLMv3Model
import torch.nn.functional as F

# 模型路径
model_path = "/data/WorkSpace/models/layoutlmv3-base"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def interpolate_pos_encoding(model, target_grid_size=24):
    """
    LayoutLMv3 全架构对齐方案：
    1. 插值 pos_embed (绝对位置)
    2. 扩容 position_embeddings (共享层)
    3. 重生 visual_bbox 和 visual_position_ids (解决 577 vs 197 冲突)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---- 1. 修复绝对位置编码 (视觉专用) ----
    target_param_name = None
    for name, param in model.named_parameters():
        if param.ndim == 3 and param.shape[1] == 197:
            target_param_name = name
            break
            
    if target_param_name:
        param = dict(model.named_parameters())[target_param_name]
        old_pos = param.data
        cls_pos = old_pos[:, :1, :]
        patch_pos = old_pos[:, 1:, :].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(target_grid_size, target_grid_size), mode='bicubic', align_corners=False)
        new_patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((cls_pos, new_patch_pos), dim=1)
        
        # 写入参数
        if "." in target_param_name:
            p_name, a_name = target_param_name.rsplit('.', 1)
            setattr(model.get_submodule(p_name), a_name, nn.Parameter(new_pos_embed.to(device, dtype)))
        else:
            setattr(model, target_param_name, nn.Parameter(new_pos_embed.to(device, dtype)))
        print(f"✅ 绝对位置编码已更新: {new_pos_embed.shape}")

    # ---- 2. 修复关键辅助张量 (解决 577 vs 197 报错的关键) ----
    
    # 更新补丁形状
    if hasattr(model, "patch_embed"):
        model.patch_embed.patch_shape = (target_grid_size, target_grid_size)
    
    # 重新生成视觉 BBox (归一化到 0-1000)
    new_num_patches = target_grid_size * target_grid_size
    step = 1000 // target_grid_size
    visual_bbox = []
    for i in range(target_grid_size):
        for j in range(target_grid_size):
            x0, y0 = j * step, i * step
            x1, y1 = (j + 1) * step, (i + 1) * step
            visual_bbox.append([x0, y0, x1, y1])
    
    # 加上 CLS token 的 bbox [0,0,0,0]，总长 577
    new_visual_bbox = torch.tensor([[0,0,0,0]] + visual_bbox, device=device).unsqueeze(0)
    # 重新生成视觉 Position IDs
    new_visual_pos_ids = torch.arange(new_num_patches + 1, device=device).unsqueeze(0)

    # 替换模型属性 (LayoutLMv3 可能会把这些存为 buffer 或 attribute)
    for m in model.modules():
        if hasattr(m, "visual_bbox") and m.visual_bbox is not None:
            m.visual_bbox = new_visual_bbox
            print(f"✅ 已重置 visual_bbox: {m.visual_bbox.shape}")
        if hasattr(m, "visual_position_ids") and m.visual_position_ids is not None:
            m.visual_position_ids = new_visual_pos_ids
            print(f"✅ 已重置 visual_position_ids: {m.visual_position_ids.shape}")

    # ---- 3. 扩容坐标 Embedding 层 ----
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "position_embeddings" in name:
            new_num = new_num_patches + 2 # 留出 buffer
            new_layer = nn.Embedding(new_num, module.embedding_dim).to(device, dtype)
            # 这种赋值需要小心，如果是父子关系
            if "." in name:
                p_name, a_name = name.rsplit('.', 1)
                setattr(model.get_submodule(p_name), a_name, new_layer)
            else:
                setattr(model, name, new_layer)
            print(f"✅ 坐标 Embedding {name} 已扩容")


# ... (保留原有的 interpolate_pos_encoding 等函数) ...

def test_visualize_embeddings(image_path):
    device = "npu" if torch.npu.is_available() else "cpu"
    
    # 1. 加载模型
    model = LayoutLMv3Model.from_pretrained(model_path).to(device)
    model.eval()

    # 执行位置编码插值（针对 384x384）
    interpolate_pos_encoding(model, target_grid_size=24)

    # 2. 修改点：读取本地图像
    # 不再使用 Image.new，而是使用 Image.open
    raw_image = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    pixel_values = transform(raw_image).unsqueeze(0).to(device)

    # 3. 提取嵌入 (Embedding)
    with torch.no_grad():
        # 只要视觉部分的输出
        outputs = model.forward(pixel_values=pixel_values)
        # last_hidden_state 形状通常是 [1, 577, 768] (对于 384x384 且 patch=16, 24x24+1=577)
        embeddings = outputs.last_hidden_state

    # 4. 可视化逻辑
    # 移除 [CLS] token (第一个 token)，剩下的 576 个是图像 patch 的嵌入
    patch_embeddings = embeddings[:, 1:, :].cpu().squeeze(0) # [576, 768]
    
    # 将高维特征压缩为 2D 形状以便显示
    # 方案 A：计算每个 patch 所有维度的平均激活强度
    heatmap = patch_embeddings.mean(dim=-1).reshape(24, 24).numpy() 
    
    # 绘图对比
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(raw_image.resize((384, 384)))
    
    plt.subplot(1, 2, 2)
    plt.title("Visual Embedding Activation")
    plt.imshow(heatmap, cmap='viridis') # 越亮代表该区域特征响应越强
    plt.colorbar()
    plt.savefig("embediings_layout.jpg")  # 同时保存一份带框的报告
    plt.show()

    print(f"✅ 嵌入可视化完成，输入尺寸: {pixel_values.shape}, 嵌入形状: {embeddings.shape}")


if __name__ == "__main__":
    test_visualize_embeddings("./5_center.jpg")