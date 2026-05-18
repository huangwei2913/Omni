import os
import math
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from transformers import LayoutLMv3Model

# 模型路径
model_path = "/data/WorkSpace/models/layoutlmv3-base"


def interpolate_pos_encoding(model, target_grid_size=24):
    """
    LayoutLMv3 全架构对齐方案：
    1. 插值 pos_embed (绝对位置)
    2. 扩容 position_embeddings (共享层)
    3. 重生 visual_bbox 和 visual_position_ids
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
        patch_pos = F.interpolate(
            patch_pos,
            size=(target_grid_size, target_grid_size),
            mode="bicubic",
            align_corners=False
        )
        new_patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((cls_pos, new_patch_pos), dim=1)

        if "." in target_param_name:
            p_name, a_name = target_param_name.rsplit(".", 1)
            setattr(model.get_submodule(p_name), a_name, nn.Parameter(new_pos_embed.to(device, dtype)))
        else:
            setattr(model, target_param_name, nn.Parameter(new_pos_embed.to(device, dtype)))
        print(f"✅ 绝对位置编码已更新: {new_pos_embed.shape}")

    # ---- 2. 修复关键辅助张量 ----
    if hasattr(model, "patch_embed"):
        model.patch_embed.patch_shape = (target_grid_size, target_grid_size)

    new_num_patches = target_grid_size * target_grid_size
    step = 1000 // target_grid_size

    visual_bbox = []
    for i in range(target_grid_size):
        for j in range(target_grid_size):
            x0, y0 = j * step, i * step
            x1, y1 = (j + 1) * step, (i + 1) * step
            visual_bbox.append([x0, y0, x1, y1])

    new_visual_bbox = torch.tensor([[0, 0, 0, 0]] + visual_bbox, device=device).unsqueeze(0)
    new_visual_pos_ids = torch.arange(new_num_patches + 1, device=device).unsqueeze(0)

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
            new_num = new_num_patches + 2
            new_layer = nn.Embedding(new_num, module.embedding_dim).to(device, dtype)
            if "." in name:
                p_name, a_name = name.rsplit(".", 1)
                setattr(model.get_submodule(p_name), a_name, new_layer)
            else:
                setattr(model, name, new_layer)
            print(f"✅ 坐标 Embedding {name} 已扩容")


def save_layer_attention_maps(model, pixel_values, out_dir, num_patches_per_side=24):
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, output_attentions=True, return_dict=True)

    attentions = outputs.attentions
    num_patches = num_patches_per_side * num_patches_per_side
    patch_start = 1
    patch_end = patch_start + num_patches

    for layer_idx, attn in enumerate(attentions, start=1):
        # attn: [B, heads, seq_len, seq_len]
        attn = attn.mean(dim=1)[0]  # [seq_len, seq_len]

        # 所有 token 对 patch 的平均 attention
        avg_attn_to_patches = attn[:, patch_start:patch_end].mean(dim=0)  # [num_patches]

        heatmap = avg_attn_to_patches.reshape(num_patches_per_side, num_patches_per_side).cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap="viridis")
        plt.title(f"Layer {layer_idx} - Avg Token -> Patch Attention")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

        save_path = os.path.join(out_dir, f"xx{layer_idx}.jpg")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"✅ 已保存: {save_path}")


def test_visualize_embeddings(image_path):
    device = "npu" if torch.npu.is_available() else "cpu"

    model = LayoutLMv3Model.from_pretrained(model_path).to(device)
    model.eval()

    # 适配 384x384
    interpolate_pos_encoding(model, target_grid_size=24)

    raw_image = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    pixel_values = transform(raw_image).unsqueeze(0).to(device)

    # 先保存每一层 attention 热力图
    save_layer_attention_maps(
        model=model,
        pixel_values=pixel_values,
        out_dir="./layer_attn_out",
        num_patches_per_side=24
    )

    # 再保留你原来的 embedding 可视化
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, return_dict=True)
        embeddings = outputs.last_hidden_state

    patch_embeddings = embeddings[:, 1:, :].cpu().squeeze(0)  # [576, 768]
    heatmap = patch_embeddings.mean(dim=-1).reshape(24, 24).numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(raw_image.resize((384, 384)))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Visual Embedding Activation")
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar()
    plt.axis("off")

    plt.savefig("embeddings_layout.jpg", dpi=200, bbox_inches="tight")
    plt.show()

    print(f"✅ 嵌入可视化完成，输入尺寸: {pixel_values.shape}, 嵌入形状: {embeddings.shape}")


if __name__ == "__main__":
    test_visualize_embeddings("./5_center.jpg")