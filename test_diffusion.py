import torch
import torch_npu
from safetensors.torch import load_file
import json
import os
from flux_decoder_core import FluxSmallDecoder # 使用刚才创建的本地类

# --- 配置路径 ---
model_dir = "/data/WorkSpace/models/FLUX.2-small-decoder"
weight_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
config_path = os.path.join(model_dir, "config.json")

with open(config_path, 'r') as f:
    config = json.load(f)

device = "npu:0"
dtype = torch.bfloat16

# --- 实例化 ---
print(f"🛠️ 正在构建 FLUX.2 专用 32-Channel 结构...")
model = FluxSmallDecoder(config).to(dtype)

# --- 加载权重 ---
print(f"📦 正在加载权重: {weight_path}")
state_dict = load_file(weight_path)

# 关键：FLUX.2 的 VAE 权重通常带有 'decoder.' 前缀
# 我们需要将其映射到 model.decoder 上
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("decoder."):
        new_state_dict[k] = v
    else:
        new_state_dict[f"decoder.{k}"] = v
msg = model.load_state_dict(new_state_dict, strict=False)
print(f"✅ 权重对齐完成! 缺失键: {len(msg.missing_keys)}")

model.to(device)
model.eval()


from PIL import Image
import numpy as np

# --- 在你之前的验证代码后面加上这段 ---
with torch.no_grad():
    # 1. 产生随机 Latent (模拟高熵视觉特征)
    test_latent = torch.randn(1, 32, 48, 48).to(device, dtype=dtype)
    
    # 2. 解码
    # 注意：FLUX VAE 输出通常在 [-1, 1] 之间
    output = model.decode(test_latent) # [1, 3, 384, 384]
    
    # 3. 后处理：映射回 [0, 255]
    output = (output + 1.0) / 2.0  # 归一化到 [0, 1]
    output = output.clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    output = (output[0] * 255).astype(np.uint8)
    
    # 4. 保存看一眼
    Image.fromarray(output).save("npu_decoder_test.png")
    print("📸 验证图片已保存为 npu_decoder_test.png，请检查是否有纹理内容。")

