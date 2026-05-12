import os
import sys
import json
import torch
import torch_npu
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file

# ==========================================
# 1. 配置物理路径 (请根据你的实际情况核对)
# ==========================================
DINOV3_SOURCE_PATH = '/data/WorkSpace/dinov3_source'
MODEL_WEIGHT_DIR = '/data/WorkSpace/models/dinov3-convnext-large-pretrain-lvd1689m'
SAFETENSORS_FILE = os.path.join(MODEL_WEIGHT_DIR, 'model.safetensors')
CONFIG_FILE = os.path.join(MODEL_WEIGHT_DIR, 'preprocessor_config.json')

# 确保源码在路径中
sys.path.append(DINOV3_SOURCE_PATH)

def test_offline_load():
    print(" 开始 DINOv3 离线点火测试...")

    # ==========================================
    # 2. 手动解析 Image Processor 逻辑
    # ==========================================
    print(f" 正在解析配置文件: {CONFIG_FILE}")
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    # 提取 JSON 中的参数
    mean = config.get("image_mean", [0.485, 0.456, 0.406])
    std = config.get("image_std", [0.229, 0.224, 0.225])
    # 注意：CoBunny 之前测试 337 效果好，这里优先遵循配置或手动指定
    target_size = (config["size"]["height"], config["size"]["width"]) 

    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    print(f"✅ 图像预处理器构建完成，目标尺寸: {target_size}")

    # ==========================================
    # 3. 加载模型结构 (Hub Local 模式)
    # ==========================================
    try:
        print(f"创建模型实例 (Source: {DINOV3_SOURCE_PATH})...")
        # 根据你的具体模型名选择，这里以 convnext_large 为例
        model = torch.hub.load(
            DINOV3_SOURCE_PATH, 
            'dinov3_convnext_large', 
            source='local', 
            pretrained=False
        )
        
        # ==========================================
        # 4. 加载 Safetensors 权重
        # ==========================================
        print(f"📥 正在加载权重文件: {SAFETENSORS_FILE}")
# ==========================================
        # 4. 加载 Safetensors 权重 (终极影分身版)
        # ==========================================
        print(f"📥 正在加载权重文件: {SAFETENSORS_FILE}")
        raw_state_dict = load_file(SAFETENSORS_FILE)
        new_state_dict = {}

        # 第一步：执行通用转换
        for key, value in raw_state_dict.items():
            new_key = key
            new_key = new_key.replace(".layers.", ".")
            new_key = new_key.replace("stages.0.downsample_layers", "downsample_layers.0")
            new_key = new_key.replace("stages.1.downsample_layers", "downsample_layers.1")
            new_key = new_key.replace("stages.2.downsample_layers", "downsample_layers.2")
            new_key = new_key.replace("stages.3.downsample_layers", "downsample_layers.3")
            new_key = new_key.replace("depthwise_conv", "dwconv")
            new_key = new_key.replace("pointwise_conv1", "pwconv1")
            new_key = new_key.replace("pointwise_conv2", "pwconv2")
            
            # 【注意】我们在这里不处理最外层的 layer_norm，只处理局部的
            if "layer_norm" in new_key and not new_key.startswith("layer_norm"):
                new_key = new_key.replace("layer_norm", "norm")
                
            new_state_dict[new_key] = value

        # 第二步：【核心补丁】将唯一的 layer_norm 复制给模型需要的两个位置
        if "layer_norm.weight" in new_state_dict:
            global_weight = new_state_dict.pop("layer_norm.weight")
            new_state_dict["norm.weight"] = global_weight.clone()
            new_state_dict["norms.3.weight"] = global_weight.clone()
            
        if "layer_norm.bias" in new_state_dict:
            global_bias = new_state_dict.pop("layer_norm.bias")
            new_state_dict["norm.bias"] = global_bias.clone()
            new_state_dict["norms.3.bias"] = global_bias.clone()

        # 第三步：加载对齐
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"⚠️ 匹配情况: {msg}")
        
        if len(msg.missing_keys) == 0:
            print("✅ 权重完美对齐！所有的坑都填平了！")
        else:
            print(f"❗ 仍有 {len(msg.missing_keys)} 个 Key 未找到: {msg.missing_keys}")
        # 4. 加载
        msg = model.load_state_dict(new_state_dict, strict=False)

        # 某些版本 DINOv3 权重可能带有 'model.' 或 'backbone.' 前缀，需要对齐
        # 如果直接 load 报错，这里可能需要写一个简单的 key 映射逻辑
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"⚠️ 匹配情况: {msg}")
        
        if len(msg.missing_keys) == 0:
            print("✅ 权重完美对齐！")
        else:
            print(f"❗ 仍有 {len(msg.missing_keys)} 个 Key 未找到，请检查转换逻辑。")


        print("✅ 权重载入成功！")

        # ==========================================
        # 5. NPU 推理测试
        # ==========================================
        model = model.npu().eval()
        print("⚡ 模型已迁移至 NPU。")

        # 模拟一张输入图片 (3, H, W)
        dummy_input = torch.randn(1, 3, target_size[0], target_size[1]).npu()
        
        with torch.no_grad():
            # 使用 NPU 混合精度加速
            with torch.npu.amp.autocast():
                output = model(dummy_input)
        
        print(f"\n✨ 奇迹发生了！测试通过。")
        print(f" 特征输出维度: {output.shape}")

    except Exception as e:
        print(f"\n❌ 测试失败，错误信息: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_offline_load()
