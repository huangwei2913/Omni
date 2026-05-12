import torch
import torch_npu
from PIL import Image
import torchvision.transforms as T
from transformers import LayoutLMv3Model, LayoutLMv3Config

model_path = "/data/WorkSpace/models/layoutlmv3-base"

def test_dimensions_pure_vision():
    device = "npu" if torch.npu.is_available() else "cpu"
    print(f"当前检测到设备: {device}")
    
    # 1. 直接加载模型
    model = LayoutLMv3Model.from_pretrained(model_path).to(device)
    model.eval()

    # 2. 手动进行图像预处理 (模仿 LayoutLMv3 的标准流程)
    # LayoutLMv3 默认均值和标准差
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dummy_image = Image.new('RGB', (224, 224), color = (73, 109, 137))
    pixel_values = transform(dummy_image).unsqueeze(0).to(device) # [1, 3, 224, 224]

    print(f"输入 Tensor 形状: {pixel_values.shape}") 

    with torch.no_grad():
        # 核心：只传 pixel_values，跳过文本输入
        outputs = model(pixel_values=pixel_values)
        
    last_hidden_state = outputs.last_hidden_state
    
    print("-" * 30)
    print(f"NPU 输出特征维度: {last_hidden_state.shape}")
    print(f"预期隐藏层维度: 768")
    
    if last_hidden_state.shape[-1] == 768:
        print("✅ 维度验证成功！无需 Tesseract 即可运行纯视觉提取。")

if __name__ == "__main__":
    test_dimensions_pure_vision()