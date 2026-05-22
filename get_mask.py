import os
import cv2
import numpy as np
import paddle
import paddle.inference as paddle_infer

# ==========================================
# 1. 初始化 910B NPU 推理引擎
# ==========================================
model_dir = '/data/WorkSpace/models/PP-OCRv5_server_det'
json_path = os.path.join(model_dir, 'inference.json')
params_path = os.path.join(model_dir, 'inference.pdiparams')

config = paddle_infer.Config(json_path, params_path)
config.enable_custom_device('npu')
config.switch_use_feed_fetch_ops(False)
predictor = paddle_infer.create_predictor(config)

# ==========================================
# 2. 完美的「等比例 & 32位对齐」预处理
# ==========================================
img_path = '494300.png'  # 你的细长测试图
orig_img = cv2.imread(img_path)
h_ori, w_ori, _ = orig_img.shape

# 严格保持长宽比，限制长边最大为 960
max_side = 960
if h_ori > w_ori:
    new_h = max_side
    new_w = int(w_ori * max_side / h_ori)
else:
    new_w = max_side
    new_h = int(h_ori * max_side / w_ori)

# 核心：向上取整到 32 的倍数，既满足 NPU 算子对齐，又绝不改变文字形状
new_h = int(np.ceil(new_h / 32.0) * 32)
new_w = int(np.ceil(new_w / 32.0) * 32)

# 执行平滑缩放
img_resized = cv2.resize(orig_img, (new_w, new_h))

# 标准化归一化
img_data = img_resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_data = (img_data - mean) / std

# HWC -> 1CHW
img_data = img_data.transpose((2, 0, 1))
img_data = np.expand_dims(img_data, axis=0)

# ==========================================
# 3. NPU 前向传播
# ==========================================
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])
input_handle.copy_from_cpu(img_data)

predictor.run()

# ==========================================
# 4. 工业级 DBNet 矢量后处理（拒绝像素级硬放大）
# ==========================================
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu()
raw_prob_map = output_data[0][0]  # 得到 (new_h, new_w) 的概率图

# 阈值化拿到初步二值图
segmentation = (raw_prob_map > 0.3).astype(np.uint8)

# 在低分辨率下提取文本行的矢量轮廓 (Contours)
contours, _ = cv2.findContours(segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 创建与【原图尺寸】一模一样的全黑画布
mask = np.zeros((h_ori, w_ori), dtype=np.uint8)

for contour in contours:
    # 过滤掉面积小于 5 像素的零星噪声点
    if cv2.contourArea(contour) < 5:
        continue
    
    # 核心操作：把低分辨率下的点阵坐标，精准投影映射回原图的分辨率坐标系
    contour_projected = contour.astype(np.float32)
    contour_projected[:, 0, 0] = contour_projected[:, 0, 0] * w_ori / new_w  # X坐标映射
    contour_projected[:, 0, 1] = contour_projected[:, 0, 1] * h_ori / new_h  # Y坐标映射
    contour_projected = contour_projected.astype(np.int32)
    
    # 在原图画布上，把这个文本行多边形内部全部填满（变成纯实心白色块）
    cv2.fillPoly(mask, [contour_projected], 255)

# ==========================================
# 5. 形态学膨胀：补偿 DBNet 缩小的文本核
# ==========================================
# 使用 5x5 的矩形核进行适度膨胀，把文本行边缘包裹得更丰满
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
final_mask = cv2.dilate(mask, kernel, iterations=1)

# 保存结果
cv2.imwrite('text_mask.png', final_mask)
print(f"🔥 修正成功！等比例矢量投影后处理完成。")
print(f"输入尺寸: {orig_img.shape} -> 实际推理尺寸: ({new_h}, {new_w}) -> 输出优质 Mask: {final_mask.shape}")