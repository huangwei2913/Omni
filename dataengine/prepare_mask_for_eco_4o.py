import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# ==================== 1. 全局配置 ====================
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "/home/huangwei/models/sam2.1-hiera-large/sam2.1_hiera_large.pt"

DATA_DIR = "/mnt/Echo-4o-Image/Instruction-Following-Image"
JSONL_PATH = os.path.join(DATA_DIR, "Instruction-Following-Image.jsonl")
MASK_OUT_DIR = os.path.join(DATA_DIR, "ocr_masks")
os.makedirs(MASK_OUT_DIR, exist_ok=True)

JSON_OUT_PATH = os.path.join(DATA_DIR, "echo_4o_train_with_masks.json")

# ==================== 2. I/O 异步写入函数 ====================
def save_mask_async(path, img):
    """通过线程池异步写入硬盘，不阻塞 GPU 推理主流程"""
    cv2.imwrite(path, img)

# ==================== 3. 单卡高性能核心进程 ====================
def worker_gpu_process(gpu_id, data_chunk, return_dict):
    # 绑定独立物理显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    
    # 开启 PyTorch 自身的算力优化后门
    torch.backends.cudnn.benchmark = True
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    try:
        model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
        predictor = SAM2ImagePredictor(model)
    except Exception as e:
        print(f"❌ GPU {gpu_id} 载入模型失败: {e}")
        return

    local_results = []
    # 开辟每个子进程专属的 I/O 异步线程池（分配 4 个线程用于快速写盘）
    io_executor = ThreadPoolExecutor(max_workers=4)
    
    pbar = tqdm(data_chunk, desc=f"🚀 GPU {gpu_id} 流水线", position=gpu_id, leave=False)
    
    for line in pbar:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue

        instruction = data.get("instruction", "").strip()
        orig_img_path = data.get("output_image", "")
        if not instruction or not orig_img_path:
            continue

        img_name = os.path.basename(orig_img_path)
        full_img_path = os.path.join(DATA_DIR, "images", img_name)
        if not os.path.exists(full_img_path):
            continue

        # 1. 快速读取与转码
        pil_img = Image.open(full_img_path).convert("RGB")
        W, H = pil_img.size
        image_np = np.array(pil_img)

        # 2. 核心 GPU 推理 (严格压榨单精度/半精度性能)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                predictor.set_image(image_np)

                # 预先构建打包好的多点 batch 矩阵，避免循环单点传入
                cx, cy = W // 2, H // 2
                point_coords = np.array([[cx, cy], [cx-20, cy], [cx+20, cy], [cx, cy-20], [cx, cy+20]], dtype=np.float32)
                point_labels = np.array([1, 1, 1, 1, 1], dtype=np.int32)

                masks, _, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )
        
        best_mask = np.array(masks[0]) > 0

        binary_mask_img = np.zeros((H, W), dtype=np.uint8)
        binary_mask_img[best_mask] = 255

        img_base_name = img_name.split('.')[0]
        mask_file_name = f"{img_base_name}_mask.png"
        mask_save_path = os.path.join(MASK_OUT_DIR, mask_file_name)
        
        # 🚨 性能提升核心：扔给后台线程去慢慢写盘，当前 GPU 马上进入下一张图的特征提取！
        io_executor.submit(save_mask_async, mask_save_path, binary_mask_img)

        # 4. 标签拼装
        if len(instruction.split()) <= 4:
            human_query = "<image>\nWhat is in this photo?"
            clean_item = instruction.replace('a photo of a ', '').replace('a photo of an ', '').replace('a photo of ', '').strip('.')
            gpt_response = f"This is a photo of {clean_item}."
        else:
            human_query = "<image>\nCan you describe what is in this image?"
            gpt_response = instruction

        local_results.append({
            "id": f"echo_{img_base_name}",
            "image": f"images/{img_name}",
            "object_mask": f"ocr_masks/{mask_file_name}",
            "conversations": [
                {"from": "human", "value": human_query},
                {"from": "gpt", "value": gpt_response}
            ]
        })

    # 关闭线程池并等待最后的写入
    io_executor.shutdown(wait=True)
    return_dict[gpu_id] = local_results


# ==================== 4. 主控集群分发 ====================
def main():
    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    print(f"🌟 发现 {num_gpus} 张强力 Tesla V100S！启动多进程异步 I/O 战术...")

    if not os.path.exists(JSONL_PATH):
        print(f"❌ 找不到输入源: {JSONL_PATH}")
        return

    print("📖 正在将所有数据集全量读入高速内存...")
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        all_lines = [line for line in f if line.strip()]
    
    total_samples = len(all_lines)
    print(f"📊 任务总计: {total_samples} 条。开始完美切片...")

    # 切分任务
    chunks = np.array_split(all_lines, num_gpus)

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_gpu_process, args=(gpu_id, chunks[gpu_id], return_dict))
        processes.append(p)
        p.start()

    print("🔥 8路强力流已全部饱和攻击。请随时通过 nvidia-smi 观察利用率飙升情况！")
    
    for p in processes:
        p.join()

    # 数据集聚拢
    final_bunny_dataset = []
    for gpu_id in range(num_gpus):
        if gpu_id in return_dict:
            final_bunny_dataset.extend(return_dict[gpu_id])

    # 封存 JSON
    print(f"💾 正在写入高内聚多模态标签文件: {JSON_OUT_PATH}")
    with open(JSON_OUT_PATH, 'w', encoding='utf-8') as json_f:
        json.dump(final_bunny_dataset, json_f, ensure_ascii=False, indent=2)

    print("🎉 [Success] 速度瓶颈已被彻底击碎，全量 Mask 生成完毕！")


if __name__ == "__main__":
    main()