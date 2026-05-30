# clean_ocr_dataset.py
import json
import os
from PIL import Image
from tqdm import tqdm

# ==================== 【请根据你的实际路径配置】 ====================
IMAGE_FOLDER = "/data/WorkSpace/datasets/OCR-Synthetic/bunny_format"
INPUT_JSON = "/data/WorkSpace/datasets/OCR-Synthetic/bunny_format/ocr_train.json"
OUTPUT_JSON = "/data/WorkSpace/datasets/OCR-Synthetic/bunny_format/ocr_train_cleaned.json"

def check_entry_valid(entry):
    """单条轻量检查"""
    # 1. 空内容与基本标签检查
    if 'conversations' not in entry:
        return False
    
    has_resp = False
    for c in entry['conversations']:
        if c.get('from', '') in ['gpt', 'assistant']:
            if c.get('value', '').strip():
                has_resp = True
                break
    if not has_resp:
        return False
        
    # 2. 图片尺寸与完好性检查
    if 'image' in entry:
        img_paths = entry['image'] if isinstance(entry['image'], list) else [entry['image']]
        for img in img_paths:
            if os.path.isabs(img):
                full_path = img
            else:
                full_path = os.path.join(IMAGE_FOLDER, img)
            
            if not os.path.exists(full_path):
                return False
                
            try:
                # 仅读取 Header，单进程下磁盘负担极小
                with Image.open(full_path) as i:
                    w, h = i.size
                    if (w * h) > 80000000 or (w / h) > 15 or (h / w) > 15:
                        return False
            except Exception:
                return False
                
    return True

if __name__ == "__main__":
    print("📂 正在读取全量原始数据到内存...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    total_count = len(raw_data)
    print(f"✅ 数据读取完成。总数据量: {total_count} 条。")
    print("🌱 采用华为昇腾环境友好的【单进程流式清洗】模式，安全第一，绝不挂机...")
    
    final_data = []
    
    # 进度条直观可见
    for entry in tqdm(raw_data, total=total_count, desc="清洗进度"):
        if check_entry_valid(entry):
            final_data.append(entry)
            
        # 兜底保障：每 50000 条数据就在后台悄悄存个盘，防止 Xshell 意外断连
        if len(final_data) % 50000 == 0 and len(final_data) > 0:
            with open(OUTPUT_JSON + ".tmp", "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False)

    dropped_count = total_count - len(final_data)
    print(f"\n📊 【终审清洗报告】")
    print(f" - 原始数据总量: {total_count} 条")
    print(f" - 过滤无效数据: {dropped_count} 条")
    print(f" - 最终留存数据: {len(final_data)} 条")
    
    print(f"💾 正在保存清洗后的纯净数据至: {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    # 删除临时文件
    if os.path.exists(OUTPUT_JSON + ".tmp"):
        os.remove(OUTPUT_JSON + ".tmp")
        
    print("🎉 完美通关！单进程安全洗完，再也不用担心华为机器挂掉了。")