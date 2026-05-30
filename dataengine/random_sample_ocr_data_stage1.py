import json
import random
import os

# 配置路径
INPUT_JSON = "/data/WorkSpace/datasets/OCR-Synthetic/bunny_format/ocr_train.json"
OUTPUT_JSON = "/data/WorkSpace/datasets/OCR-Synthetic/bunny_format/ocr_train_200k.json"
SAMPLE_SIZE = 200000

def sample_dataset():
    print(f"📂 正在读取清洗后的数据集: {INPUT_JSON} ...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total = len(data)
    print(f"✅ 读取完成，总共有 {total} 条数据。")
    
    if total <= SAMPLE_SIZE:
        print("⚠️ 总量不足 20 万，将直接使用全部数据。")
        final_data = data
    else:
        print(f"🎲 正在随机抽取 {SAMPLE_SIZE} 条样本...")
        random.seed(42)  # 固定随机种子，确保可复现
        final_data = random.sample(data, SAMPLE_SIZE)
        
    print(f"💾 正在保存到: {OUTPUT_JSON} ...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False)
        
    print(f"🎉 处理完成！已生成 20 万规模训练集。")

if __name__ == "__main__":
    sample_dataset()