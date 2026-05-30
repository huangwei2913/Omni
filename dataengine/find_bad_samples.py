import json
import os

# ==================== 配置路径 ====================
INPUT_JSON = "/data/WorkSpace/datasets/OCR-Synthetic/bunny_format/ocr_train_200k.json"
# 将坏样本单独存一个文件，方便你查看它们到底长啥样
BAD_SAMPLES_REPORT = "/data/WorkSpace/datasets/OCR-Synthetic/bunny_format/bad_samples_report.json"
# (可选) 生成一个剔除坏样本后的纯净版 200k
CLEANED_200K_JSON = "/data/WorkSpace/datasets/OCR-Synthetic/bunny_format/ocr_train_200k_strict.json"

def audit_dataset():
    print(f"🔍 正在读取数据集: {INPUT_JSON} ...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total = len(data)
    print(f"✅ 读取完成，总计 {total} 条数据。开始严格质检...\n")
    
    bad_samples = []
    good_samples = []
    
    for idx, entry in enumerate(data):
        # 1. 统计文本中宣告了几个 <image> 坑位
        text_image_count = 0
        if 'conversations' in entry:
            for conv in entry['conversations']:
                # 统计所有的 <image> 标签
                text_image_count += conv.get('value', '').count('<image>')
                
        # 2. 统计实际提供了几张图片
        actual_image_count = 0
        if 'image' in entry and entry['image']:
            img_data = entry['image']
            if isinstance(img_data, list):
                # 如果是个列表，算列表长度
                actual_image_count = len(img_data)
            elif isinstance(img_data, str):
                # 如果是个单字符串，算 1 张图
                actual_image_count = 1
                
        # 3. 核心对比：坑位数量必须等于图片数量
        if text_image_count != actual_image_count:
            bad_info = {
                "dataset_index": idx,
                "sample_id": entry.get("id", "Unknown"),
                "error_reason": f"坑位不匹配! 文本里写了 {text_image_count} 个 <image>, 但实际提供了 {actual_image_count} 张图。",
                "conversations_preview": entry.get("conversations", [])[0:1], # 只看第一轮对话确认一下
                "image_field": entry.get("image", None)
            }
            bad_samples.append(bad_info)
        else:
            good_samples.append(entry)

    # ==================== 输出报告 ====================
    print("-" * 50)
    print("📊 【数据集质检报告】")
    print(f" - 总扫描数据量: {total}")
    print(f" - 完美对齐数据: {len(good_samples)}")
    print(f" - 发现脏数据量: {len(bad_samples)}")
    print("-" * 50)

    if len(bad_samples) > 0:
        print(f"\n🚨 发现了 {len(bad_samples)} 条图文不对齐的坏数据！")
        # 打印前 3 个坏数据看看情况
        for i in range(min(3, len(bad_samples))):
            print(f"  [坏样本示例 {i+1}]: 索引 {bad_samples[i]['dataset_index']} -> {bad_samples[i]['error_reason']}")
        
        print(f"\n💾 正在保存坏样本详情至: {BAD_SAMPLES_REPORT}")
        with open(BAD_SAMPLES_REPORT, "w", encoding="utf-8") as f:
            json.dump(bad_samples, f, ensure_ascii=False, indent=2)
            
        print(f"💾 正在生成剔除坏样本后的终极纯净版至: {CLEANED_200K_JSON}")
        with open(CLEANED_200K_JSON, "w", encoding="utf-8") as f:
            json.dump(good_samples, f, ensure_ascii=False)
            
        print("\n💡 建议：修改你的训练 .sh 脚本，将 data_path 指向新的 ocr_train_200k_strict.json 即可彻底根绝此类崩溃！")
    else:
        print("\n🎉 太完美了！这 20 万条数据没有任何图文不对齐的问题！")

if __name__ == "__main__":
    audit_dataset()