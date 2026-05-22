import h5py
import json

# 替换为你实际的文件路径
h5_path = '/data/WorkSpace/datasets/OCR-Synthetic/en/train/train_060.h5'

def check_keys(file_path):
    with h5py.File(file_path, 'r') as f:
        # 获取 annotations 这个数据集里的第一条记录
        # 这里用 index 0 来读取第一条数据
        raw_anno = f['annotations'][0].decode('utf-8')
        anno_dict = json.loads(raw_anno)
        
        print("--- 样本中的字段列表 (Keys) ---")
        for key in anno_dict.keys():
            print(f"字段名: {key}")
            
        print("\n--- 预览一条数据的内容 ---")
        # 打印一下第一个键的值，看看是不是文本
        first_key = list(anno_dict.keys())[0]
        print(f"示例内容 ({first_key}): {anno_dict[first_key]}")

check_keys(h5_path)