import json
from collections import Counter
import random
from PIL import Image
import os

if __name__ == '__main__':
    data_path = "datasets/oracle_lines.json"
    box_path = "datasets/oracle_boxs.json"
    data_dir = "../image_mask_dataset/dataset/拓片"
    
    
    # 加载数据集
    with open(data_path, "r", encoding = "utf-8") as f:
        datasets = json.load(f)

    all_lines = []

    # 统计字频
    codes = []
    for lines in datasets:
        for line in lines["line"]:
            codes += line
            all_lines.append([line, lines["name"]])

    statistic_numbers = Counter(codes)
    
    new_train_dataset = []
    
    
    train_code_set = set()
    
    # 加载数据集
    with open(data_path, "r", encoding = "utf-8") as f:
        datasets = json.load(f)
    with open(box_path, "r", encoding = "utf-8") as f:
        boxs = json.load(f)
    
    datasets_ = []
    for item, box in zip(datasets, boxs):
        item["box"] = box["boxs"]
        size = Image.open(os.path.join(data_dir, item["name"])).size
        item["width"] = size[0]
        item["height"] = size[1]
        
        datasets_.append(item)
    dataset = datasets_
    
    others = []
    for item in datasets:
        up_train_code_set = set()
            
        for line in item["line"]:
            for code in line:
                up_train_code_set.add(code)
        
        if len(up_train_code_set.union(train_code_set)) > len(train_code_set):
            train_code_set = up_train_code_set.union(train_code_set)
            new_train_dataset.append(item)
        else:
            others.append(item)
    
    new_test_dataset = []
    
    random.shuffle(others)
    
    # 对others进行划分 
    train_test_ratio = 0.9
    
    for item in others:
        if len(new_test_dataset) / len(datasets) < 1 - train_test_ratio:
            new_test_dataset.append(item)
        else:
            new_train_dataset.append(item)

    # 保存数据集
    with open("datasets/train_dataset.json", "w", encoding = "utf-8") as f:
        json.dump(new_train_dataset, f, ensure_ascii = False, indent = 4)

    with open("datasets/test_dataset.json", "w", encoding = "utf-8") as f:
        json.dump(new_test_dataset, f, ensure_ascii = False, indent = 4)
    
    with open("datasets/all_dataset.json", "w", encoding = "utf-8") as f:
        json.dump(datasets, f, ensure_ascii = False, indent = 4)