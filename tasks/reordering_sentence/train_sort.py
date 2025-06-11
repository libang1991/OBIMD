# train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from sortdata1 import CharOrderDataset, SimpleCharIDTokenizer  
from model_sort import  PositionalCharModel, PositionalCharModel1
import json
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

def custom_collate_fn(batch):
    batch_dict = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "posv": torch.stack([item["posv"] for item in batch]),
        "posh": torch.stack([item["posh"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "line": [item["line"] for item in batch],    # 保留为列表
        "box": [item["box"] for item in batch],      # 保留为列表
        "name": [item["name"] for item in batch],    # 保留为列表
    }
    return batch_dict   


if __name__ == "__main__":
    with open("datasets/sorttrain.json", "r") as f:
        raw_data = json.load(f)
    with open("datasets/sorttest.json", "r") as f:
        test_data = json.load(f)
    # 1. 准备数据
    tokenizer = SimpleCharIDTokenizer(vocab_file="datasets/vacabsort.txt")

    dataset = CharOrderDataset(raw_data, tokenizer, max_len=40, use_order_pos=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    testset = CharOrderDataset(test_data, tokenizer, max_len=40, is_train=False)
    val_dataloader = DataLoader(testset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    
    # 2. 初始化模型
    #原始模型
    model = PositionalCharModel(vocab_size=2750, char_dim=64, pos_dim=64, hidden_dim=128, max_pos=40)
 
    model = model.cuda()

    # 3. 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)  # -100 是 padding label
    ranking_loss_fn = nn.MarginRankingLoss(margin=1.0, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambda_rank = 1  # 排序损失的权重
    maxpairs = 10
    # 4. 训练循环
    for epoch in range(200):
        model.train()
        total_loss = []
        train_correct = 0
        train_total = 0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].cuda()
            input_posv = batch["posv"].cuda()
            input_posh = batch["posh"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
    
            outputs = model(input_ids, input_posv, input_posh, attention_mask)  # [B, T, vocab] or [B, T, T]
            
            loss_cls = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            loss = loss_cls 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            total_loss.append(loss.item())
            # ======= 训练准确率统计 =======
            
            pred_positions = outputs.argmax(dim=-1)
            mask = (labels != -100)
            train_correct += ((pred_positions == labels) & mask).sum().item()
            train_total += mask.sum().item()
           
        train_acc = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch}: loss = {sum(total_loss) / len(total_loss):.4f}| train acc = {train_acc:.4f}")
            # 保存模型
        model_filename = f"sortmain.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")
    # ======= 验证集评估 =======
   
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].cuda()
                input_posv = batch["posv"].cuda()
                input_posh = batch["posh"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()

                outputs = model(input_ids, input_posv, input_posv, attention_mask)  # [B, T, T] (每个token对应原始位置)
                pred_positions = outputs.argmax(dim=-1)     # [B, T]
                
                mask = (labels != -100)                     # 忽略 padding
                correct += ((pred_positions == labels) & mask).sum().item()
                total += mask.sum().item()
            

        acc = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {acc:.4f}")
 