import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sortdata1 import CharOrderDataset, SimpleCharIDTokenizer  
from model_sort import  PositionalCharModel
import json
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
def hungarian_decode(logits, mask):
    """
    logits: [B, T, num_classes]
    mask: [B, T] 0/1 表示有效字符
    返回每个样本中每个字符的最佳分配位置（按匈牙利算法）
    """
    batch_size, seq_len, num_classes = logits.size()
    pred_positions = torch.zeros(batch_size, seq_len, dtype=torch.long)

    for i in range(batch_size):
        valid_len = mask[i].sum().item()
        if valid_len == 0:
            continue

        logits_i = logits[i][:valid_len]  # [valid_len, num_classes]
        cost = -F.log_softmax(logits_i, dim=-1).detach().cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost)
        pred_positions[i, :valid_len] = torch.tensor(col_ind, dtype=torch.long)

    return pred_positions.cuda()
from PIL import Image, ImageDraw, ImageFont
import os

def visualize_char_sequence(
    image_root,
    save_dir,
    item,
    label_order,
    pred_order,
    font_path=None
):
    """
    可视化标签顺序和预测顺序，叠加到已有图上，避免覆盖。
    """
    os.makedirs(save_dir, exist_ok=True)
    image_name = item['name']
    base_image_path = os.path.join(image_root, image_name)

    label_path = os.path.join(save_dir, f"{image_name}_label.png")
    pred_path = os.path.join(save_dir, f"{image_name}_pred.png")

    if not os.path.exists(base_image_path):
        print(f"[!] 图片不存在: {base_image_path}")
        return

    # 字体
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, size=14)
    else:
        font = ImageFont.load_default()

    def draw_sequence(image, chars, boxes, order, color="blue"):
        draw = ImageDraw.Draw(image)
        points = []
        for i, idx in enumerate(order):
            if idx >= len(chars):
                continue
            char = chars[idx]
            box_str = boxes[idx]
            x1, y1, w, h = map(int, box_str.split(","))
            x2, y2 = x1 + w, y1 + h

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            points.append((cx, cy))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            draw.text((x1, y1), str(i), fill=color, font=font)

        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=color, width=2)
            draw.ellipse(
                [points[i][0] - 2, points[i][1] - 2, points[i][0] + 2, points[i][1] + 2],
                fill=color
            )

        return image

    chars = item['line']
    boxes = item['box']

    if len(chars) != len(boxes):
        print(f"[!] 字符和框数量不一致: {image_name}")
        return

    # --- 可视化 label 顺序 ---
    if os.path.exists(label_path):
        image_label = Image.open(label_path).convert("RGB")
    else:
        image_label = Image.open(base_image_path).convert("RGB")

    image_label = draw_sequence(image_label, chars, boxes, label_order, color="green")
    image_label.save(label_path)

    # --- 可视化 pred 顺序 ---
    if os.path.exists(pred_path):
        image_pred = Image.open(pred_path).convert("RGB")
    else:
        image_pred = Image.open(base_image_path).convert("RGB")

    image_pred = draw_sequence(image_pred, chars, boxes, pred_order, color="blue")
    image_pred.save(pred_path)

    print(f"[?] 可视化保存: {image_name}_label.png & {image_name}_pred.png")

def custom_collate_fn(batch):
    batch_dict = {}

    # 处理张量类字段
    for key in ['input_ids', 'attention_mask', 'posv', 'posh', 'labels']:
        batch_dict[key] = torch.stack([item[key] for item in batch])

    # 保留非张量字段（如 name, line, box）为 list
    for key in ['name', 'line', 'box']:
        batch_dict[key] = [item[key] for item in batch]

    return batch_dict
if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== 加载数据 ======
    # with open("datasets/sorttest.json", "r") as f:
    #     test_data = json.load(f)
    with open("datasets/sorttest.json", "r") as f:
        test_data = json.load(f)
    # 初始化 tokenizer，读取词表（假设你有 vocabsort.txt）
    tokenizer = SimpleCharIDTokenizer(vocab_file="datasets/vacabsort.txt")

    # 构建 Dataset 和 DataLoader（is_train=False 禁用打乱）

    test_dataset  = CharOrderDataset(test_data, tokenizer, max_len=40, is_train=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn = custom_collate_fn)
    # ====== 加载模型 ======
    model = PositionalCharModel(vocab_size=2750, char_dim=64, pos_dim=64, hidden_dim=128, max_pos=40)

    model.load_state_dict(torch.load('sortmain.pt'))  # 替换模型路径
    model.to(device)
    model.eval()

    # ====== 验证指标统计 ======
    correct = 0
    top3_correct = 0
    total = 0
    total_distance = 0
    unique_correct = 0
    exact_match = 0
    model.eval()
    correct = 0
    total = 0
   
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].cuda()
            input_posv = batch["posv"].cuda()
            input_posh = batch["posh"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            names = batch['name']
            lines = batch['line']
            boxes = batch['box']
            outputs = model(input_ids, input_posv, input_posv, attention_mask)  # [B, T, T] (每个token对应原始位置)
            pred_positions = outputs.argmax(dim=-1)     # [B, T]
          
            mask = (labels != -100)                     # 忽略 padding
            correct += ((pred_positions == labels) & mask).sum().item()
            total += mask.sum().item()
            # top-3 正确 & ADE 计算
            # logits: (B, L, num_classes)
            topk = outputs.topk(3, dim=-1).indices  # [B, T, 3]

            # top-3 correct
            top3_correct += ((topk == labels.unsqueeze(-1)) & mask.unsqueeze(-1)).any(-1).sum().item()

            # ADE: 平均绝对误差（只对 mask 部分）
            distance = (pred_positions - labels).abs()
            total_distance += (distance * mask).sum().item()
            
            j = 0
            pred_order = pred_positions[j][:attention_mask[j].sum()].tolist()
            label_order = labels[j][:attention_mask[j].sum()].tolist()

            item = {
                "name": names[j],
                "line": lines[j],
                "box": boxes[j],
            }
            # # 预测结果可视化
            # visualize_char_sequence(
            #     image_root=r"datasets/fasmile",
            #     save_dir="vis_results1",
            #     item=item,
            #     label_order=label_order,
            #     pred_order=pred_order,
            #     font_path="simhei.ttf"  # 可选
            # )
            
    acc = correct / total if total > 0 else 0
    top3_acc = top3_correct / total if total > 0 else 0
    ade = total_distance / total if total > 0 else 0

    print(f"Validation Accuracy (Top-1): {acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Average Distance Error (ADE): {ade:.4f}")