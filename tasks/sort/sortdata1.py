import torch

import random
from torch.utils.data import Dataset, DataLoader

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
class CharOrderDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32, is_train=True, use_order_pos=True):
        """
        data: List of dicts, each containing 'line' and 'box'
        tokenizer: 字符编码为 token_id
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train=is_train
        self.use_order_pos=use_order_pos
       
        for item in data:
            lines = item['line']
            boxes = item['box']
            width = item.get('width')
            height = item.get('height')
            name = item.get('name')
            for line, box in zip(lines, boxes):
                valid = []
                valid_count = 0
                for i, ch in enumerate(line):
                    
                    if ch is None:
                        token = "[UNK]"
                    else:
                        token = ch
                    if token != "[UNK]":
                        valid_count += 1
                    valid.append((i, token, box[i]))
                
                if valid_count <= 1:
                    continue  # 跳过有效字符 ≤ 1 的样本
                self.samples.append((valid, width, height, name))

            # for line, box in zip(lines, boxes):
            #     # 保留有效字符（非 None）
            #     valid = [(i, ch, box[i]) for i, ch in enumerate(line) if ch is not None]
            #     if len(valid) <= 1:
            #         continue  # 跳过有效字符 ≤ 1 的样本
            #     self.samples.append(valid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, width, height,name = self.samples[idx]
      
        # 拆分出原始字符和 box
        orig_indices = [i for i, _, _ in sample]
        chars = [ch for _, ch, _ in sample]
        boxes = [b for _, _, b in sample]

        # 打乱顺序
        # items = list(zip(orig_indices, chars, boxes))
        # random.shuffle(items)
        # shuffled_indices, shuffled_chars, shuffled_boxes = zip(*items)
        if self.is_train:
            items = list(zip(orig_indices, chars, boxes))
            random.shuffle(items)
            shuffled_indices, shuffled_chars, shuffled_boxes = zip(*items)
        else:
            shuffled_indices, shuffled_chars, shuffled_boxes = orig_indices, chars, boxes
        # 编码字符
        inputs = self.tokenizer([list(shuffled_chars)], padding='max_length', truncation=True,
                                max_length=self.max_len, return_tensors='pt')

        # 提取前 max_len 的 box 坐标
        coords = [list(map(int, b.split(','))) for b in shuffled_boxes[:self.max_len]]
        xs = [x for x, y, _, _ in coords]
        ys = [y for x, y, _, _ in coords]
        if self.use_order_pos:
        # 计算每个 x/y 在当前样本中的相对位置（排序秩序）
            x_order = {v: i for i, v in enumerate(sorted(xs))}
            y_order = {v: i for i, v in enumerate(sorted(ys))}

            posh = [x_order[x]+1 for x in xs]
            posv = [y_order[y]+1 for y in ys]
        else:
            # 计算每个样本中字符的最小/最大坐标
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # 避免除以 0（如果所有 x 或 y 值相同）
            x_range = x_max - x_min if x_max > x_min else 1
            y_range = y_max - y_min if y_max > y_min else 1

            
            # 使用归一化坐标
            posh = [int((x - x_min) / x_range * 1000)+1 for x in xs]
            posv = [int((y - y_min) / y_range * 1000)+1 for y in ys]


        # padding 到 max_len
        pad_len = self.max_len - len(posh)
        posv += [1] * pad_len
        posh += [1] * pad_len

        # 构造标签：打乱后，每个字符的原始位置
        labels = list(shuffled_indices)[:self.max_len]
        labels += [-100] * (self.max_len - len(labels))

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'posv': torch.tensor(posv),
            'posh': torch.tensor(posh),
            'labels': torch.tensor(labels),
            'name': name,
            'line': list(shuffled_chars),     # 加入字符列表
            'box': shuffled_boxes 
        }


class SimpleCharIDTokenizer:
    def __init__(self, vocab_file):
        """
        vocab_file: 文件路径，每行一个token，格式应为：
            [PAD]
            [UNK]
            a
            b
            ...
        """
        self.vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = idx

        self.pad_token_id = self.vocab.get("[PAD]", 0)
        self.unk_token_id = self.vocab.get("[UNK]", 1)

    def __call__(self, list_of_char_lists, padding='max_length', truncation=True, max_length=40, return_tensors=None):
        all_input_ids = []
        all_attention_masks = []

        for char_list in list_of_char_lists:
            ids = [self.vocab.get(ch, self.unk_token_id) for ch in char_list]
            if truncation:
                ids = ids[:max_length]
            pad_len = max_length - len(ids)
            ids += [self.pad_token_id] * pad_len
            mask = [1] * (len(ids) - pad_len) + [0] * pad_len

            all_input_ids.append(ids)
            all_attention_masks.append(mask)

        result = {
            "input_ids": torch.tensor(all_input_ids),
            "attention_mask": torch.tensor(all_attention_masks)
        }
        return result
 

if __name__ == "__main__":
    # 加载数据
    import json

    with open("datasets/sorttrain.json", "r") as f:
        raw_data = json.load(f)
    max_len = 0
    max_example = None
    vocab_file="datasets/vacabsort.txt"
    tokenizer = SimpleCharIDTokenizer(vocab_file="datasets/vacabsort.txt")
    dataset = CharOrderDataset(raw_data, tokenizer, max_len=40, is_train=True, use_order_pos=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    print(f"总样本数量: {len(dataset)}") 
    # 遍历
    for batch in dataloader:
        print(batch["input_ids"][12])      # [B, T]
        print(batch["posh"][12])        # [B, T]
        print(batch["labels"][12])         # [B, T]
        break
    # for batch_idx, batch in enumerate(dataloader):
    #     posh = batch["posh"]  # [B, T]

    #     # 检查 > 1000
    #     mask_gt = posh > 1000
    #     # 检查 == 0
    #     mask_eq0 = posh == 0

    #     if mask_gt.any() or mask_eq0.any():
    #         if mask_gt.any():
    #             indices = torch.nonzero(mask_gt)  # 返回 (batch_idx, token_idx)
    #             for i, j in indices:
    #                 value = posh[i, j].item()
    #                 print(f"[>1000] Batch {batch_idx}, Sample {i}, Token {j} has posh = {value}")

    #         if mask_eq0.any():
    #             indices = torch.nonzero(mask_eq0)
    #             for i, j in indices:
    #                 value = posh[i, j].item()
    #                 print(f"[==0] Batch {batch_idx}, Sample {i}, Token {j} has posh = {value}")
            
    #         break
