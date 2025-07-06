from transformers import BertTokenizer
import json
from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np
import torch
from scipy.spatial import Delaunay
from collections import defaultdict

def build_neighbor_attention_mask(positions, add_cls_token=True, max_length=512):
    """
    根据字符框坐标生成邻接图注意力掩码，用于控制 Transformer 注意力。
    
    Args:
        positions: List of [x0, y0, x1, y1]，每个元素表示一个字符框坐标。
        add_cls_token: 如果为 True，则将第0位留给 [CLS]，且所有字符都能看到它。

    Returns:
        attention_mask: torch.BoolTensor of shape [N+1, N+1] if add_cls_token else [N, N]
                        True 表示可见，False 表示不可见。
    """
    points = np.array([[x0, y0] for x0, y0, _, _ in positions])
    points = points[1:]
    N = len(points)
    
    if N < 4:
        N = N + 1 if add_cls_token else N
        attention_mask = torch.zeros((max_length, max_length), dtype=torch.bool)
        attention_mask[:N, :N] = True
        return attention_mask
    # 构建 Delaunay 三角剖分
    tri = Delaunay(points)
    
    # 构建邻接字典
    adjacency = defaultdict(set)
    for triangle in tri.simplices:
        for i in range(3):
            a = triangle[i]
            b = triangle[(i + 1) % 3]
            adjacency[a].add(b)
            adjacency[b].add(a)
    
    # 初始化掩码
    size = N + 1 if add_cls_token else N
        
    attention_mask = torch.zeros((max_length, max_length), dtype=torch.bool)
    
    for i in range(size):
        attention_mask[i, i] = True  # 自注意力
        
        start_i = i + 1 if add_cls_token else i
        for j in adjacency[start_i]:
            start_j = j + 1 if add_cls_token else j
            attention_mask[start_i, start_j] = True
    
    if add_cls_token:
        # [CLS] 可看到所有字符，所有字符也可看到 [CLS]
        attention_mask[0, 0] = True
        attention_mask[0, 1: size] = True
        attention_mask[1: size, 0] = True

    return attention_mask


class ObcReadingOrderDataset(Dataset):
    """
    支持结构化阅读顺序任务的数据集加载，包括：
    - 多句划分（用于对比学习）
    - 每个 token 的位置（用于句内排序）
    - 整体句子数量预测（K 值）
    """

    def __init__(self, data_dir: str, tokenizer: BertTokenizer, enhance_random_char_p: float = 0.4) -> None:
        super().__init__()
        with open(data_dir, "r", encoding="utf-8") as f:
            self.datasets = json.load(f)
        self.tokenizer = tokenizer
        self.enhance_random_char_p = enhance_random_char_p
        self.vocab = list(self.tokenizer.vocab)
        self.seq_max_length = 100

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        entry = self.datasets[index]
        lines = entry["line"]           # [[char1, char2, ...], [char1, char2, ...], ...]
        boxes = entry["box"]            # [[box1, box2, ...], [box1, box2, ...], ...]
        image_w = entry["width"]
        image_h = entry["height"]
        name = entry["name"]

        all_tokens = []
        all_positions = []
        sentence_ids = []
        order_within_sent = []

        sent_count = len(lines)
        
        all_positions.append([0, 0, 0, 0])
        for sent_id, (line, line_boxes) in enumerate(zip(lines, boxes)):
            for order_in_sent, (char, box_str) in enumerate(zip(line, line_boxes)):
                if char is None:
                    char = "[UNK]"
                
                if random.random() < self.enhance_random_char_p:
                    # 随机替换一个字符, 用于模拟目标检测识别错误
                    char = random.choice(self.vocab)
                
                if random.random() < self.enhance_random_char_p:
                    # 用于模拟目标检测识别错误
                    char = "[UNK]"
                
                box = list(map(int, box_str.split(",")))
                x0, y0, w, h = box
                x_center = x0 + w / 2
                y_center = y0 + h / 2
                
                # 添加随机偏移
                x_center += random.uniform(-w * 0.1, w * 0.1)
                y_center += random.uniform(-h * 0.1, h * 0.1)
                
                norm_box = [x_center / image_w, y_center / image_h, w / image_w, h / image_h]
                all_tokens.append(char)
                all_positions.append(norm_box)
                sentence_ids.append(sent_id)
                order_within_sent.append(order_in_sent)

        # 打乱
        token_ids = [i + 1 for i in range(len(all_tokens))]
        random.shuffle(token_ids)
        token_ids_cls = [0] + token_ids
        
        
        
        # tokenize
        encoding = self.tokenizer(all_tokens, is_split_into_words=True,
                                  return_attention_mask=True,
                                  return_token_type_ids=False,
                                  return_tensors="pt")
        
        
        pos_tensor = torch.tensor(all_positions, dtype=torch.float32)[token_ids_cls]
        sent_ids_tensor = torch.tensor(sentence_ids, dtype=torch.long)[[i - 1 for i in token_ids]]
        order_tensor = torch.tensor(order_within_sent, dtype=torch.long)[[i - 1 for i in token_ids]]
    

        return {
            "input_ids": encoding["input_ids"].squeeze(0)[:-1][token_ids_cls], #去除掉 [SEP]
            "position_tensor": pos_tensor,
            "sentence_ids": sent_ids_tensor,
            "order_ids": order_tensor,
            "K": sent_count,
            "name": name
        }
    def padding(self, values, max_length, value):
        seq_len = values.size(0)
        if seq_len >= max_length:
            # 如果长度已经够长，直接截断
            return values[:max_length]
        
        # 计算需要补齐的长度
        pad_len = max_length - seq_len

        # 生成补齐部分的 tensor，维度与 values 除第一个维度外一致
        # shape 为 (pad_len, ...)
        pad_shape = (pad_len,) + values.shape[1:]
        pad_tensor = torch.full(pad_shape, fill_value=value, dtype=values.dtype, device=values.device)

        # 在第一个维度拼接
        padded_values = torch.cat([values, pad_tensor], dim=0)

        return padded_values


    def collate_fn(self, batch):
        pad_token_id = self.tokenizer.get_added_vocab()['[PAD]']
        max_length = max([item["input_ids"].shape[0] for item in batch])
        
        input_ids = torch.stack([self.padding(item["input_ids"], max_length, pad_token_id) for item in batch])
        # attention_mask = torch.stack([self.padding(item["attention_mask"], max_length, 0) for item in batch])
        attention_mask = [build_neighbor_attention_mask(item["position_tensor"], add_cls_token=True, max_length=max_length) for item in batch]
        attention_mask = torch.stack(attention_mask)
        position_tensor = torch.stack([self.padding(torch.ones_like(item["position_tensor"]), max_length, 0) for item in batch])
        
        max_length = max_length - 1
        sentence_ids = torch.stack([self.padding(item["sentence_ids"], max_length, self.seq_max_length) for item in batch])
        order_ids = torch.stack([self.padding(item["order_ids"], max_length, -1) for item in batch])
        Ks = torch.tensor([item["K"] for item in batch], dtype=torch.long)
        names = [item["name"] for item in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "positions": position_tensor,
            "sentence_ids": sentence_ids,
            "order_ids": order_ids,
            "K": Ks,
            "names": names
        }



if __name__ == "__main__":
    
    bert_tokenizer = BertTokenizer.from_pretrained("oracle_bert")
    bert_tokenizer: BertTokenizer
    
    dataset = ObcReadingOrderDataset(
        tokenizer = bert_tokenizer,
        data_dir = "datasets/train_dataset.json"
    )
    
    print(len(dataset))
        
    obc_train_loader = DataLoader(dataset, batch_size = 16, shuffle = True, collate_fn = dataset.collate_fn)
    
    for index, (batch, mask_label, _) in enumerate(obc_train_loader):
        output = batch["input_ids"][batch["input_ids"] == bert_tokenizer.mask_token_id]
        
        print(output.shape)
        print(mask_label.shape)
        print(bert_tokenizer.batch_decode(batch["input_ids"]))
        print(batch["position_ids"])
        print(mask_label)
        break
        