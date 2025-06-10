# model.py
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from sortdata1 import CharOrderDataset, SimpleCharIDTokenizer 
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
class PositionalCharModel(nn.Module):
    def __init__(self, vocab_size, char_dim=64, pos_dim=64, hidden_dim=128, max_pos=1000):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_dim)
        self.posv_embedding = nn.Embedding(max_pos, pos_dim)
        self.posh_embedding = nn.Embedding(max_pos, pos_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=char_dim + 2 * pos_dim, nhead=8),
            num_layers=2
        )

        self.classifier = nn.Linear(char_dim + 2 * pos_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 40)  # 假设最多字符数为 40，用于分类到原始位置

    def forward(self, input_ids, posv, posh, attention_mask=None):
        char_emb = self.char_embedding(input_ids)   # [B, T, char_dim]
        posv_emb = self.posv_embedding(posv)        # [B, T, pos_dim]
        posh_emb = self.posh_embedding(posh)        # [B, T, pos_dim]

        x = torch.cat([char_emb, posv_emb, posh_emb], dim=-1)  # [B, T, D]

        x = x.transpose(0, 1)  # Transformer 需要 [T, B, D]
        x = self.transformer(x, src_key_padding_mask=(attention_mask == 0))
        x = x.transpose(0, 1)  # [B, T, D]

        hidden = self.classifier(x)
        logits = self.out(hidden)  # [B, T, num_classes]

        return logits
class KernelAttention2D(nn.Module):
    def __init__(self, d_model, sigma=16.0):
        super().__init__()
        self.sigma = sigma
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, posv=None, posh=None):
        if posv is None or posh is None:
            return torch.zeros_like(x)

        B, T, D = x.shape
        pos = torch.stack([posv, posh], dim=-1).float()  # [B, T, 2]
        pos_diff = pos[:, :, None, :] - pos[:, None, :, :]  # [B, T, T, 2]
        dist_sq = (pos_diff ** 2).sum(dim=-1)  # [B, T, T]

        kernel_weight = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # [B, T, T]
        kernel_weight = kernel_weight / kernel_weight.sum(dim=-1, keepdim=True)
        kernel_weight = self.dropout(kernel_weight)

        return kernel_weight @ x


class HybridBlock(nn.Module):
    def __init__(self, d_model, nhead, use_kernel_attn=True):
        super().__init__()
        self.use_kernel_attn = use_kernel_attn

        self.std_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.kernel_attn = KernelAttention2D(d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(0.1)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

        if use_kernel_attn:
            self.gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("gate", torch.tensor(1.0))

    def forward(self, x, posv=None, posh=None, attn_mask=None):
        residual = x
        x = self.norm1(x)

        std_out, _ = self.std_attn(x, x, x, key_padding_mask=attn_mask)
        kernel_out = self.kernel_attn(x, posv, posh)
        g = torch.sigmoid(self.gate)

        x = residual + g * self.dropout(std_out) + (1 - g) * kernel_out

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class HybridTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=2, use_kernel_attn=True):
        super().__init__()
        self.use_kernel_attn = use_kernel_attn
        self.layers = nn.ModuleList([
            HybridBlock(d_model, nhead, use_kernel_attn)
            for _ in range(num_layers)
        ])

    def forward(self, x, posv=None, posh=None, attn_mask=None):
        for layer in self.layers:
            x = layer(x, posv, posh, attn_mask)
        return x


# # 插值核函数
# def rbf_kernel(pos1, pos2, gamma=0.1):
#     diff = pos1[:, None, :, :] - pos2[:, :, None, :]  # [B, T, T, 2]
#     dist2 = (diff ** 2).sum(dim=-1)  # [B, T, T]
#     return torch.exp(-gamma * dist2)

# # 混合注意力模块
# class HybridAttention(nn.Module):
#     def __init__(self, d_model, nhead):
#         super().__init__()
#         self.norm = nn.LayerNorm(d_model)
#         self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
#         self.gate_param = nn.Parameter(torch.tensor(0.5))
#     def forward(self, x, posv=None, posh=None, mask=None):
#         x = self.norm(x)

#         if posv is None or posh is None:
#             out, _ = self.attn(x, x, x, key_padding_mask=mask)
#             return out

#         # 构造 2D 坐标
#         coord = torch.stack([posv.float(), posh.float()], dim=-1)  # [B, T, 2]
#         sim = rbf_kernel(coord, coord)  # [B, T, T]

#         # 掩码处理（用于 padding）
#         if mask is not None:
#             sim = sim.masked_fill(mask[:, None, :], -1e9)  # broadcasting: [B, T, T] * [B, 1, T]

#         sim_weights = torch.softmax(sim, dim=-1)  # [B, T, T]
#         # gate = torch.sigmoid(self.gate_param)[None, :, None]
#         gate = torch.sigmoid(self.gate_param)  # 标量或向量
#         out = gate * x + (1 - gate) * (sim_weights @ x)
#         # out = sim_weights @ x  # [B, T, D]
#         return out

# # 混合 Transformer Encoder（多层）
# class HybridTransformerEncoder(nn.Module):
#     def __init__(self, d_model, nhead=8, num_layers=2):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             HybridAttention(d_model, nhead) for _ in range(num_layers)
#         ])
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x, posv=None, posh=None, mask=None):
#         for layer in self.layers:
#             x = x + layer(x, posv, posh, mask)
#         return self.norm(x)

# 主模型
class PositionalCharModel1(nn.Module):
    def __init__(self, vocab_size, char_dim=64, pos_dim=64, hidden_dim=128, max_pos=1000, use_coord_attention=True, use_kernel_attn=False):
        super().__init__()
        self.use_coord_attention = use_coord_attention
        self.use_kernel_attn = use_kernel_attn
        self.char_embedding = nn.Embedding(vocab_size, char_dim)
        if use_coord_attention:
            self.posv_embedding = nn.Embedding(max_pos, pos_dim)
            self.posh_embedding = nn.Embedding(max_pos, pos_dim)
            d_model = char_dim + 2 * pos_dim
        else:
            d_model = char_dim

        self.transformer = HybridTransformerEncoder(
            d_model=d_model,
            nhead=8,
            num_layers=2,
            use_kernel_attn= self.use_kernel_attn
            
        )

        self.classifier = nn.Linear(d_model, hidden_dim)
        self.out = nn.Linear(hidden_dim, 40)  # 假设最多字符数为 40

    
    
    def forward(self, input_ids, posv=None, posh=None, attention_mask=None):
        char_emb = self.char_embedding(input_ids)  # [B, T, char_dim]

        if self.use_coord_attention and posv is not None and posh is not None:
            posv_emb = self.posv_embedding(posv)    # [B, T, pos_dim]
            posh_emb = self.posh_embedding(posh)    # [B, T, pos_dim]
            x = torch.cat([char_emb, posv_emb, posh_emb], dim=-1)  # [B, T, D]
        else:
            x = char_emb  # [B, T, char_dim]

        if attention_mask is not None:
            mask = (attention_mask == 0)
        else:
            mask = None

        x = self.transformer(x, posv, posh, mask)  # [B, T, D]
        hidden = self.classifier(x)
        logits = self.out(hidden)  # [B, T, 40]
        return logits
if __name__ == '__main__':
    with open("datasets/sorttrain.json", "r") as f:
        raw_data = json.load(f)
    tokenizer = SimpleCharIDTokenizer(vocab_file="datasets/vacabsort.txt")

    dataset = CharOrderDataset(raw_data, tokenizer, max_len=40, use_order_pos=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    model = PositionalCharModel1(
    vocab_size=2750,
    char_dim=64,
    pos_dim=64,
    hidden_dim=128,
    max_pos=1111,
    use_coord_attention=False
    )
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].cuda()
        input_posv = batch["posv"].cuda()
        input_posh = batch["posh"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        outputs = model(input_ids, input_posv, input_posv, attention_mask)  # [B, T, vocab] or [B, T, T]
        break