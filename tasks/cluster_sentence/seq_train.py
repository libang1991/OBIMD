from obc_dataset_seq import ObcReadingOrderDataset, DataLoader
from seq_split_model import SeqProcessMlModel
from transformers import BertTokenizer
import json
import os
import glob
import torch.nn as nn
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F



def cluster_loss(x, labels, ignore_id=-1, intra_weight=1.0, inter_weight=0.1, margin=1.0):
    """
    计算聚类损失函数，包含类内距离和类间距离的约束。

    参数：
    x (Tensor): 输入的特征张量，形状为(B, SEQ_LEN, DIM)
    labels (Tensor): 标签张量，形状为(B, SEQ_LEN)，ignore_id表示忽略的标签
    ignore_id (int): 需要忽略的标签值，默认为-1
    intra_weight (float): 类内损失的权重
    inter_weight (float): 类间损失的权重
    margin (float): 类间距离的最小间隔，小于此值将受到惩罚

    返回：
    Tensor: 计算得到的损失值
    """
    B = x.size(0)
    total_intra_loss = 0.0
    total_inter_loss = 0.0
    num_samples_intra = 0
    num_samples_inter = 0

    for b in range(B):
        # 获取当前样本的有效元素和标签
        mask = labels[b] != ignore_id
        x_b = x[b][mask]  # (n_valid, DIM)
        labels_b = labels[b][mask]  # (n_valid,)
        n_valid = x_b.size(0)
        if n_valid == 0:
            continue  # 跳过无有效元素的样本

        # 获取唯一标签并计算每个标签的中心点
        unique_labels = torch.unique(labels_b)
        num_labels = unique_labels.size(0)
        if num_labels == 0:
            continue

        centers = []
        for label in unique_labels:
            mask_label = (labels_b == label)
            if mask_label.sum() == 0:
                continue
            center = x_b[mask_label].mean(dim=0)
            centers.append(center)
        if not centers:
            continue
        centers = torch.stack(centers)  # (num_centers, DIM)

        # 计算类内损失：每个元素到对应中心的距离平方的均值
        # 将标签映射到中心点的索引
        idx = torch.bucketize(labels_b, torch.sort(unique_labels).values)
        expanded_centers = centers[idx]
        intra_loss = torch.mean(torch.sum((x_b - expanded_centers) ** 2, dim=1))
        total_intra_loss += intra_loss
        num_samples_intra += 1

        # 计算类间损失：中心点间距离小于margin的惩罚项
        if len(centers) >= 2:
            dist_matrix = torch.cdist(centers.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
            triu_indices = torch.triu_indices(len(centers), len(centers), offset=1)
            pairwise_dists = dist_matrix[triu_indices[0], triu_indices[1]]
            hinge = torch.clamp(margin - pairwise_dists, min=0)
            inter_loss = torch.mean(hinge ** 2)
            total_inter_loss += inter_loss
            num_samples_inter += 1

    # 计算平均损失
    avg_intra = total_intra_loss / num_samples_intra if num_samples_intra > 0 else 0.0
    avg_inter = total_inter_loss / num_samples_inter if num_samples_inter > 0 else 0.0

    total_loss = intra_weight * avg_intra + inter_weight * avg_inter
    return total_loss

def improved_cluster_loss(x, labels, ignore_id=-1, intra_weight=1.0, inter_weight=0.1):
    """
    改进版聚类损失函数，增强类内紧凑性与类间分离性。

    参数：
        x (Tensor): 特征张量，形状为 (B, SEQ_LEN, DIM)
        labels (Tensor): 标签张量，形状为 (B, SEQ_LEN)，ignore_id 表示忽略标签
        ignore_id (int): 被忽略的标签值
        intra_weight (float): 类内紧凑损失权重
        inter_weight (float): 类间推远损失权重

    返回：
        Tensor: 聚类损失
    """
    B, _, D = x.shape
    total_intra_loss = 0.0
    total_inter_loss = 0.0
    intra_count = 0
    inter_count = 0

    for b in range(B):
        mask = labels[b] != ignore_id
        x_b = x[b][mask]         # (n_valid, D)
        labels_b = labels[b][mask]
        if x_b.size(0) < 2:
            continue

        # L2 归一化特征向量
        x_b = F.normalize(x_b, p=2, dim=1)

        unique_labels = torch.unique(labels_b)
        centers = []
        label_to_indices = {}

        for label in unique_labels:
            idx = (labels_b == label).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() < 2:
                continue  # 至少两个点才有 pairwise 距离
            label_to_indices[label.item()] = idx
            centers.append(x_b[idx].mean(dim=0))

        if not centers:
            continue

        centers = torch.stack(centers)
        centers = F.normalize(centers, p=2, dim=1)

        # === 类内损失：平均所有类内点对的距离平方 ===
        intra_loss = 0.0
        for label, idx in label_to_indices.items():
            points = x_b[idx]  # (n, D)
            if points.size(0) < 2:
                continue
            dists = torch.pdist(points, p=2)  # (n*(n-1)/2,)
            intra_loss += torch.mean(dists ** 2)
            intra_count += 1
        if intra_count > 0:
            total_intra_loss += intra_loss

        # === 类间损失：中心点对 exp(-距离) 惩罚 ===
        if centers.size(0) >= 2:
            pairwise_center_dist = torch.pdist(centers, p=2)
            inter_loss = torch.mean(torch.exp(-pairwise_center_dist))  # 推远中心
            total_inter_loss += inter_loss
            inter_count += 1

    avg_intra = total_intra_loss / intra_count if intra_count > 0 else 0.0
    avg_inter = total_inter_loss / inter_count if inter_count > 0 else 0.0

    total_loss = intra_weight * avg_intra + inter_weight * avg_inter
    return total_loss


def grouped_cross_entropy_loss(item_ids, item_pred, sentence_id, ignore_id=-1):
    """
    对 item_pred 和 item_ids 根据 sentence_id 分组，并计算每组的交叉熵损失。

    参数:
    - item_ids: [S] LongTensor，真实标签
    - item_pred: [S, C] FloatTensor，模型预测的 logits
    - sentence_id: [S] LongTensor，句子或段落分组编号
    - ignore_id: int，表示需要忽略的 sentence_id 值

    返回:
    - total_loss: 标量 tensor，所有有效组的平均交叉熵损失
    """

    unique_ids = sentence_id.unique()
    total_loss = 0.0
    count = 0

    for sid in unique_ids:
        if sid.item() == ignore_id:
            continue
        # 获取当前组的 mask
        mask = (sentence_id == sid)
        group_preds = item_pred[mask]      # [n_i, C]
        group_labels = item_ids[mask]      # [n_i]
        group_preds = group_preds

        loss = F.cross_entropy(group_preds, group_labels)
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=item_pred.device)
    
    return total_loss / count



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 10000
    
    bert_tokenizer = BertTokenizer.from_pretrained("oracle_bert_sub")
    
    dataset = ObcReadingOrderDataset(
        tokenizer=bert_tokenizer,
        data_dir="datasets_sub/train_dataset.json"
    )
    
    # 4 表示 x0, y0, x1, y1
    model = SeqProcessMlModel("oracle_bert_sub", 4, 100, dataset.seq_max_length + 1).to(device)
    model = nn.DataParallel(model, device_ids = [0, 1, 2, 3, 4, 5, 6, 7], output_device = 0)
    
    # TensorBoard logger
    log_dir = f"logs_SEQ/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Modify checkpoint directory based on dataset class name
    checkpoint_dir = f"checkpoints_exp_sub/{dataset.__class__.__name__}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(f"{checkpoint_dir}/best.pt"):
        model.load_state_dict(torch.load(f"{checkpoint_dir}/last_split.pt"))
    
    print(len(dataset))
    
    obc_train_loader = DataLoader(
        dataset,
        batch_size = 32 * 8,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers = 8
    )
    
    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    shcedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2, eta_min = 1e-7)
    
    best_loss = float("inf")
    
    for epoch in range(epochs):
        tbar_loader = tqdm.tqdm(obc_train_loader, desc="Epoch: {}".format(epoch))
        avg_losses = []
        model.train()
        
        for index, batch_input in enumerate(tbar_loader):
            # 检测这批数据中是否存在 mask的标签
            # 总体损失函数 loss_total = loss_bert + loss_translate
            
            # 将数据放到GPU上
            batch_input = {k: v.to(device) for k, v in batch_input.items() if isinstance(v, torch.Tensor)}
            
            inputs = {
                "input_ids": batch_input['input_ids'],
                "attention_mask": batch_input['attention_mask'],
                "position_ids": batch_input['positions'],
            }
            
            k_gt = batch_input['K']
            order_ids = batch_input['order_ids']
            sentence_ids = batch_input['sentence_ids']
            
            # 前向传播
            seq_logic, sort_logic, k_pred = model(inputs, sentence_ids)
            # B, S, D  B, S, C B, C
            
            loss = 0.0
            
            k_loss = loss_func(k_pred, k_gt)
            seq_loss_loss = cluster_loss(seq_logic, sentence_ids, dataset.seq_max_length)
            
            sort_loss = 0
            for item_ids, item_pred, sentence_id in zip(order_ids, sort_logic, sentence_ids):
                sort_loss += grouped_cross_entropy_loss(item_ids, item_pred, sentence_id, dataset.seq_max_length)
            sort_loss /= order_ids.size()[0]

            loss = seq_loss_loss * 5
            
            # 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_losses.append(loss.item())
            
            
            _, k_pred_labels = torch.max(k_pred, dim=1)
            k_acc = (k_pred_labels == k_gt).float().mean()
            sort_acc = 0
            for item_ids, item_pred in zip(order_ids, sort_logic):
                vailid_ids = item_ids[item_ids != dataset.seq_max_length]
                pred_labels = item_pred[vailid_ids]
                item_ids = item_ids[vailid_ids]
                
                sort_acc += (torch.max(pred_labels, dim=1)[0] == item_ids).float().mean()
                
            sort_acc /= order_ids.size(0)



            tbar_loader.set_postfix({
                "loss": loss.item(),
                "avg_loss": sum(avg_losses) / len(avg_losses),
                "seq_loss_loss":  seq_loss_loss.item(),
                "sort_loss": sort_loss.item(),
                "k_loss": k_loss.item(),
                "sort_acc": sort_acc.item(),
                "k_acc": k_acc.item(),
                "lr": shcedule.get_last_lr()
            })
            
            
        # TensorBoard logging
        avg_loss = sum(avg_losses) / len(avg_losses)
        writer.add_scalar('Loss/avg_loss', avg_loss, epoch)
        writer.add_scalar('LearningRate', shcedule.get_last_lr()[0], epoch)
        
        shcedule.step()
            
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best.pt")
        torch.save(model.state_dict(), f"{checkpoint_dir}/last.pt")
    
    # Close the TensorBoard writer after training
    writer.close()
