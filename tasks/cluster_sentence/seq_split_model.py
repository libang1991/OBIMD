from transformers import BertConfig, BertModel, BertTokenizer
# from transformers.models.
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import Optional

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, F_dim: int, H_dim: int, gamma: float):
        super().__init__()
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2) 
        self.Wr = nn.Linear(self.M // 2, self.F_dim // 2, bias=False)
        
        self.Ws = nn.Sequential(
            nn.Linear(self.M // 2, self.F_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.F_dim, self.F_dim // 2)
        )
        
        
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.F_dim)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        
        B, N, D = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x[:, :, :self.M // 2])
        scale_projected = self.Ws(x[:, :, self.M // 2:])
        projected = projected * scale_projected
        
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        PEx = self.mlp(F)
        return PEx

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, point_dim: int):
        super().__init__()
        
        # token 级别测embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)        
        
        self.position_embeddings = LearnableFourierPositionalEncoding(
            point_dim, config.hidden_size, config.hidden_size // 2, gamma = 10)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = position_embeddings + inputs_embeds
    
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SeqProcessMlModel(nn.Module):
    def __init__(self, bert_model_name, point_dim: int, max_legnth: int, seq_max_length: int):
        super(SeqProcessMlModel, self).__init__()
        # 构建一个 bert 模型
        bert_config = BertConfig.from_pretrained(bert_model_name)
        self.model = BertModel(bert_config)
        
        # 用于 token 级别的 embedding
        self.model.embeddings = BertEmbeddings(bert_config, point_dim)
        
        # 句子级别的 embedding
        self.seq_embeddings = nn.Embedding(seq_max_length, bert_config.hidden_size)

        # 初始化
        self.seq_embeddings.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
        self.model.embeddings.word_embeddings.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
        
        # 构建交叉注意力
        self.decode_layer = nn.TransformerDecoderLayer(
            d_model = bert_config.hidden_size,
            nhead = bert_config.num_attention_heads,
            dim_feedforward = bert_config.intermediate_size,
            dropout = bert_config.attention_probs_dropout_prob,
            activation = bert_config.hidden_act,
            layer_norm_eps = bert_config.layer_norm_eps,
            batch_first = True
        )
        self.decodes = nn.TransformerDecoder(self.decode_layer, bert_config.num_hidden_layers)
        
        self.seq_mlp = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(bert_config.hidden_size // 2, bert_config.hidden_size)
        )
        
        self.pred_k_mlp = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(bert_config.hidden_size // 2, seq_max_length)
        )
        
        self.sort_mlp = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(bert_config.hidden_size // 2, max_legnth)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, bert_config.hidden_size), requires_grad=True)

    
    def forward(self, inputs, seq_ids):
        
        # 分句子
        bert_pred = self.model(**inputs)
        memory = bert_pred["last_hidden_state"]

        seq_logic = self.seq_mlp(memory[:, 1:, :])

        
        # 根据分好的句号进行排序
        seq_embedding = self.seq_embeddings(seq_ids)
        mask = ~inputs['attention_mask'].bool()

        sort_hidden_state = self.decodes(seq_embedding, memory)
        sort_logic = self.sort_mlp(sort_hidden_state)

        # 预测有几个句子
        k_pred = self.pred_k_mlp(bert_pred['pooler_output'])
        
        return seq_logic, sort_logic, k_pred

    def cluster_pred(self, inputs):
         # 分句子
        bert_pred = self.model(**inputs)
        memory = bert_pred["last_hidden_state"]

        seq_logic = self.seq_mlp(memory[:, 1:, :])
        
        k_pred = self.pred_k_mlp(bert_pred['pooler_output'])
        
        return F.normalize(seq_logic, dim=-1), k_pred.max(-1)[1]

        
        

if __name__ == "__main__":
    
    model = SeqProcessMlModel("oracle_bert", 4, 100, seq_max_length=20)
    
    
    loss_func = nn.CrossEntropyLoss()
    
    batch = {}
    batch["input_ids"] = torch.randint(0, 1000, size = (16, 18), dtype = torch.int32)
    batch["token_type_ids"] = torch.zeros((16, 18), dtype = torch.int32)
    batch["attention_mask"] = torch.ones((16, 18), dtype = torch.int32)
    batch["position_ids"] = torch.randn((16, 18, 4))


    seq = torch.zeros(size = (16, 18), dtype = torch.int32)
    
    seq_logic, sort_logic, k_pred = model(batch, seq)
    print(seq_logic.mean(-1))
    print(seq_logic.shape)
    print(sort_logic.shape)
    print(k_pred.shape)