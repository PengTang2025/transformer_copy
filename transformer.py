import torch
import torch.nn as nn
import numpy as np

# default: d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1

class PositionalEncoding(nn.Module):
    # 是transformer调用的函数，而不是数据预处理
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 预先计算所有位置的编码，存放在buffer里，只在初始化计算一次，不随反向传播更新
        pe = torch.zeros(max_len, d_model)
        # arange: 生成一个从 0 到 max_len-1 的一维张量
        # unsqueeze(1): 在第二维增加一个维度，变为 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维使用 cos
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的 shape 为 (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # 广播机制，节省空间
        x = x + self.pe[:, :seq_len]
        return x

class TransformerCopyModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        # d_model：你希望模型用多大的空间来“想象”和“记忆”每个 token
        super(TransformerCopyModel, self).__init__()
        self.d_model = d_model
        # 输入嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        # 单层编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        # num_layers层编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出线性映射层
        self.fc_out = nn.Linear(d_model, vocab_size)
        # # 为了方便后续提取注意力权重，我们保存最后一层 encoder 的注意力
        # self.last_attn = None

    def forward(self, src):
        # src shape: (batch_size, seq_len)
        # 论文中的 trick，让嵌入和位置编码的量级相近，不至于“位置编码淹没嵌入信息”
        emb = self.embedding(src) * np.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        # Transformer 编码器输出
        out = self.transformer_encoder(emb)
        # # 保存最后一层编码器的输出作为隐层特征
        # self.last_hidden = out
        # 输出层映射到词汇表维度
        out = self.fc_out(out) 
        # shape: (batch_size, seq_len, vocab_size)
        # 其中第三维是每个token在词汇表中每个位置 的logits，softmax后为概率
        return out