import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import random

def plot_loss(train_losses, num_epochs):
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', color='red', linestyle='-', linewidth=2)
    plt.title("Training Loss Curve", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    pass

def plot_embeddings(model, vocab_size):
    # 可视化输入嵌入向量的分布 (使用 PCA 降维至 2 维)
    # 从 embedding 层中抽取嵌入矩阵，shape: (vocab_size, d_model)
    embeddings = model.embedding.weight.cpu().detach().numpy()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.subplot(2,2,2)
    scatter = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=np.arange(vocab_size), cmap="viridis", s=100)
    plt.title("Embedding Distribution (PCA)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.colorbar(scatter)
    pass

def plot_sample_prediction(seq_len, sample_input, sample_target, model):
    with torch.no_grad():
        sample_output = model(sample_input)
    sample_pred = torch.argmax(sample_output, dim=-1).cpu().numpy()[0]
    plt.plot(range(seq_len), sample_target, marker='o', color='blue', linestyle='-', label="Ground Truth")
    plt.plot(range(seq_len), sample_pred, marker='x', color='green', linestyle='--', label="Prediction")
    plt.title("Prediction vs Ground Truth", fontsize=14)
    plt.xlabel("Sequence Position", fontsize=12)
    plt.ylabel("Token ID", fontsize=12)
    plt.legend()
    plt.grid(True)
    pass

def plot_attention_heatmap(model, sample_input):
    # 为了获取注意力权重，我们重新构造一个包含 MultiheadAttention 的模块
    # 注意：由于 nn.TransformerEncoderLayer 内部没有直接返回注意力权重，我们这里单独使用 nn.MultiheadAttention 来模拟其中一层注意力
    # 选取 sample_input 的嵌入表示作为查询、键、值，要求 batch_first=True
    emb_sample = model.embedding(sample_input) * np.sqrt(model.d_model)
    emb_sample = model.pos_encoder(emb_sample)  # shape: (1, seq_len, d_model)
    multihead_attn = nn.MultiheadAttention(embed_dim=model.d_model, num_heads=8, batch_first=True)
    # 通过 forward 时设置 need_weights=True 得到注意力权重
    attn_output, attn_weights = multihead_attn(emb_sample, emb_sample, emb_sample, need_weights=True, average_attn_weights=False)
    # 此时 attn_weights 的形状为 (1, 8, 10, 10)
    attn_heatmap = attn_weights[0, 0].cpu().detach().numpy().T # 形状为 (10, 10)
    im = plt.imshow(attn_heatmap, cmap="plasma", aspect='auto')
    plt.gca().invert_yaxis()  # 让 y=0 出现在图底部，符合坐标系习惯
    plt.title("Attention Weight Heatmap", fontsize=14)
    plt.xlabel("Query Position(i)", fontsize=12)
    plt.ylabel("Key Position(j)", fontsize=12)
    plt.colorbar(im)
    pass

def visualize(model, train_losses, num_epochs, vocab_size, seq_len, test_input, test_target, device):
     # 可视化预测结果
    # 随机选择一个样本进行可视化
    sample_idx = random.randint(0, len(test_input)-1)
    sample_input = test_input[sample_idx].unsqueeze(0).to(device)  # shape: (1, seq_len)
    sample_true = test_target[sample_idx].numpy()
    
    plt.figure(figsize=(18, 12), constrained_layout=True)
    plt.subplot(2,2,1)
    # 绘制训练损失曲线
    plot_loss(train_losses, num_epochs)
    plt.subplot(2,2,2)
    # 可视化嵌入向量分布
    plot_embeddings(model, vocab_size)
    plt.subplot(2,2,3)
    # 绘制样本预测结果
    plot_sample_prediction(seq_len, sample_input, sample_true, model)
    plt.subplot(2,2,4)
    # 绘制注意力权重热图
    plot_attention_heatmap(model, sample_input)
    
    plt.suptitle("Transformer Input/Output Embedding and Linear Transformation Analysis", fontsize=16)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()