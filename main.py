import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import random

from dataset import prepare_data
from transformer import TransformerCopyModel
from train import train
from visualize import visualize

def main(my_seed, vocab_size, seq_len, num_epochs, batch_size, learning_rate):
    
    torch.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取 Dataset
    train_input, train_target, test_input, test_target = prepare_data(num_samples, seq_len, vocab_size, my_seed)
    
    # 模型定义
    model = TransformerCopyModel(vocab_size=vocab_size, max_seq_len=seq_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    train_losses = train(model, train_input, train_target, criterion, optimizer, device, num_epochs, batch_size)

    # # 测试
    # model.eval()
    # with torch.no_grad():
    #     # 对测试集进行预测
    #     test_input = test_input.to(device)
    #     test_target = test_target.to(device)
    #     test_outputs = model(test_input)  # shape: (batch_size, seq_len, vocab_size)
    #     # 取每个时间步最大概率的 token 作为预测结果
    #     predictions = torch.argmax(test_outputs, dim=-1).cpu().numpy()

    # 可视化
    model.eval()
    visualize(model, train_losses, num_epochs, vocab_size, seq_len, test_input, test_target, device)
    
if __name__ == "__main__":
    
    # 参数
    my_seed = 42
    vocab_size = 50
    seq_len = 10
    num_samples = 10000
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001
      
    main(my_seed, vocab_size, seq_len, num_epochs, batch_size, learning_rate)
    
