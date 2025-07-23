import numpy as np
from sklearn.model_selection import train_test_split
import torch

def generate_data(num_samples, seq_len, vocab_size):
    # 随机生成整数序列，每个整数范围在 [1, vocab_size-1] 内，保留 0 作为 padding 的位置
    data = np.random.randint(1, vocab_size, size=(num_samples, seq_len))
    return data

def prepare_data(num_samples, seq_len, vocab_size, random_state, test_ratio=0.2):
    total_data = generate_data(num_samples, seq_len, vocab_size)
    train_data, test_data = train_test_split(total_data, test_size=test_ratio, random_state=random_state)
    train_input = torch.LongTensor(train_data)
    train_target = torch.LongTensor(train_data)
    test_input = torch.LongTensor(test_data)
    test_target = torch.LongTensor(test_data)
    return train_input, train_target, test_input, test_target