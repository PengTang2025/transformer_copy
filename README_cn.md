# Transformer Copy Task with Attention Visualization

本项目基于一篇讲解 Transformer 原理的文章及其附带的示例代码（实现了 Copy Task 与注意力可视化），在其基础上进行了**深度重构与增强**。最终呈现的项目结果与可视化效果如下：

<img width="1607" height="885" alt="image" src="https://github.com/user-attachments/assets/432a00c6-6d73-4641-a82d-76c6c215f984" />


## 🔧 项目改进内容

1. **逻辑梳理与函数封装**  
   将原始代码中分散、混乱的逻辑进行了系统性整理，完成了函数化和模块化，使整体流程清晰易读，便于复用与扩展：
   - `dataset.py`：生成用于 copy task 的训练与验证数据
   - `coderlayer_with_attn.py`：自定义 Encoder 和 Decoder 层以提取真实注意力权重
   - `transformer.py`:应用自定义的Encoder实现TransformerCopyModel
   - `train.py`：训练主逻辑
   - `visualize.py`：可视化函数
   - `main.py`：完整的训练 + 可视化流程执行文件

3. **数据生成流程规范化**  
   优化了 Copy Task 所用数据集的生成方式：原版代码训练集与测试集数据由分别调用generate_data函数生成；更新后，训练集与测试集由单次调用generate_data函数生成原始数据，再随机分割产生。
   此外，还更新了训练集与测试集的比例。
   通过优化使其更符合科学研究规范。
    ```
    # before
    def generate_data(num_samples, seq_len, vocab_size):
        # 随机生成整数序列，每个整数范围在 [1, vocab_size-1] 内，保留 0 作为 padding 的位置
        data = np.random.randint(1, vocab_size, size=(num_samples, seq_len))
        return data
    # 生成训练和测试数据
    train_data = generate_data(num_samples, seq_len, vocab_size)
    test_data = generate_data(200, seq_len, vocab_size)  # 测试样本数较少
    # 将 numpy 数组转换为 tensor
    train_input = torch.LongTensor(train_data)
    train_target = torch.LongTensor(train_data)  # 复制任务：目标与输入一致
    test_input = torch.LongTensor(test_data)
    test_target = torch.LongTensor(test_data)
    ```
    ```
    # after
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
    ```
3. **可视化美化与修正**  
   对原始 attention 可视化中存在的 label 重叠问题进行了调整与修复，图像排版更合理、标注更清晰。
    ```
    # before: tight_layout方法
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ```
    ```
    # after: constrained_layout方法
    plt.figure(figsize=(18, 12), constrained_layout=True)
    ```
    效果：before(左)→after(右)
    <p align="center">
      <img src="https://github.com/user-attachments/assets/2051e636-0a24-4fb3-8209-ebb060af2b15" width="45%"/>
      <img src="https://github.com/user-attachments/assets/40b1678a-a7b0-41d5-8047-eb46639acdbc" width="45%"/>
    </p>

4. **真实注意力权重提取与展示**  
   最重要的一点：原始项目中展示的注意力图仅是**训练外**调用 `nn.MultiheadAttention` 层生成的模拟数据，**并非模型实际 forward 中的 attention weights**。  
   本项目通过自定义 Transformer 层（coderlayer_with_attn.py)，成功从模型内部提取真正的注意力矩阵，实现了更具解释性和分析价值的可视化。
   此外，鉴于最终结果组图中其余三张图都是按照笛卡尔坐标系呈现，在本次改进中，也将注意力可视化热图由原本的图片坐标系（原点在左上角，x轴：key(j), y轴：query(i)）改为笛卡尔坐标系（原点在左下角，x轴：query(i), y轴：key(j)）展示,使得query(i)与key(j)之间的注意力关系更加直观。
    ```
    # before
    # 为了获取注意力权重，我们重新构造一个包含 MultiheadAttention 的模块
    # 注意：由于 nn.TransformerEncoderLayer 内部没有直接返回注意力权重，我们这里单独使用 nn.MultiheadAttention 来模拟其中一层注意力
    # 选取 sample_input 的嵌入表示作为查询、键、值，要求 batch_first=True
    emb_sample = model.embedding(sample_input) * np.sqrt(model.d_model)
    emb_sample = model.pos_encoder(emb_sample)  # shape: (1, seq_len, d_model)
    multihead_attn = nn.MultiheadAttention(embed_dim=model.d_model, num_heads=8, batch_first=True)
    # 通过 forward 时设置 need_weights=True 得到注意力权重
    attn_output, attn_weights = multihead_attn(emb_sample, emb_sample, emb_sample, need_weights=True, average_attn_weights=False)
    ```
    ```
    # after
    _ = model(sample_input)  # 触发 forward，计算注意力权重
    attn_weights = model.last_attn  # 获取最后一层的注意力权重
    ```
    坐标系更新：before(左)→after(右)
    <p align="center">
      <img src="https://github.com/user-attachments/assets/2a9a1991-c066-43f1-8481-1ea2dadaa4c1" width="45%"/>
      <img src="https://github.com/user-attachments/assets/c5833b8c-0d4b-4155-b496-02fda99d11e1" width="45%"/>
    </p>
    注意力逻辑更新：before(左)→after(右)
    <p align="center">
      <img src="https://github.com/user-attachments/assets/be57481b-f7b6-48b5-bee2-c39420828e5a" width="45%"/>
      <img src="https://github.com/user-attachments/assets/629729a5-9c01-4b7d-a9ac-5cfbc4fa8188" width="45%"/>
    </p>

## 🙏 致谢与引用

本项目参考并基于微信公众号「机器学习初学者」发布的文章  
[《【论文复现】从零实现Transformer，并可视化Attention！》](https://mp.weixin.qq.com/s/BCECx-0C9E_wY4ZyRrZ5uQ) 中的教学示例代码，  
在其基础上进行了结构重构、attention 权重提取方式优化、可视化增强等改进，致谢原作者的分享。

## 📜 License

MIT License © 2025 PengTang
