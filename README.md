# Transformer Copy Task with Attention Visualization

This project is a **thorough refactor and enhancement** based on a tutorial article about Transformer and its accompanying code that implemented the Copy Task with attention visualization. The final result, including the attention visualization output, is shown below:

<img width="1607" height="885" alt="image" src="https://github.com/user-attachments/assets/432a00c6-6d73-4641-a82d-76c6c215f984" />

## 🔧 Improvements & Enhancements

1. **Logical Refactoring and Modularization**  
   The original code was fragmented and unstructured. In this project, the logic has been reorganized into clean, reusable, and modular components:
   - `dataset.py`: Generates training and validation data for the copy task
   - `coderlayer_with_attn.py`: Custom encoder and decoder layers that expose real attention weights
   - `transformer.py`: Defines the TransformerCopyModel using custom encoder layers
   - `train.py`: Main training script
   - `visualize.py`: Visualization utilities
   - `main.py`: End-to-end training + visualization pipeline

2. **Standardized Data Generation Process**  
   Instead of separately generating training and testing data, this version first generates the full dataset once and then performs a randomized train/test split.  
   The train/test ratio has also been adjusted for better experimental rigor.

```python
# before
def generate_data(num_samples, seq_len, vocab_size):
    data = np.random.randint(1, vocab_size, size=(num_samples, seq_len))
    return data
train_data = generate_data(num_samples, seq_len, vocab_size)
test_data = generate_data(200, seq_len, vocab_size)
```

```python
# after
def prepare_data(num_samples, seq_len, vocab_size, random_state, test_ratio=0.2):
    total_data = generate_data(num_samples, seq_len, vocab_size)
    train_data, test_data = train_test_split(total_data, test_size=test_ratio, random_state=random_state)
    ...
    return train_input, train_target, test_input, test_target
```

3. **Improved Visualization Aesthetics**  
   Fixed overlapping labels in the original attention plots and improved the figure layout for clarity:

```python
# before
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
```

```python
# after
plt.figure(figsize=(18, 12), constrained_layout=True)
```

   Result: before (left) → after (right)  
   <p align="center">
     <img src="https://github.com/user-attachments/assets/2051e636-0a24-4fb3-8209-ebb060af2b15" width="45%"/>
     <img src="https://github.com/user-attachments/assets/40b1678a-a7b0-41d5-8047-eb46639acdbc" width="45%"/>
   </p>

4. **Extracting Real Attention Weights**  
   One of the most critical improvements: the original project visualized attention maps from a separate `nn.MultiheadAttention` module outside the training process.  
   In this refactored version, we extract **true attention weights** from the model's actual forward pass using custom Transformer layers (`coderlayer_with_attn.py`).  
   For general usage of these layers, see [modified_transformer_to_visualize_attention](https://github.com/PengTang2025/modified_transformer_to_visualize_attention).  
   
   Furthermore, the original heatmaps used an image-style coordinate system (origin top-left, x: key(j), y: query(i)). We've updated the plots to use Cartesian coordinates (origin bottom-left, x: query(i), y: key(j)) for better interpretability.

```python
# before
emb_sample = model.embedding(sample_input) * np.sqrt(model.d_model)
emb_sample = model.pos_encoder(emb_sample)
multihead_attn = nn.MultiheadAttention(embed_dim=model.d_model, num_heads=8, batch_first=True)
attn_output, attn_weights = multihead_attn(emb_sample, emb_sample, emb_sample, need_weights=True, average_attn_weights=False)
```

```python
# after
_ = model(sample_input)  # triggers forward pass
attn_weights = model.last_attn  # retrieves true attention weights
```

   Coordinate system: before (left) → after (right)  
   <p align="center">
     <img src="https://github.com/user-attachments/assets/2a9a1991-c066-43f1-8481-1ea2dadaa4c1" width="45%"/>
     <img src="https://github.com/user-attachments/assets/c5833b8c-0d4b-4155-b496-02fda99d11e1" width="45%"/>
   </p>

   Attention logic: before (left) → after (right)  
   <p align="center">
     <img src="https://github.com/user-attachments/assets/be57481b-f7b6-48b5-bee2-c39420828e5a" width="45%"/>
     <img src="https://github.com/user-attachments/assets/629729a5-9c01-4b7d-a9ac-5cfbc4fa8188" width="45%"/>
   </p>

## 🙏 Acknowledgement & Reference

This project is based on the tutorial article published by the WeChat public account **"机器学习初学者" (Beginner in Machine Learning)**:

> [From-scratch Transformer Implementation with Attention Visualization](https://mp.weixin.qq.com/s/BCECx-0C9E_wY4ZyRrZ5uQ)

All code has been restructured and enhanced, including improvements to architecture, visualization, and attention weight extraction. Sincere thanks to the original author for the valuable reference.

## 📜 License

MIT License © 2025 Peng Tang
