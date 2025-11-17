# Neural Style Transfer

基于 Gatys 等人论文 "A Neural Algorithm of Artistic Style" 的 PyTorch 实现。

该项目使用预训练的 VGG19 网络提取图像特征，将一张"风格图"的艺术风格应用到一张"内容图"上，生成具有艺术风格的新图像。

## 特性

- ✅ 基于 VGG19 的特征提取
- ✅ 自动设备检测 (CUDA / MPS / CPU)
- ✅ 可调节的风格和内容权重
- ✅ 完整的命令行接口
- ✅ 详细的训练进度输出

## 环境要求

- Python 3.7+
- PyTorch 2.0+
- torchvision 0.15+
- Pillow 9.0+

## 安装

```bash
# 克隆项目
git clone https://github.com/mikeywangzq/Neural-Style-Transfer.git
cd Neural-Style-Transfer

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python neural_style_transfer.py \
    --content path/to/content.jpg \
    --style path/to/style.jpg \
    --output output.jpg
```

### 高级选项

```bash
python neural_style_transfer.py \
    --content path/to/content.jpg \
    --style path/to/style.jpg \
    --output output.jpg \
    --imsize 512 \
    --steps 300 \
    --style-weight 1000000 \
    --content-weight 1
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--content` | str | 必需 | 内容图路径 |
| `--style` | str | 必需 | 风格图路径 |
| `--output` | str | output.jpg | 输出图像路径 |
| `--imsize` | int | 512 | 图像处理尺寸 (像素) |
| `--steps` | int | 300 | 优化迭代次数 |
| `--style-weight` | float | 1000000 | 风格损失权重 |
| `--content-weight` | float | 1 | 内容损失权重 |

## 技术细节

### 架构

1. **特征提取**: 使用预训练的 VGG19 网络
   - 内容层: `conv4_2` (第 21 层)
   - 风格层: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` (第 0, 5, 10, 19, 28 层)

2. **损失函数**:
   - **内容损失**: 生成图像与内容图在 `conv4_2` 层特征的 MSE
   - **风格损失**: 生成图像与风格图在多个层 Gram 矩阵的 MSE 总和
   - **总损失**: `content_weight × content_loss + style_weight × style_loss`

3. **优化器**: Adam (学习率 0.01)

4. **关键实现**:
   - 将 VGG19 中的 `ReLU(inplace=True)` 替换为 `ReLU(inplace=False)` 以支持梯度反向传播
   - 使用 ImageNet 标准化参数 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - Gram 矩阵归一化: 除以 (通道数 × 高度 × 宽度)

### Gram 矩阵

Gram 矩阵用于捕捉图像的风格信息:

```
G = F × F^T / (C × H × W)
```

其中 F 是特征图 (C, H×W)

## 示例结果

输入内容图 + 风格图 → 输出风格化图像

## 性能提示

- **GPU 加速**: 如有 CUDA 或 MPS (Apple Silicon) 可用,会自动启用
- **图像尺寸**: 较大的图像需要更多内存和时间,建议从 512×512 开始
- **迭代次数**: 300 步通常足够,可根据效果调整
- **权重调节**:
  - 增加 `style-weight` 使风格更强烈
  - 增加 `content-weight` 使内容保留更多

## 参考文献

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). "A Neural Algorithm of Artistic Style". arXiv:1508.06576

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request!