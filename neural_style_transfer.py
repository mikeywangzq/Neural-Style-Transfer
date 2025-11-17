"""
Neural Style Transfer Implementation
Based on "A Neural Algorithm of Artistic Style" by Gatys et al.

使用预训练的 VGG19 网络将风格图的艺术风格应用到内容图上。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import argparse
import os


# ====================== 设备设置 ======================
def get_device():
    """自动检测并返回可用的计算设备 (CUDA, MPS, 或 CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple MPS 设备")
    else:
        device = torch.device("cpu")
        print("使用 CPU 设备")
    return device


# ====================== 图像加载与预处理 ======================
def image_loader(image_path, imsize=512, device='cpu'):
    """
    加载图像并进行预处理

    Args:
        image_path: 图像文件路径
        imsize: 图像调整后的大小
        device: 计算设备

    Returns:
        处理后的图像张量 (1, 3, H, W)
    """
    # VGG 网络的标准均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    # 图像预处理流程
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # 统一调整大小
        transforms.ToTensor(),  # 转换为张量
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)  # 添加 batch 维度

    # 归一化
    image = (image - mean) / std

    return image.to(device, torch.float)


def unloader(tensor):
    """
    将张量转换回 PIL 图像

    Args:
        tensor: 图像张量 (1, 3, H, W)

    Returns:
        PIL 图像对象
    """
    # VGG 标准均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    # 移到 CPU 并克隆
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # 移除 batch 维度

    # 反归一化
    image = image * std + mean
    image = image.clamp(0, 1)  # 限制在 [0, 1] 范围内

    # 转换为 PIL 图像
    image = transforms.ToPILImage()(image)
    return image


# ====================== Gram 矩阵 ======================
def gram_matrix(features):
    """
    计算特征图的 Gram 矩阵 (用于风格表示)

    Args:
        features: 特征图张量 (B, C, H, W)

    Returns:
        归一化的 Gram 矩阵 (B, C, C)
    """
    b, c, h, w = features.size()

    # 将特征图展平为 (B, C, H*W)
    features = features.view(b, c, h * w)

    # 计算 Gram 矩阵: G = F * F^T
    gram = torch.bmm(features, features.transpose(1, 2))

    # 归一化
    gram = gram / (c * h * w)

    return gram


# ====================== 损失函数 ======================
class ContentLoss(nn.Module):
    """内容损失: 计算生成图像与内容图在特征空间的距离"""

    def __init__(self, target_features):
        """
        Args:
            target_features: 内容图的目标特征
        """
        super(ContentLoss, self).__init__()
        self.target = target_features.detach()  # 不需要梯度

    def forward(self, input_features):
        """
        Args:
            input_features: 生成图像的特征

        Returns:
            内容损失值
        """
        loss = nn.functional.mse_loss(input_features, self.target)
        return loss


class StyleLoss(nn.Module):
    """风格损失: 计算生成图像与风格图的 Gram 矩阵距离"""

    def __init__(self, target_features_list):
        """
        Args:
            target_features_list: 风格图在多个层的目标特征列表
        """
        super(StyleLoss, self).__init__()
        # 计算并存储目标 Gram 矩阵
        self.target_grams = [gram_matrix(f).detach() for f in target_features_list]

    def forward(self, input_features_list):
        """
        Args:
            input_features_list: 生成图像在多个层的特征列表

        Returns:
            风格损失值
        """
        loss = 0
        for input_features, target_gram in zip(input_features_list, self.target_grams):
            input_gram = gram_matrix(input_features)
            loss += nn.functional.mse_loss(input_gram, target_gram)
        return loss


# ====================== VGG19 特征提取器 ======================
class StyleTransferModel(nn.Module):
    """基于 VGG19 的风格迁移模型"""

    def __init__(self, device='cpu'):
        """
        Args:
            device: 计算设备
        """
        super(StyleTransferModel, self).__init__()

        # 加载预训练的 VGG19 特征提取器
        vgg = models.vgg19(pretrained=True).features.to(device).eval()

        # 将所有参数设为不需要梯度
        for param in vgg.parameters():
            param.requires_grad = False

        # 替换所有的 inplace ReLU
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)

        self.vgg = vgg

        # 定义内容层和风格层的索引
        # VGG19 层索引: conv1_1=0, conv2_1=5, conv3_1=10, conv4_1=19, conv4_2=21, conv5_1=28
        self.content_layers = [21]  # conv4_2
        self.style_layers = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

    def forward(self, x):
        """
        前向传播,提取内容层和风格层的特征

        Args:
            x: 输入图像张量

        Returns:
            字典包含 'content' 和 'style' 特征列表
        """
        content_features = []
        style_features = []

        for i, layer in enumerate(self.vgg):
            x = layer(x)

            if i in self.content_layers:
                content_features.append(x)

            if i in self.style_layers:
                style_features.append(x)

            # 如果已经提取了所有需要的特征,可以提前结束
            if i > max(max(self.content_layers), max(self.style_layers)):
                break

        return {
            'content': content_features,
            'style': style_features
        }


# ====================== 风格迁移主函数 ======================
def run_style_transfer(content_path, style_path, num_steps=300,
                      style_weight=1000000, content_weight=1,
                      imsize=512, output_path='output.jpg'):
    """
    执行风格迁移

    Args:
        content_path: 内容图路径
        style_path: 风格图路径
        num_steps: 优化迭代次数
        style_weight: 风格损失权重
        content_weight: 内容损失权重
        imsize: 图像大小
        output_path: 输出图像路径

    Returns:
        生成的图像张量
    """
    # 设备设置
    device = get_device()

    # 加载图像
    print("\n加载图像...")
    content_img = image_loader(content_path, imsize, device)
    style_img = image_loader(style_path, imsize, device)
    print(f"内容图尺寸: {content_img.shape}")
    print(f"风格图尺寸: {style_img.shape}")

    # 初始化生成图像 (从内容图开始)
    input_img = content_img.clone()
    input_img.requires_grad_(True)

    # 构建模型
    print("\n构建 VGG19 特征提取器...")
    model = StyleTransferModel(device)

    # 提取目标特征
    print("提取内容和风格特征...")
    with torch.no_grad():
        content_features = model(content_img)['content']
        style_features = model(style_img)['style']

    # 初始化损失函数
    content_loss_fn = ContentLoss(content_features[0]).to(device)
    style_loss_fn = StyleLoss(style_features).to(device)

    # 定义优化器 (优化输入图像)
    optimizer = optim.Adam([input_img], lr=0.01)

    # 优化循环
    print(f"\n开始风格迁移 (共 {num_steps} 步)...")
    print(f"内容权重: {content_weight}, 风格权重: {style_weight}\n")

    for step in range(num_steps):
        # 将输入图像限制在有效范围内
        with torch.no_grad():
            # 反归一化到 [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)
            input_img_denorm = input_img * std + mean
            input_img_denorm.clamp_(0, 1)
            # 重新归一化
            input_img.data = (input_img_denorm - mean) / std

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        features = model(input_img)

        # 计算损失
        c_loss = content_loss_fn(features['content'][0])
        s_loss = style_loss_fn(features['style'])

        total_loss = content_weight * c_loss + style_weight * s_loss

        # 反向传播
        total_loss.backward()

        # 更新图像
        optimizer.step()

        # 打印进度
        if (step + 1) % 50 == 0:
            print(f"Step [{step + 1}/{num_steps}] | "
                  f"总损失: {total_loss.item():.2f} | "
                  f"内容损失: {c_loss.item():.4f} | "
                  f"风格损失: {s_loss.item():.2f}")

    print("\n风格迁移完成!")

    # 保存结果
    print(f"保存结果到 {output_path}...")
    output_image = unloader(input_img)
    output_image.save(output_path)
    print("保存完成!")

    return input_img


# ====================== 主程序 ======================
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True,
                       help='内容图路径')
    parser.add_argument('--style', type=str, required=True,
                       help='风格图路径')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='输出图像路径 (默认: output.jpg)')
    parser.add_argument('--imsize', type=int, default=512,
                       help='图像大小 (默认: 512)')
    parser.add_argument('--steps', type=int, default=300,
                       help='优化步数 (默认: 300)')
    parser.add_argument('--style-weight', type=float, default=1000000,
                       help='风格权重 (默认: 1000000)')
    parser.add_argument('--content-weight', type=float, default=1,
                       help='内容权重 (默认: 1)')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.content):
        print(f"错误: 找不到内容图 '{args.content}'")
        return

    if not os.path.exists(args.style):
        print(f"错误: 找不到风格图 '{args.style}'")
        return

    # 执行风格迁移
    print("=" * 60)
    print("Neural Style Transfer")
    print("=" * 60)

    run_style_transfer(
        content_path=args.content,
        style_path=args.style,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        imsize=args.imsize,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
