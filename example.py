"""
示例脚本: 展示如何在 Python 代码中使用神经风格迁移
"""

from neural_style_transfer import run_style_transfer, get_device

def main():
    """示例: 在代码中调用风格迁移"""

    # 设置参数
    content_image_path = "path/to/content.jpg"
    style_image_path = "path/to/style.jpg"
    output_path = "output.jpg"

    print(f"使用设备: {get_device()}")

    # 执行风格迁移
    result = run_style_transfer(
        content_path=content_image_path,
        style_path=style_image_path,
        num_steps=300,           # 迭代次数
        style_weight=1e6,        # 风格权重 (越大风格越强)
        content_weight=1,        # 内容权重 (越大内容保留越多)
        imsize=512,              # 图像大小
        output_path=output_path
    )

    print(f"\n完成! 结果已保存到 {output_path}")


if __name__ == '__main__':
    # 注意: 请先准备好内容图和风格图,并修改上面的路径
    # 然后运行: python example.py

    print("这是一个示例脚本。")
    print("请修改 content_image_path 和 style_image_path,然后取消注释 main() 调用。")
    print("\n推荐使用命令行方式:")
    print("python neural_style_transfer.py --content content.jpg --style style.jpg")

    # 取消下面这行的注释来运行示例
    # main()
