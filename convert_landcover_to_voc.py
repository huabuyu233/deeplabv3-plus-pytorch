"""
将LandCover.ai数据集转换为VOC格式
- 切分为512x512的图像
- 转换为VOC2007格式（JPEGImages和SegmentationClass）
- 不划分训练集/验证集，后续使用voc_annotation.py进行划分
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import rasterio
from PIL import Image

# ==================== 配置参数 ====================
# LandCover.ai数据集路径
LANDCOVER_DATA_DIR = r"D:\文件文档\毕业设计和论文\数据集\landcover.ai.v1"

# VOC输出路径
VOC_OUTPUT_DIR = r"E:\project\vscode project\deeplabv3-plus-pytorch\VOCdevkit\VOC2007"

# 切片大小
TARGET_SIZE = 512

# 是否过滤掉纯背景的切片（可选，减少训练数据量）
FILTER_BACKGROUND_ONLY = True

# ==================== 类别定义 ====================
# LandCover.ai类别: 0-Background, 1-Building, 2-Woodland, 3-Water, 4-Road
# 这些类别ID会被直接保留，符合VOC格式要求（像素值=类别ID）
CLASS_NAMES = ["background", "building", "woodland", "water", "road"]
NUM_CLASSES = 5  # 类别总数（包括背景）


def check_voc_dirs(voc_dir):
    """检查VOC格式目录结构是否存在"""
    dirs = [
        "JPEGImages",
        "SegmentationClass",
        "ImageSets/Segmentation"
    ]
    for d in dirs:
        dir_path = os.path.join(voc_dir, d)
        if not os.path.exists(dir_path):
            print(f"⚠ 目录不存在，正在创建: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            print(f"✓ 目录已存在: {dir_path}")


def slice_and_convert(landcover_dir, output_dir, filter_background=True):
    """
    将LandCover.ai数据集切片并转换为VOC格式

    Args:
        landcover_dir: LandCover.ai数据集根目录
        output_dir: VOC输出目录
        filter_background: 是否过滤纯背景切片
    """
    images_dir = os.path.join(landcover_dir, "images")
    masks_dir = os.path.join(landcover_dir, "masks")

    # 获取所有图像和掩膜路径
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.tif")))

    assert len(img_paths) == len(mask_paths), f"图像({len(img_paths)})和掩膜({len(mask_paths)})数量不匹配！"
    
    print(f"找到 {len(img_paths)} 对图像/掩膜文件")

    # 输出目录
    jpg_output = os.path.join(output_dir, "JPEGImages")
    mask_output = os.path.join(output_dir, "SegmentationClass")

    total_slices = 0
    filtered_slices = 0
    saved_slices = 0

    # 处理每张图像
    for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths), desc="处理图像"):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]

        # 使用rasterio读取GeoTiff文件
        try:
            with rasterio.open(img_path) as src:
                img = src.read()  # 形状为 (channels, height, width)
                # 转换为 (height, width, channels)
                if img.shape[0] < img.shape[2]:  # 通道在第一个维度
                    img = np.transpose(img, (1, 2, 0))
            
            with rasterio.open(mask_path) as src:
                mask = src.read()
                # 掩膜可能是 (1, H, W) 或 (H, W)
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]  # 取第一个通道
        except Exception as e:
            print(f"⚠ 警告: 无法读取 {img_filename}: {e}，跳过")
            continue

        # 确保图像是RGB三通道
        if img.ndim == 2:
            # 灰度图转RGB
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[2] == 4:
                # RGBA转RGB
                img = img[:, :, :3]
            elif img.shape[2] > 3:
                # 只取前3通道
                img = img[:, :, :3]
            elif img.shape[2] < 3:
                # 单通道或双通道，复制为3通道
                img = np.repeat(img[:, :, :1], 3, axis=2)

        # 确保掩膜是2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # 确保类别ID在正确范围内 (0-4)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # 切片处理
        slice_idx = 0
        for y in range(0, img.shape[0], TARGET_SIZE):
            for x in range(0, img.shape[1], TARGET_SIZE):
                # 提取切片
                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

                # 检查是否是完整切片
                if img_tile.shape[0] != TARGET_SIZE or img_tile.shape[1] != TARGET_SIZE:
                    slice_idx += 1
                    continue

                total_slices += 1
                
                # 生成文件名（与官方划分文件格式一致）
                slice_name = f"{img_filename}_{slice_idx:03d}"

                # 检查是否是纯背景（在保存前检查）
                if filter_background:
                    unique_classes = np.unique(mask_tile)
                    if len(unique_classes) == 1 and unique_classes[0] == 0:
                        filtered_slices += 1
                        slice_idx += 1
                        continue

                # 保存图像（RGB转JPEG）
                # 确保数据类型为uint8
                if img_tile.dtype != np.uint8:
                    # 归一化到0-255（如果需要）
                    if img_tile.max() > 255:
                        img_tile = (img_tile / img_tile.max() * 255).astype(np.uint8)
                    else:
                        img_tile = img_tile.astype(np.uint8)
                
                img_save_path = os.path.join(jpg_output, f"{slice_name}.jpg")
                # 使用PIL保存JPEG
                Image.fromarray(img_tile).convert('RGB').save(img_save_path, 'JPEG', quality=95)

                # 保存掩膜（PNG格式，保持像素值为类别ID）
                mask_save_path = os.path.join(mask_output, f"{slice_name}.png")
                # 确保掩膜是uint8
                if mask_tile.dtype != np.uint8:
                    mask_tile = mask_tile.astype(np.uint8)
                Image.fromarray(mask_tile, mode='L').save(mask_save_path, 'PNG')
                
                saved_slices += 1
                slice_idx += 1

    # 统计信息
    print("\n" + "="*60)
    print("转换完成！统计信息:")
    print(f"总切片数（不含不完整边缘）: {total_slices}")
    print(f"过滤的纯背景切片: {filtered_slices}")
    print(f"实际保存切片数: {saved_slices}")
    print(f"保存到: {output_dir}")
    print("="*60)


def verify_dataset(voc_dir, num_samples=5):
    """验证转换后的数据集"""
    import random

    jpg_dir = os.path.join(voc_dir, "JPEGImages")
    mask_dir = os.path.join(voc_dir, "SegmentationClass")

    # 随机检查几张图
    jpg_files = [f for f in os.listdir(jpg_dir) if f.endswith('.jpg')]

    if len(jpg_files) == 0:
        print("⚠ 警告: 没有找到图像文件！")
        return

    print("\n" + "="*60)
    print("验证数据集...")
    print(f"图像总数: {len(jpg_files)}")

    # 检查掩膜
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    print(f"掩膜总数: {len(mask_files)}")
    
    # 检查文件匹配
    jpg_basenames = set([os.path.splitext(f)[0] for f in jpg_files])
    mask_basenames = set([os.path.splitext(f)[0] for f in mask_files])
    
    if jpg_basenames != mask_basenames:
        print(f"⚠ 警告: 图像和掩膜文件不匹配！")
        missing_in_mask = jpg_basenames - mask_basenames
        if missing_in_mask:
            print(f"  缺少掩膜的图像: {list(missing_in_mask)[:5]}...")
    
    print("="*60)

    # 随机抽样检查
    sample_files = random.sample(jpg_files, min(num_samples, len(jpg_files)))

    for jpg_file in sample_files:
        base_name = os.path.splitext(jpg_file)[0]
        mask_file = base_name + ".png"

        jpg_path = os.path.join(jpg_dir, jpg_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"\n样本: {base_name}")
            print(f"  ⚠ 掩膜文件不存在: {mask_path}")
            continue

        # 使用PIL读取图像
        try:
            img = Image.open(jpg_path).convert('RGB')
            img_np = np.array(img)
            
            mask = Image.open(mask_path)
            mask_np = np.array(mask)
        except Exception as e:
            print(f"\n样本: {base_name}")
            print(f"  ⚠ 无法读取文件: {e}")
            continue

        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]

        unique_classes = np.unique(mask_np)

        print(f"\n样本: {base_name}")
        print(f"  图像尺寸: {img_np.shape} (高×宽×通道)")
        print(f"  掩膜尺寸: {mask_np.shape} (高×宽)")
        print(f"  包含类别ID: {unique_classes}")

        # 统计每个类别的像素数量
        total_pixels = mask_np.shape[0] * mask_np.shape[1]
        for cls_id in sorted(unique_classes):
            pixel_count = np.sum(mask_np == cls_id)
            ratio = pixel_count / total_pixels * 100
            cls_name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else "unknown"
            print(f"    类别 {int(cls_id)} ({cls_name:10s}): {pixel_count:6d} 像素 ({ratio:5.2f}%)")


if __name__ == "__main__":
    print("="*60)
    print("LandCover.ai → VOC2007 转换工具")
    print("="*60)
    print(f"\n数据源: {LANDCOVER_DATA_DIR}")
    print(f"输出目标: {VOC_OUTPUT_DIR}\n")

    # 检查VOC目录
    check_voc_dirs(VOC_OUTPUT_DIR)

    # 执行转换
    print("\n" + "="*60)
    print("开始转换...")
    print("="*60)
    slice_and_convert(
        landcover_dir=LANDCOVER_DATA_DIR,
        output_dir=VOC_OUTPUT_DIR,
        filter_background=FILTER_BACKGROUND_ONLY
    )

    # 验证数据集
    verify_dataset(VOC_OUTPUT_DIR)

    print("\n" + "="*60)
    print("✓ 转换流程完成！")
    print(f"\n下一步操作:")
    print(f"  1. 检查上方统计信息是否正确")
    print(f"  2. 运行 voc_annotation.py 生成 train.txt 和 val.txt")
    print(f"     python voc_annotation.py")
    print(f"  3. 在 train.py 中设置:")
    print(f"     - num_classes = 5  # 类别数（不含背景）")
    print(f"     - backbone = 'mobilenet'")
    print(f"     - model_path = 'model_data/deeplab_mobilenetv2.pth' (如有预训练权重)")
    print(f"  4. 运行 python train.py 开始训练")
    print("="*60)
