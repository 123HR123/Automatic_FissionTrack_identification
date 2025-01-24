import os
from PIL import Image

# 定义源文件夹和目标文件夹路径
source_folder = 'C:/Users/h1399/Desktop/目标检测/RCNN系列/Maskrcnn/MaskRCNN/images/train'


# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.tif'):
        # 构建完整的文件路径
        tif_file_path = os.path.join(source_folder, filename)
        # 打开并读取tif文件
        with Image.open(tif_file_path) as img:
            # 构建新的jpg文件路径
            jpg_filename = filename.replace('.tif', '.jpg')
            jpg_file_path = os.path.join(source_folder, jpg_filename)

            # 将图像保存为jpg格式
            img.save(jpg_file_path, 'JPEG')

print("转换完成！")
