"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.在玩具 Balloon 数据集上训练并实现色彩飞溅效果。

版权所有 (c) 2018 Matterport, Inc.
许可协议：MIT 许可协议（详细信息请参见 LICENSE 文件）
编写者：Waleed Abdulla

------------------------------------------------------------

使用方法：导入模块（请参见 Jupyter notebooks 示例），或从命令行运行：


这是有关 Mask R-CNN 的说明和使用指南，原文如下：

Mask R-CNN

在玩具 Balloon 数据集上训练并实现色彩飞溅效果。

版权所有 (c) 2018 Matterport, Inc.
许可协议：MIT 许可协议（详细信息请参见 LICENSE 文件）
编写者：Waleed Abdulla

使用方法：导入模块（请参见 Jupyter notebooks 示例），或从命令行运行：

    # 从预训练的 COCO 权重开始训练新模型
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # 继续训练之前训练过的模型
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # 从 ImageNet 权重开始训练新模型
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # 对图像应用色彩飞溅效果
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL 或文件路径>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import time
# 项目根目录

ROOT_DIR = os.path.abspath("../../") # 返回上两级指定路径的绝对路径

# 加载 Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# COCO训练权重文件路径
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# 保存日志和模型检查点的目录，如果没有通过命令行参数--logs提供
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BalloonConfig(Config):
    """用于在数据集上进行训练的配置
        继承自基础配置类并重写了一些值
    """
    # 为配置提供一个可识别的名称
    NAME = "fission_track"

    # 我们使用具有12GB内存的GPU，可以加载两张图像。如果您使用较小的GPU，请适当调整
    IMAGES_PER_GPU = 1

    # 类别数 (包括背景)
    NUM_CLASSES = 1 + 5 # 背景 + 类别

    # 每个周期的训练步数
    STEPS_PER_EPOCH = 40

    # 跳过置信度低于90%的检测
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """加载Balloon数据集的一个子集
        dataset_dir: 数据集的根目录
        subset: 要加载的子集: train or val。
        """
        # 在这里增加类别
        self.add_class("balloon", 1, "complete")
        self.add_class("balloon", 2, "incomplete")
        self.add_class("balloon", 3, "incompletedistinct")
        self.add_class("balloon", 4, "completefuzzy")
        self.add_class("balloon", 5, "mineral")

        # 训练还是验证数据集?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # 加载注释
        # VGG图像标注工具（VIA，最多到1.6版本）以如下形式保存每张图像：

        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # 我们主要关心每个区域的x和y坐标。
        # 注意：在VIA 2.0中，regions从字典变为列表。

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # 不需要字典的键

        # VIA 工具即使在图像没有任何注释时也会在 JSON 中保存它们。跳过没有注释的图像
        # 将regions取出来
        annotations = [a for a in annotations if a['regions']]

        # 添加图片
        for a in annotations:
            # 获取组成每个对象实例轮廓的多边形的 x 和 y 坐标点。这些坐标存储在 shape_attributes 中（参见上面的 JSON 格式）。if 条件用于支持 VIA 1.x 和 2.x 版本。

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]



            # load_mask()函数需要图片尺寸来将多边形转换成掩码
            # 不幸的是，VIA 并没有在 JSON 中包含图像的尺寸信息，因此我们必须读取图像。由于数据集很小，这样做还是可行的。
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path) #image为numpy数组(height, width, channels)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # 使用文件名作为唯一图片id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        classID=np.zeros(len(info["polygons"]),dtype=np.uint8)
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            if p['name']=='complete':
                classID[i,] = 1
            elif p['name'] == 'incomplete':
                classID[i,] = 2
            elif p['name'] == 'incompletedistinct':
                classID[i,] = 3
            elif p['name'] == 'completefuzzy':
                classID[i,] = 4
            else:
                classID[i,] = 5


        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), classID

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """训练模型."""
    # 训练集
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(os.path.join(ROOT_DIR, "images"), "train")
    dataset_train.prepare()

    # 测试集
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(os.path.join(ROOT_DIR, "images"), "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    start_time = time.time()
    config = BalloonConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    weight='coco'
    # Select weights file to load
    if weight == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weight == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weight == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weight



    # Load weights
    print("Loading weights ", weights_path)
    if weight == "coco":
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    train(model)
    elapsed_time = time.time() - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60

    # 输出转换后的时间
    print(f"运行时间：{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

