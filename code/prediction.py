import os
import json
import cv2
import skimage
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.model import load_image_gt
import numpy as np
columns = ["Image", "num","True Positive", "False Positive", "False Negative", "Precision", "Recall", "F1"]
df = pd.DataFrame(columns=columns)
class BalloonConfig(Config):
    """用于在玩具数据集上进行训练的配置
        继承自基础配置类并重写了一些值
    """
    # 为配置提供一个可识别的名称
    NAME = "fission_track"

    USE_MINI_MASK = False

    # 我们使用具有12GB内存的GPU，可以加载两张图像。如果您使用较小的GPU，请适当调整
    IMAGES_PER_GPU =1

    # 类别数 (包括背景)
    NUM_CLASSES = 1 + 5 # 背景 + 类别

    # 每个周期的训练步数
    STEPS_PER_EPOCH = 40

    # 跳过置信度低于70%的检测
    DETECTION_MIN_CONFIDENCE = 0.6

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
                names = [r['shape_attributes'].get('name', 'unknown') for r in a['regions'].values()]
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
                names=names,
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
                if p['name'] == 'complete':
                    classID[i,] = 1
                elif p['name'] == 'incomplete':
                    classID[i,] = 2
                elif p['name'] == 'incompletedistinct':
                    classID[i,] = 3
                elif p['name'] == 'completefuzzy':
                    classID[i,] = 4
                else:
                    classID[i,] = 5

        return mask, classID

path='C:/Users/h1399/Desktop/DT/RCNN/Maskrcnn/MaskRCNN/images/train/'
config=BalloonConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir='C:/Users/h1399/Desktop/DT/RCNN/Maskrcnn/MaskRCNN/logs')
checkpoint=model.find_last()
print("Loading weights from ", checkpoint)
model.load_weights(checkpoint, by_name=True)
class_names=['BG','FT', 'IFT','IFT_dis','FT_fuz','mineral']
from mrcnn.visualize import display_instances_2
for i in range(16,17):
    filename = f'{i:03}.jpg'
    image=cv2.imread(os.path.join(path, filename))
    result= model.detect([image])[0]
    display_instances_2(image,result['rois'],result['masks'],result['class_ids'],class_names,result['angles'],
                        scores = result['scores'], title = "检测结果",
                        figsize = (16, 16), ax = None,
                        show_mask = True, show_bbox = True,
                        colors = None, captions = None
                        )


    def evaluate(model, dataset_dir, subset):
        """评估模型的性能."""
        # 加载数据集
        dataset = BalloonDataset()
        dataset.load_balloon(dataset_dir, subset)
        dataset.prepare()

        image_ids = dataset.image_ids
        augment = False
        augmentation = None
        for image_id in image_ids[i-1:i]:  # 数字跟后面的一致
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=augmentation,
                              use_mini_mask=config.USE_MINI_MASK)
        # 注意验证数据集和'C:/Users/h1399/Desktop/Maskrcnn/MaskRCNN/images/train/005.jpg'路径，不然重合面积计算会错误
        iou_threshold = 0.2
        score_threshold = 0.3
        # 缩放因子
        scale = 2
        # 填充参数
        padding = ((0, 0), (0, 0), (0, 0))
        resized_mask = utils.resize_mask(gt_masks, scale, padding)
        gt_match, pred_match, overlaps = utils.compute_matches(
            gt_boxes, gt_class_ids, resized_mask,
            result['rois'], result['class_ids'], result['scores'], result['masks'],
            iou_threshold=iou_threshold, score_threshold=score_threshold)
        print('第',i,'张图片Labelme标记的裂变径迹个数:', len(gt_class_ids))

        def compute_tp_fp_fn(gt_match, pred_match, iou_threshold=0.3):
            # 假设真实标注框（ground truth）和预测框（predicted）索引从0开始
            num_gt = len(gt_match)
            num_pred = len(pred_match)

            # 初始化 TP、FP、FN
            true_positive = 0
            false_positive = 0
            false_negative = 0

            # 找到匹配的真实标注框和预测框
            matched_gt = set(gt_match[gt_match >= 0])
            matched_pred = set(pred_match[pred_match >= 0])

            # 计算 TP 和 FP
            for i in range(num_pred):
                if pred_match[i] >= 0:  # 该预测框匹配到真实标注框
                    true_positive += 1
                else:
                    false_positive += 1

            # 计算 FN
            for i in range(num_gt):
                if gt_match[i] == -1:  # 该真实标注框没有匹配到任何预测框
                    false_negative += 1

            return true_positive, false_positive, false_negative

        TP, FP, FN = compute_tp_fp_fn(gt_match, pred_match, iou_threshold=0.2)
        try:
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1 = 0
            print('验证集路径有误，请重新确认')
        df.loc[len(df)] = [filename, len(gt_class_ids),TP, FP, FN, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
        return precision, recall, f1


    ROOT_DIR = 'C:/Users/h1399/Desktop/DT/RCNN/Maskrcnn/MaskRCNN'

    evaluate(model, os.path.join(ROOT_DIR, "images"), "val")
df.to_excel('C:/Users/h1399/Desktop/evaluation_results4.xlsx', index=False)


def count_instances(result):
    """
    统计检测到的实例个数
    :param result: 从模型检测得到的结果列表，例如 [result]，result 是一个字典。
    :return: 实例的数量
    """
    if not result or not isinstance(result, list) or len(result) == 0:
        return 0

    # 获取检测结果中的第一个图像的结果
    detections = result[0]

    # 计算 'rois' 数组的行数，即检测到的实例数量
    num_instances = detections['rois'].shape[0]

    return num_instances
num_instances = count_instances(result)
print(f"检测到的实例数量: {num_instances}")



