"""
Mask R-CNN
显示和可视化功能。

版权所有 (c) 2017 Matterport, Inc.
根据 MIT 许可证授权（详细信息见 LICENSE 文件）
作者：Waleed Abdulla
"""

import os
import sys
import random
import itertools
import pandas as pd
import colorsys
import cv2
import math
import numpy as np
from scipy.linalg import eig
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """
    显示给定的一组图像，可以选择性地附加标题。
    images: 图像张量的列表或数组，格式为 HWC。
    titles: 可选。要显示在每个图像旁的标题列表。
    cols: 每行显示的图像数量。
    cmap: 可选。使用的颜色。例如，“Blues”。
    norm: 可选。一个 Normalize 实例，用于将值映射到颜色。
    interpolation: 可选。用于显示的图像插值方法。
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


import colorsys
import random

def random_colors(N, bright=True):
    """ 生成随机颜色，过滤固定颜色后保持总数不变。 """
    random.seed(0)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]  # (色调，饱和度，亮度)
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))  # 转换为RGB
    random.shuffle(colors)  # 打乱顺序

    # 过滤掉与固定颜色重复的颜色
    fixed_colors = [(0, 1, 0), (0, 0, 1)]  # 绿色、蓝色

    def is_similar(color1, color2, threshold=0.1):
        """ 判断两个颜色是否相似 """
        return all(abs(c1 - c2) < threshold for c1, c2 in zip(color1, color2))

    # 过滤随机颜色
    filtered_colors = [color for color in colors if not any(is_similar(color, fc) for fc in fixed_colors)]

    # 计算需要添加的颜色数量
    num_to_add = N - len(filtered_colors)

    # 如果过滤后颜色数量小于N，补充
    if num_to_add == 1:
        # 随机选取一些颜色补充
        new_colors1 = (1,1,0) #黄色
        filtered_colors.append(new_colors1)
    elif num_to_add == 2:
        # 随机选取一些颜色补充
        new_colors1 = (1,1,0)
        new_colors2 = (0.5,0,0.5) #紫色
        filtered_colors.append(new_colors1)
        filtered_colors.append(new_colors2)
    # 定义7个固定的颜色（RGB范围为0到1）


    return filtered_colors



def apply_mask(image, mask, color, alpha=0.3):
    """将给定掩码用于图像上
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])  #为1变色，为0保持
    return image


import numpy as np
from skimage.metrics import structural_similarity as ssim
def display_instances_2(image, boxes, masks, class_ids, class_names, angles=None,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    N = boxes.shape[0]
    FT_length={} #储存裂变径迹截距长度
    if not N:
        print("\n*** No instances to display *** \n")
        return
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    unique_class_ids = np.unique(class_ids)
    color_map = random_colors(len(unique_class_ids))
    color_dict = {class_id: color for class_id, color in zip(unique_class_ids, color_map)}
    print(color_dict)
    color_dict={1: (0.0, 1.0, 1.0), 2: (1.0, 0.0, 0.0), 3: (1.0, 1.0, 0.0), 4: (0.5, 0.0, 1.0),5:(0,0.5,0.5)}

    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def pca_object_direction(mask):
        # 获取掩码中非零像素的坐标
        coords = np.column_stack(np.where(mask > 0))

        # 如果没有物体，返回 None
        if coords.shape[0] == 0:
            return None

        # 使用PCA分析物体的主方向
        pca = PCA(n_components=2)
        pca.fit(coords)

        # 主成分方向：第一个主成分向量
        principal_direction = pca.components_[0]

        # 计算主方向的角度（单位：度数）
        angle = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi
        if angle>0:
            angle=angle-180
        #角度为x轴正方形顺时针转到垂直于最长轴的方向，数值为-（顺时针）

        return angle

    # Function to calculate structural similarity after 180 degree center rotation for the mask
    def check_center_rotation_similarity(mask, box):
        y1, x1, y2, x2 = box
        object_mask = mask[y1:y2, x1:x2]
        mask_center_x = (x1 + x2) // 2
        mask_center_y = (y1 + y2) // 2
        mask_center = (mask_center_x, mask_center_y)

        # Rotate the mask around its center by 180 degrees
        height, width = object_mask.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(mask_center, 180, 1.0)
        object_mask = np.array(object_mask).astype(np.uint8)
        rotated_object_mask = cv2.warpAffine(object_mask, rotation_matrix, (width, height))

        # 计算重叠区域（交集）
        overlap = np.logical_and(object_mask, rotated_object_mask).sum()

        # 计算并集
        union = np.logical_or(object_mask, rotated_object_mask).sum()

        # 如果并集为0，返回重叠为0，避免除以0
        if union == 0:
            return 0

        # 计算IoU（交并比）
        iou = overlap / union
        return iou

    def calculate_length(pca_angle,angle,object_length, object_width):
        '---------------------以下插入角度-------------'
        alpha = np.rad2deg(abs(pca_angle))
        beta = np.rad2deg(abs(angle))

        length = 2 * np.sqrt((1/ (1 / object_length ** 2 + (np.tan(alpha - beta)) ** 2 / object_width ** 2)))
        return length
    for i in range(N):
        color = color_dict[class_ids[i]]
        #color =(1,0,0)

        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]

        # 计算物体的宽度和长度,###掩码的形状为(h,w,N)其中h,w为输入图片的尺寸，N为实例个数。
        '''假设输入图像的大小为 1024x1024，有两个物体 A 和 B：
        掩码 mask_A 是一个 1024x1024 的矩阵，其中只有物体 A 所在的像素值为 1，其余位置（包括物体 B 的像素区域）为 0。
        掩码 mask_B 也是一个 1024x1024 的矩阵，只有物体 B 所在的像素值为 1，其余位置（包括物体 A 的像素区域）为 0。'''
        mask = masks[:, :, i]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5) #返回物体边缘上点的集合
        pca_angle=0
        if len(contours) > 0:
            # 筛选有效的轮廓
            contours = [cnt for cnt in contours if len(cnt) >= 5]
            contours = [cnt.astype(np.float32) for cnt in contours]

            # 选择最大轮廓
            contour = max(contours, key=cv2.contourArea)

            # 构造矩阵 D
            x = contour[:, 0]
            y = contour[:, 1]
            D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T

            # 构造矩阵 C
            C = np.array([[0, 0, 2, 0, 0],
                          [0, -1, 0, 0, 0],
                          [2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

            # 计算 D^T * D
            DT_D = np.dot(D.T, D)

            # 求解特征值问题
            eigvals, eigvecs = eig(DT_D, C)

            # 选择最小的特征值对应的特征向量
            A = eigvecs[:, np.argmin(eigvals)]

            # 提取椭圆参数
            a, b, c, d, e, f = A

            # 计算椭圆的中心
            center_x = (c * e - b * d) / (b ** 2 - 4 * a * c)
            center_y = (a * e - b * d) / (b ** 2 - 4 * a * c)
            center = (center_x, center_y)

            # 计算椭圆的轴长和旋转角度
            temp = np.sqrt((a - c) ** 2 + 4 * b ** 2)
            axis_1 = np.sqrt(2 * (a + c + temp))
            axis_2 = np.sqrt(2 * (a + c - temp))
            angle = 0.5 * np.arctan2(2 * b, (a - c))

            # 确保 major_axis 和 minor_axis 的赋值正确
            major_axis = max(axis_1, axis_2)
            minor_axis = min(axis_1, axis_2)

            # 输出椭圆的长短轴
            object_length = minor_axis
            object_width = major_axis

            a = sigmoid(object_width - 33.3) * (object_width / object_length)
            # 根据条件更改颜色
            if a > 0.38 and object_width < 100 and object_length < 100:
                color = (0, 1, 0)  # 设置为绿色

            # Check for objects with width and length > 100
            elif object_width > 250 and object_length > 250:
                # 进行180度中心旋转相似度检测
                similarity_score = check_center_rotation_similarity(mask, boxes[i])

                if similarity_score > 0.4:
                    pca_angle = pca_object_direction(mask[y1:y2, x1:x2])
                    print(f"矿物中心旋转的相似值为 {similarity_score}，主方向角度为 {pca_angle:.2f} 度")

                else:
                    print(f"矿物中心旋转的相似值为 {similarity_score}")
                    print('无法检测到矿物的c轴')
                continue #不显示颜色，(0,0,1)显示蓝色
                #color = (0, 0, 1)
            length=calculate_length(pca_angle,angle,object_length, object_width)
            FT_length[i+1] = length


        # 绘制边界框
        if show_bbox and color is not None:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = " ".format(label, score) if score else label
            angle = angles[i]
            #caption = "NO.{} {} {:.2f}".format(i + 1, label, score) if score else "{}\nAngle: {:.2f}".format(label)

            #caption = "NO.{} {} {:.2f}".format(i + 1, label, score) if score else "{}\nAngle: {:.2f}".format(label)

        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
        # 掩码
        if show_mask and color is not None:
            masked_image = apply_mask(masked_image, mask, color)

        for verts in contours:
            verts = np.fliplr(verts) - 1
            if color is not None:
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    #print(FT_length)
    df = pd.DataFrame(list(FT_length.items()), columns=['Index', 'FT_length'])
    # 写入 Excel 文件
    #df.to_excel('C:/Users/h1399/Desktop/conclusion/FT_length_data1.xlsx', index=False)

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
def display_instances_1(image, boxes, masks, class_ids, class_names, angles=None,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
        return
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    unique_class_ids = np.unique(class_ids)
    color_map = random_colors(len(unique_class_ids))
    color_dict = {class_id: color for class_id, color in zip(unique_class_ids, color_map)}

    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def pca_object_direction(mask):
        # 获取掩码中非零像素的坐标
        coords = np.column_stack(np.where(mask > 0))

        # 如果没有物体，返回 None
        if coords.shape[0] == 0:
            return None

        # 使用PCA分析物体的主方向
        pca = PCA(n_components=2)
        pca.fit(coords)

        # 主成分方向：第一个主成分向量
        principal_direction = pca.components_[0]

        # 计算主方向的角度（单位：度数）
        angle = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi
        if angle>0:
            angle=angle-180
        #角度为x轴正方形顺时针转到垂直于最长轴的方向，数值为-（顺时针）

        return angle

    # Function to calculate structural similarity after 180 degree center rotation for the mask
    def check_center_rotation_similarity(mask, box):
        y1, x1, y2, x2 = box
        object_mask = mask[y1:y2, x1:x2]
        mask_center_x = (x1 + x2) // 2
        mask_center_y = (y1 + y2) // 2
        mask_center = (mask_center_x, mask_center_y)

        # Rotate the mask around its center by 180 degrees
        height, width = object_mask.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(mask_center, 180, 1.0)
        object_mask = np.array(object_mask).astype(np.uint8)
        rotated_object_mask = cv2.warpAffine(object_mask, rotation_matrix, (width, height))

        # 计算重叠区域（交集）
        overlap = np.logical_and(object_mask, rotated_object_mask).sum()

        # 计算并集
        union = np.logical_or(object_mask, rotated_object_mask).sum()

        # 如果并集为0，返回重叠为0，避免除以0
        if union == 0:
            return 0

        # 计算IoU（交并比）
        iou = overlap / union
        return iou

    for i in range(N):
        #color = color_dict[class_ids[i]]
        color =(1,0,0)

        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]

        # 计算物体的宽度和长度,###掩码的形状为(h,w,N)其中h,w为输入图片的尺寸，N为实例个数。
        '''假设输入图像的大小为 1024x1024，有两个物体 A 和 B：
        掩码 mask_A 是一个 1024x1024 的矩阵，其中只有物体 A 所在的像素值为 1，其余位置（包括物体 B 的像素区域）为 0。
        掩码 mask_B 也是一个 1024x1024 的矩阵，只有物体 B 所在的像素值为 1，其余位置（包括物体 A 的像素区域）为 0。'''
        mask = masks[:, :, i]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5) #返回物体边缘上点的集合

        if len(contours) > 0:
            # 筛选有效的轮廓
            contours = [cnt for cnt in contours if len(cnt) >= 5]
            contours = [cnt.astype(np.float32) for cnt in contours]

            # 选择最大轮廓
            contour = max(contours, key=cv2.contourArea)

            # 构造矩阵 D
            x = contour[:, 0]
            y = contour[:, 1]
            D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T

            # 构造矩阵 C
            C = np.array([[0, 0, 2, 0, 0],
                          [0, -1, 0, 0, 0],
                          [2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

            # 计算 D^T * D
            DT_D = np.dot(D.T, D)

            # 求解特征值问题
            eigvals, eigvecs = eig(DT_D, C)

            # 选择最小的特征值对应的特征向量
            A = eigvecs[:, np.argmin(eigvals)]

            # 提取椭圆参数
            a, b, c, d, e, f = A

            # 计算椭圆的中心
            center_x = (c * e - b * d) / (b ** 2 - 4 * a * c)
            center_y = (a * e - b * d) / (b ** 2 - 4 * a * c)
            center = (center_x, center_y)

            # 计算椭圆的轴长和旋转角度
            temp = np.sqrt((a - c) ** 2 + 4 * b ** 2)
            axis_1 = np.sqrt(2 * (a + c + temp))
            axis_2 = np.sqrt(2 * (a + c - temp))
            angle = 0.5 * np.arctan2(2 * b, (a - c))

            # 确保 major_axis 和 minor_axis 的赋值正确
            major_axis = max(axis_1, axis_2)
            minor_axis = min(axis_1, axis_2)

            # 输出椭圆的长短轴
            object_length = minor_axis
            object_width = major_axis

            a = sigmoid(object_width - 33.3) * (object_width / object_length)
            # 根据条件更改颜色
            if a > 0.4 and object_width < 100 and object_length < 100:
                color = (0, 1, 0)  # 设置为绿色

            # Check for objects with width and length > 100
            elif object_width > 250 and object_length > 250:
                # 进行180度中心旋转相似度检测
                similarity_score = check_center_rotation_similarity(mask, boxes[i])

                if similarity_score > 0.4:
                    pca_angle = pca_object_direction(mask[y1:y2, x1:x2])
                    print(f"矿物中心旋转的相似值为 {similarity_score}，主方向角度为 {pca_angle:.2f} 度")
                    '---------------------以下插入角度-----------'
                else:
                    print(f"矿物中心旋转的相似值为 {similarity_score}")
                    print('无法检测到矿物的c轴')
                continue# 不显示颜色，(0,0,1)显示蓝色
                #color = (0, 0, 1)

        # 绘制边界框
        if show_bbox and color is not None:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # 掩码
        if show_mask and color is not None:
            masked_image = apply_mask(masked_image, mask, color)

        for verts in contours:
            verts = np.fliplr(verts) - 1
            if color is not None:
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()



def display_instances(image, boxes, masks, class_ids, class_names,angles=None,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] 以图像坐标表示的边界框。
    masks: [height, width, num_instances] 每个实例的掩码。
    class_ids: [num_instances] 每个实例的类别 ID。
    class_names: 数据集中类别名称的列表。
    scores: （可选）每个边界框的置信度分数。
    title: （可选）图像的标题。
    show_mask, show_bbox: 是否显示掩码和边界框。
    figsize: （可选）图像的尺寸。
    colors: （可选）为每个对象使用的颜色数组。
    captions: （可选）用于每个对象的字符串列表作为标题。
    """
    #
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # 如果未传递轴（axis），则创建一个并自动调用show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    #生产随机颜色
    #colors = colors or random_colors(N)

    # Generate unique colors for each class
    unique_class_ids = np.unique(class_ids)
    color_map = random_colors(len(unique_class_ids))
    color_dict = {class_id: color for class_id, color in zip(unique_class_ids, color_map)}


    # 显示图像边界外的区域
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        #color = colors[i]
        color=color_dict[class_ids[i]]

        # 边框
        if not np.any(boxes[i]):
            # 没有边界框跳过此实例。可能在图像裁剪时丢失
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none') #linewidth线宽，alpha透明度，linestyle虚线，facecolor填充颜色
            ax.add_patch(p) #添加补丁

        # 标签
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
            angle = angles[i]
            caption = "NO.{} {} {:.2f}\nA: {:.2f}".format(i+1,label, score,angle) if score else "{}\nAngle: {:.2f}".format(label)

        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # 掩码
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # 掩码多边形，填充以确保与图像边缘接触的掩码多边形正确显示
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8) #创建大小比原来掩码大两行两列的掩码
        padded_mask[1:-1, 1:-1] = mask #放在中心
        contours = find_contours(padded_mask, 0.5) #找到边界

        '''自己加的模块'''
        if len(contours) > 0:
            # 筛选有效的轮廓
            contours = [cnt for cnt in contours if len(cnt) >= 5]
            contours = [cnt.astype(np.float32) for cnt in contours]

            # 选择最大轮廓
            contour = max(contours, key=cv2.contourArea)

            # 构造矩阵 D
            x = contour[:, 0]
            y = contour[:, 1]
            D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T

            # 构造矩阵 C
            C = np.array([[0, 0, 2, 0, 0],
                          [0, -1, 0, 0, 0],
                          [2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

            # 计算 D^T * D
            DT_D = np.dot(D.T, D)

            # 求解特征值问题
            eigvals, eigvecs = eig(DT_D, C)

            # 选择最小的特征值对应的特征向量
            A = eigvecs[:, np.argmin(eigvals)]

            # 提取椭圆参数
            a, b, c, d, e, f = A

            # 计算椭圆的中心
            center_x = (c * e - b * d) / (b ** 2 - 4 * a * c)
            center_y = (a * e - b * d) / (b ** 2 - 4 * a * c)
            center = (center_x, center_y)

            # 计算椭圆的轴长和旋转角度
            temp = np.sqrt((a - c) ** 2 + 4 * b ** 2)
            axis_1 = np.sqrt(2 * (a + c + temp))
            axis_2 = np.sqrt(2 * (a + c - temp))
            angle = 0.5 * np.arctan2(2 * b, (a - c))

            # 确保 major_axis 和 minor_axis 的赋值正确
            major_axis = max(axis_1, axis_2)
            minor_axis = min(axis_1, axis_2)

            # 输出椭圆的长短轴
            object_length = minor_axis
            object_width = major_axis

            print(f"第{i + 1}个裂变径迹的长度为{object_length}，宽度为{object_width}，长宽比为{object_length / object_width:.2f}")




        for verts in contours:
            # 减去填充并将 (y, x) 转换为 (x, y)
            verts = np.fliplr(verts) - 1 #转换
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()

def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """于同一张图上展示GT和预测实例的差异"""
    # 将预测结果与真实标注匹配
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # 真实值 = 绿色. 预测值 = 红色
    colors = [(0, 1, 0, .8)] * len(gt_match)\
           + [(1, 0, 0, 1)] * len(pred_match)
    # 合并GT和预测值
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # 每个实例的标题显示 scores/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
    # 设置标题
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # 调用display_instances函数
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)

    from sklearn.metrics import precision_recall_fscore_support

    def evaluate_model(gt_class_ids, pred_class_ids, overlaps, pred_scores, iou_threshold=0.5):
        # 计算真阳性、假阳性和假阴性
        true_positives = []
        false_positives = []
        false_negatives = []

        for i in range(len(gt_class_ids)):
            matched = pred_match[i] > -1
            if matched:
                true_positives.append(1)
                false_negatives.append(0)
            else:
                false_negatives.append(1)

        for j in range(len(pred_class_ids)):
            if pred_match[j] == -1:
                false_positives.append(1)
            else:
                false_positives.append(0)

        # 计算精度、召回率和 F1 分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_class_ids, pred_class_ids, average='weighted')

        return precision, recall, f1


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """显示给定图像以及图像中若干类的最显著的掩码"""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """绘制精确度-召回率曲线。

    AP: IoU >= 0.5 时的平均精确度
    precisions: 精确度值列表
    recalls: 召回率值列表
    """
    # 绘制P-R曲线
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """绘制一个网格显示真实对象的分类情况。

    gt_class_ids: [N] 整数，真实类别 ID
    pred_class_id: [N] 整数，预测的类别 ID
    pred_scores: [N] 浮点数，预测类别的概率分数
    overlaps: [pred_boxes, gt_boxes] 预测框和真实框的 IoU 重叠度
    class_names: 数据集中所有类别名称的列表
    threshold: 浮点数，预测一个类别所需的概率阈值
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with different
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominent each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)
