"""
Mask R-CNN
Display and visualization functions.

Copyright (c) 2017 Matterport, Inc
According to the MIT license authorization (detailed information can be found in the LICENSE file)
Author: Waleed Abdulla
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
    Display a given set of images and selectively attach titles.
    images:  A list or array of image tensors in HWC format.
    titles:  Optional. To display a list of titles next to each image.
    cols:  The number of images displayed per line.
    cmap:  Optional. The color used. For example, “Blues”。
    norm:  Optional. A Normalize instance used to map values to colors.
    interpolation:  Optional. Image interpolation method used for display.
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
    """Generate random colors, filter fixed colors, and keep the total number unchanged."""
    random.seed(0)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]  #(Color tone, saturation, brightness)
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))  # Convert to RGB
    random.shuffle(colors)  # Shuffle the order

    # Filter out colors that overlap with fixed colors
    fixed_colors = [(0, 1, 0), (0, 0, 1)]  # Green, blue

    def is_similar(color1, color2, threshold=0.1):
        """Determine if two colors are similar"""
        return all(abs(c1 - c2) < threshold for c1, c2 in zip(color1, color2))

    # Filter random colors
    filtered_colors = [color for color in colors if not any(is_similar(color, fc) for fc in fixed_colors)]

    # Calculate the number of colors to be added
    num_to_add = N - len(filtered_colors)

    # If the number of colors after filtering is less than N, supplement
    if num_to_add == 1:
        # Randomly select some colors to supplement
        new_colors1 = (1,1,0) #yellow
        filtered_colors.append(new_colors1)
    elif num_to_add == 2:
        # Randomly select some colors to supplement
        new_colors1 = (1,1,0)
        new_colors2 = (0.5,0,0.5) #purple
        filtered_colors.append(new_colors1)
        filtered_colors.append(new_colors2)
    # Define 7 fixed colors (RGB range from 0 to 1)


    return filtered_colors



def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])  #Change color for 1, maintain for 0
    return image


import numpy as np
from skimage.metrics import structural_similarity as ssim
def display_instances_2(image, boxes, masks, class_ids, class_names, angles=None,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    N = boxes.shape[0]
    FT_length={} #Storage of fission track intercept length
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
        # Retrieve the coordinates of non-zero pixels in the mask
        coords = np.column_stack(np.where(mask > 0))

        # If there is no object, return None
        if coords.shape[0] == 0:
            return None

        # Using PCA to analyze the principal direction of an object
        pca = PCA(n_components=2)
        pca.fit(coords)

        # Principal Component Direction: First Principal Component Vector
        principal_direction = pca.components_[0]

        # Calculate the angle of the main direction (unit: degrees)
        angle = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi
        if angle>0:
            angle=angle-180
        # The angle is clockwise from the x-axis square to the direction perpendicular to the longest axis, with a value of - (clockwise)

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

        # Calculate the overlapping area (intersection)
        overlap = np.logical_and(object_mask, rotated_object_mask).sum()

        # Calculate union
        union = np.logical_or(object_mask, rotated_object_mask).sum()

        # If the union is 0, return the overlap as 0 to avoid dividing by 0
        if union == 0:
            return 0

        # Calculate IoU (Intersection over Union)
        iou = overlap / union
        return iou

    def calculate_length(pca_angle,angle,object_length, object_width):
        '------------------The following insertion angles-------------'
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

        # Calculate the width and length of the object. The shape of the mask is (h, w, N), where h and w are the dimensions of the input image and N is the number of instances.
        # Assuming the size of the input image is 1024x1024, there are two objects A and B:
        # Mask Mask-A is a 1024x1024 matrix, where only the pixel value of object A is 1, and the rest of the positions (including the pixel area of object B) are 0.
        # Mask Mask-B is also a 1024x1024 matrix, with only the pixel value of object B being 1, and the rest of the positions (including the pixel area of object A) being 0'''
        mask = masks[:, :, i]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5) #Return the set of points on the edge of an object
        pca_angle=0
        if len(contours) > 0:
            # Filter effective contours
            contours = [cnt for cnt in contours if len(cnt) >= 5]
            contours = [cnt.astype(np.float32) for cnt in contours]

            # Select the maximum contour
            contour = max(contours, key=cv2.contourArea)

            # Construct matrix D
            x = contour[:, 0]
            y = contour[:, 1]
            D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T

            # Construct matrix C
            C = np.array([[0, 0, 2, 0, 0],
                          [0, -1, 0, 0, 0],
                          [2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

            # count D^T * D
            DT_D = np.dot(D.T, D)

            # Solving eigenvalue problems
            eigvals, eigvecs = eig(DT_D, C)

            # Select the eigenvector corresponding to the smallest eigenvalue
            A = eigvecs[:, np.argmin(eigvals)]

            # Extract ellipse parameters
            a, b, c, d, e, f = A

            # Calculate the center of an ellipse
            center_x = (c * e - b * d) / (b ** 2 - 4 * a * c)
            center_y = (a * e - b * d) / (b ** 2 - 4 * a * c)
            center = (center_x, center_y)

            # Calculate the axis length and rotation angle of an ellipse
            temp = np.sqrt((a - c) ** 2 + 4 * b ** 2)
            axis_1 = np.sqrt(2 * (a + c + temp))
            axis_2 = np.sqrt(2 * (a + c - temp))
            angle = 0.5 * np.arctan2(2 * b, (a - c))

            # Ensure the correct assignment of major axis and minor axis values
            major_axis = max(axis_1, axis_2)
            minor_axis = min(axis_1, axis_2)

            # Output the major and minor axes of an ellipse
            object_length = minor_axis
            object_width = major_axis

            a = sigmoid(object_width - 33.3) * (object_width / object_length)
            # Change color according to conditions
            if a > 0.38 and object_width < 100 and object_length < 100:
                color = (0, 1, 0)  # Set to green

            # Check for objects with width and length > 100
            elif object_width > 250 and object_length > 250:
                # Perform 180 degree center rotation similarity detection
                similarity_score = check_center_rotation_similarity(mask, boxes[i])

                if similarity_score > 0.4:
                    pca_angle = pca_object_direction(mask[y1:y2, x1:x2])
                    print(f"The similarity value of mineral center rotation is {similarity_score}，The main direction angle is {pca_angle:.2f} 度")

                else:
                    print(f"The similarity value of mineral center rotation is {similarity_score}")
                    print('Unable to detect the c-axis of minerals')
                continue #Do not display color, (0,0,1) displays blue
                #color = (0, 0, 1)
            length=calculate_length(pca_angle,angle,object_length, object_width)
            FT_length[i+1] = length


        # Draw a bounding box
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
        # mask
        if show_mask and color is not None:
            masked_image = apply_mask(masked_image, mask, color)

        for verts in contours:
            verts = np.fliplr(verts) - 1
            if color is not None:
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    #print(FT_length)
    df = pd.DataFrame(list(FT_length.items()), columns=['Index', 'FT_length'])
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
        # Retrieve the coordinates of non-zero pixels in the mask
        coords = np.column_stack(np.where(mask > 0))

        # If there is no object, return None
        if coords.shape[0] == 0:
            return None

        # Using PCA to analyze the principal direction of an object
        pca = PCA(n_components=2)
        pca.fit(coords)

        # Principal Component Direction: First Principal Component Vector
        principal_direction = pca.components_[0]

        # Calculate the angle of the main direction (unit: degrees)
        angle = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi
        if angle>0:
            angle=angle-180
        # The angle is clockwise from the x-axis square to the direction perpendicular to the longest axis, with a value of - (clockwise)

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

        # Calculate the overlapping area (intersection)
        overlap = np.logical_and(object_mask, rotated_object_mask).sum()

        # Calculate union
        union = np.logical_or(object_mask, rotated_object_mask).sum()

        # If the union is 0, return the overlap as 0 to avoid dividing by 0
        if union == 0:
            return 0

        # Calculate IoU (Intersection over Union)
        iou = overlap / union
        return iou

    for i in range(N):
        #color = color_dict[class_ids[i]]
        color =(1,0,0)

        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]

        # Calculate the width and length of the object. The shape of the mask is (h, w, N), where h and w are the dimensions of the input image and N is the number of instances.
        # Assuming the size of the input image is 1024x1024, there are two objects A and B:
        # Mask Mask-A is a 1024x1024 matrix, where only the pixel value of object A is 1, and the rest of the positions (including the pixel area of object B) are 0.
        # Mask Mask-B is also a 1024x1024 matrix, with only the pixel value of object B being 1, and the rest of the positions (including the pixel area of object A) being 0'''
        mask = masks[:, :, i]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5) # Return the set of points on the edge of an object

        if len(contours) > 0:
            # Filter effective contours
            contours = [cnt for cnt in contours if len(cnt) >= 5]
            contours = [cnt.astype(np.float32) for cnt in contours]

            # Select the maximum contour
            contour = max(contours, key=cv2.contourArea)

            # Construct matrix D
            x = contour[:, 0]
            y = contour[:, 1]
            D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T

            # Construct matrix C
            C = np.array([[0, 0, 2, 0, 0],
                          [0, -1, 0, 0, 0],
                          [2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

            # calculate D^T * D
            DT_D = np.dot(D.T, D)

            # Solving eigenvalue problems
            eigvals, eigvecs = eig(DT_D, C)

            # Select the eigenvector corresponding to the smallest eigenvalue
            A = eigvecs[:, np.argmin(eigvals)]

            # Extract ellipse parameters
            a, b, c, d, e, f = A

            # Calculate the center of an ellipse
            center_x = (c * e - b * d) / (b ** 2 - 4 * a * c)
            center_y = (a * e - b * d) / (b ** 2 - 4 * a * c)
            center = (center_x, center_y)

            # Calculate the axis length and rotation angle of an ellipse
            temp = np.sqrt((a - c) ** 2 + 4 * b ** 2)
            axis_1 = np.sqrt(2 * (a + c + temp))
            axis_2 = np.sqrt(2 * (a + c - temp))
            angle = 0.5 * np.arctan2(2 * b, (a - c))

            # Ensure the correct assignment of major axis and minor axis values
            major_axis = max(axis_1, axis_2)
            minor_axis = min(axis_1, axis_2)

            # Output the major and minor axes of an ellipse
            object_length = minor_axis
            object_width = major_axis

            a = sigmoid(object_width - 33.3) * (object_width / object_length)
            # Change color according to conditions
            if a > 0.4 and object_width < 100 and object_length < 100:
                color = (0, 1, 0)  # Set to green

            # Check for objects with width and length > 100
            elif object_width > 250 and object_length > 250:
                # Perform 180 degree center rotation similarity detection
                similarity_score = check_center_rotation_similarity(mask, boxes[i])

                if similarity_score > 0.4:
                    pca_angle = pca_object_direction(mask[y1:y2, x1:x2])
                    print(f"The similarity value of mineral center rotation is {similarity_score}，The main direction angle is {pca_angle:.2f} ")
                    '------------------The following insertion angles---------------'
                else:
                    print(f"The similarity value of mineral center rotation is {similarity_score}")
                    print('Unable to detect the c-axis of minerals')
                continue# Do not display color, (0,0,1) displays blue
                # color = (0, 0, 1)

        # Draw a bounding box
        if show_bbox and color is not None:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Mask
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
    Boxes: [num_instance, (y1, x1, y2, x2, class_i)] are bounding boxes represented by image coordinates.
    Masks: [height, width, num_instance] Masks for each instance.
    Class_ids: [num_instance] The class ID of each instance.
    class_names:  List of category names in the dataset.
    scores:  (Optional) Confidence score for each bounding box.
    title:  (Optional) The title of the image.
    show_mask, show_bbox:  Whether to display masks and bounding boxes.
    figsize:  (Optional) Size of the image.
    colors:  (Optional) Color array used for each object.
    captions:  (Optional) Use a string list for each object as the title.
    """
    #
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If the axis is not passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Produce random colors
    # colors = colors or random_colors(N)

    # Generate unique colors for each class
    unique_class_ids = np.unique(class_ids)
    color_map = random_colors(len(unique_class_ids))
    color_dict = {class_id: color for class_id, color in zip(unique_class_ids, color_map)}


    # Display the area outside the image boundary
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        #color = colors[i]
        color=color_dict[class_ids[i]]

        # box
        if not np.any(boxes[i]):
            # There is no bounding box to skip this instance. Possible loss during image cropping
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p) # Add patch

        # label
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

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask polygon, filled to ensure that the mask polygon in contact with the image edge is displayed correctly
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8) #Create a mask that is two rows and two columns larger in size than the original mask
        padded_mask[1:-1, 1:-1] = mask # Placed in the center
        contours = find_contours(padded_mask, 0.5) # Find the boundary

        if len(contours) > 0:
            # Filter effective contours
            contours = [cnt for cnt in contours if len(cnt) >= 5]
            contours = [cnt.astype(np.float32) for cnt in contours]

            # Select the maximum contour
            contour = max(contours, key=cv2.contourArea)

            # Construct matrix D
            x = contour[:, 0]
            y = contour[:, 1]
            D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T

            # Construct matrix C
            C = np.array([[0, 0, 2, 0, 0],
                          [0, -1, 0, 0, 0],
                          [2, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

            # calulate D^T * D
            DT_D = np.dot(D.T, D)

            # Solving eigenvalue problems
            eigvals, eigvecs = eig(DT_D, C)

            # Select the eigenvector corresponding to the smallest eigenvalue
            A = eigvecs[:, np.argmin(eigvals)]

            # Extract ellipse parameters
            a, b, c, d, e, f = A

            # Calculate the center of an ellipse
            center_x = (c * e - b * d) / (b ** 2 - 4 * a * c)
            center_y = (a * e - b * d) / (b ** 2 - 4 * a * c)
            center = (center_x, center_y)

            # Calculate the axis length and rotation angle of an ellipse
            temp = np.sqrt((a - c) ** 2 + 4 * b ** 2)
            axis_1 = np.sqrt(2 * (a + c + temp))
            axis_2 = np.sqrt(2 * (a + c - temp))
            angle = 0.5 * np.arctan2(2 * b, (a - c))

            # Ensure the correct assignment of major axis and minor axis values
            major_axis = max(axis_1, axis_2)
            minor_axis = min(axis_1, axis_2)

            # Output the major and minor axes of an ellipse
            object_length = minor_axis
            object_width = major_axis

            print(f"The length of the {i + 1}th fission track is {object_length}, the width is {object_width}, and the aspect ratio is {object_length / object_width:.2f}.")




        for verts in contours:
            # Subtract padding and convert (y, x) to (x, y)
            verts = np.fliplr(verts) - 1 # convert
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
    """Show the difference between GT and predicted instances on the same image"""
    #Match the predicted results with the actual annotations
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # True value=green Predicted value=Red
    colors = [(0, 1, 0, .8)] * len(gt_match)\
           + [(1, 0, 0, 1)] * len(pred_match)
    # Merge GT and predicted values
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # The title of each instance displays scores/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
    # Set Title
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)

    from sklearn.metrics import precision_recall_fscore_support

    def evaluate_model(gt_class_ids, pred_class_ids, overlaps, pred_scores, iou_threshold=0.5):
        # Calculate true positives, false positives, and false negatives
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

        # Calculate accuracy, recall rate, and F1 score
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
    """Display the most prominent masks for a given image and several classes within the image"""
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
    """Draw a precision recall curve.

    AP: Average accuracy when IoU>=0.5
    precisions:  Precision Value List
    recalls:  Recall Value List
    """
    # Draw P-R curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid to display the classification of real objects.

    Gt_class_ids: [N] integer, real category ID
    Pred_class_id: [N] integer, predicted category ID
    Pred_Scores: [N] Floating point number, probability score for predicting categories
    Overlaps: IoU overlap between predicted boxes and real boxes [pred-boxes, gt-boxes]
    class_names:  List of all category names in the dataset
    threshold:  Floating point number, the probability threshold required to predict a category
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
