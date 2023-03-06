# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import cv2

def label2yolobox(labels, info_img, maxsize, lrflip=False):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    h, w, nh, nw, dx, dy, _ = info_img
    x1 = labels[:, 0] / w
    y1 = labels[:, 1] / h
    x2 = (labels[:, 0] + labels[:, 2]) / w
    y2 = (labels[:, 1] + labels[:, 3]) / h
    labels[:, 0] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 1] = (((y1 + y2) / 2) * nh + dy) / maxsize
    labels[:, 2] *= nw / w / maxsize
    labels[:, 3] *= nh / h / maxsize
    labels[:,:4]=np.clip(labels[:,:4],0.,0.99)
    if lrflip:
        labels[:, 0] = 1 - labels[:, 0]
    return labels


def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # h, w, nh, nw, dx, dy,_ = info_img
    # y1, x1, y2, x2 = box
    # box_h = ((y2 - y1) / nh) * h
    # box_w = ((x2 - x1) / nw) * w
    # y1 = ((y1 - dy) / nh) * h
    # x1 = ((x1 - dx) / nw) * w
    # label = [y1, x1, y1 + box_h, x1 + box_w]
    h, w, nh, nw, dx, dy, _ = info_img
    x1, y1, x2, y2 = box[:4]
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [x1, y1,x1 + box_w, y1 + box_h]
    return np.concatenate([np.array(label),box[4:]])


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


# 计算头部图像的二值图（224，224）
def get_head_box_channel(x_min, y_min, x_max, y_max, width, height, resolution):
    head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution - 1)

    head_channel = np.zeros((resolution, resolution), dtype=np.float32)
    head_channel[head_box[1]:head_box[3], head_box[0]:head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


# 计算2D高斯热力图
def draw_labelmap(img, pt, sigma):
    img = to_numpy(img)

    # 检查高斯核在图像内
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        return to_torch(img)

    # 生成高斯热力图
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # 归一化，使得热力图中心点的值为1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # 可用的高斯尺寸
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # 图像尺寸
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    img = img/np.max(img)
    # np.save("./gaze.npy", img)

    return to_torch(img)

def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    x, y = map(int,[gaze_pts[0]*w.float(), gaze_pts[1]*h.float()])
    x = min(x, w-1)
    y = min(y, h-1)
    target_map[y, x] = 1

    return target_map