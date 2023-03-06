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

import os
import cv2
from PIL import Image
import json, re, en_vectors_web_lg, random
import numpy as np

import torch
import torch.utils.data as Data
from torchvision import transforms

from .utils import label2yolobox, get_head_box_channel, draw_labelmap


def get_img_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]))
    return transforms.Compose(transform_list)


def get_gaze_transform():
    transform_list = []
    transform_list.append(transforms.Resize((224, 224)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


class TableGazeDataSet(Data.Dataset):
    def __init__(self, cfg):
        super(TableGazeDataSet, self).__init__()

        self.split = cfg["dataset"]["split"]
        self.input_shape = cfg["dataset"]["input_shape"]
        self.input_gaze_shape = cfg["dataset"]["input_gaze_shape"]
        self.output_gaze_shape = cfg["dataset"]["output_gaze_shape"]
        self.anns_path = cfg["dataset"]["anns_path"]
        self.image_path = cfg["dataset"]["img_path"]
        self.mask_path = cfg["dataset"]["mask_path"]
        self.transforms = get_img_transform()
        self.transforms_gaze = get_gaze_transform()

        # 读取数据集文本
        if self.split == "train":
            with open(self.anns_path["train"]) as f:
                lines = f.readlines()
        else:
            with open(self.anns_path["test"]) as f:
                lines = f.readlines()
        self.lines = lines
        self.data_size = len(self.lines)
        print(' ========== Dataset size:', self.data_size)

        # 构建文本embedding
        with open(self.anns_path["train"]) as f:
            lines_emb = f.readlines()
        self.lines_emb = lines_emb
        self.token_to_ix, self.ix_to_token, self.pretrained_emb, max_token = self.tokenize()
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        self.max_token = cfg["dataset"]["max_token_length"]
        if self.max_token == -1:
            self.max_token = max_token

        print('Max token length:', max_token, 'Trimmed to:', self.max_token)
        print('Finished!')
        print('')

    # 词向量初始化
    def tokenize(self):
        token_to_ix = {'PAD': 0, 'UNK': 1, 'CLS': 2}
        pretrained_emb = []
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for line in self.lines_emb:
            line_data = line.split()
            stop = len(line_data)
            for i in range(1, len(line_data)):
                if (line_data[i] == '~'):
                    stop = i
                    break
            sentences = []
            sent_stop = stop + 1
            for i in range(stop + 1, len(line_data)):
                if line_data[i] == '~':
                    sentences.append(line_data[sent_stop:i])
                    sent_stop = i + 1
            sentences.append(line_data[sent_stop:len(line_data)])
            sent = sentences[0]

            ref = ""
            for i in range(0, len(sent)):
                ref = ref + sent[i] + " "

            words = re.sub(r"([.,'!?\"()*#:;])", '', ref.lower()).replace('-', ' ').replace('/', ' ').split()

            if len(words) > max_token:
                max_token = len(words)

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        ix_to_token = {}
        for item in token_to_ix:
            ix_to_token[token_to_ix[item]] = item

        print("token_to_ix:\n", token_to_ix)

        return token_to_ix, ix_to_token, pretrained_emb, max_token

    def proc_ref(self, ref, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)
        words = re.sub(r"([.,'!?\"()*#:;])", '', ref.lower()).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']
            if ix + 1 == max_token:
                break

        return ques_ix

    def load_refs(self, idx):
        line_data = self.lines[idx].split()
        stop = len(line_data)
        for i in range(1, len(line_data)):
            if (line_data[i] == '~'):
                stop = i
                break
        sentences = []
        sent_stop = stop + 1
        for i in range(stop + 1, len(line_data)):
            if line_data[i] == '~':
                sentences.append(line_data[sent_stop:i])
                sent_stop = i + 1
        sentences.append(line_data[sent_stop:len(line_data)])
        choose_index = np.random.choice(len(sentences))
        sent = sentences[choose_index]

        ref = ""
        for i in range(0, len(sent)):
            ref = ref + sent[i] + " "

        ref = self.proc_ref(ref, self.token_to_ix, self.max_token)
        return ref

    def preprocess_info(self, image, image_gaze, mask, box, iid, head):
        # 处理RGB图像
        h, w, _ = image.shape
        imgsize = self.input_shape[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)

        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

        image = cv2.resize(image, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = image
        sized = self.transforms(sized)

        # 处理凝视输入图像
        # 扩展面部的bbox
        x_min = int(head[0])
        y_min = int(head[1])
        x_max = int(head[2])
        y_max = int(head[3])
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        # 面部位置二值图像head_img和面部裁切图像face
        width, height = image_gaze.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        head_img = get_head_box_channel(x_min, y_min, x_max, y_max, width, height, self.input_gaze_shape).unsqueeze(0)
        face = image_gaze.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        image_gaze = self.transforms_gaze(image_gaze)
        face = self.transforms_gaze(face)

        # 生成凝视热力图标注
        gaze_x = (box[0][0]+box[0][2]/2)
        gaze_y = (box[0][1]+box[0][3]/2)
        gaze_x = gaze_x / 640.0
        gaze_y = gaze_y / 480.0
        gaze_heatmap = torch.zeros(self.output_gaze_shape, self.output_gaze_shape)
        gaze_heatmap = draw_labelmap(gaze_heatmap, [gaze_x * self.output_gaze_shape, gaze_y * self.output_gaze_shape], 2)
        gaze_pts = np.array([gaze_x, gaze_y])

        # 处理info标签
        info_img = (h, w, nh, nw, dx, dy, iid)

        # 读取mask图像
        mask = np.expand_dims(mask, -1).astype(np.float32)
        mask = cv2.resize(mask, (nw, nh))
        mask = np.expand_dims(mask, -1).astype(np.float32)
        sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
        sized_mask[dy:dy + nh, dx:dx + nw, :] = mask
        sized_mask = np.transpose(sized_mask, (2, 0, 1))

        # 计算bbox
        sized_box = label2yolobox(box, info_img, self.input_shape[0])

        return sized, sized_mask, sized_box, info_img, image_gaze, head_img, face, gaze_heatmap, gaze_pts

    def load_img_feats(self, idx):
        line_data = self.lines[idx].split()

        # 读取RGB图像
        img_path = line_data[0]
        image = cv2.imread(os.path.join(self.image_path, img_path), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_gaze = Image.open(os.path.join(self.image_path, img_path))
        image_gaze = image_gaze.convert('RGB')

        # 读取bbox  [[x1, y1, x2, y2, class]] ---> [[class, x, y, w, h]]
        stop = 2
        box_data = np.array([np.array(list(map(int, box.split(',')))) for box in line_data[1:stop]])
        box = np.zeros([1, 5])
        box[0] = box_data[0][:-1]
        box_class = box[0][4]
        box_w = box[0][2] - box[0][0]
        box_h = box[0][3] - box[0][1]
        box_x = box[0][0]
        box_y = box[0][1]
        box[0][0] = box_x
        box[0][1] = box_y
        box[0][2] = box_w
        box[0][3] = box_h
        box[0][4] = box_class

        # 读取head_box
        head = line_data[2].split(',')
        head = (int(head[0]), int(head[1]), int(head[2]), int(head[3]))

        # 读取iid
        iid = int(box_data[0][-2])

        # 读取mask编号
        mask_id = int(img_path.split('/')[1][0:-4])

        # 读取mask图像
        if self.split == "train":
            mask = cv2.imread(os.path.join(self.mask_path, 'train/{}.png'.format(mask_id)), cv2.IMREAD_UNCHANGED)
        else:
            mask = cv2.imread(os.path.join(self.mask_path, 'test/{}.png'.format(mask_id)), cv2.IMREAD_UNCHANGED)
        mask = np.array(mask, dtype=np.float32)

        return image, image_gaze, mask, box, mask_id, iid, head

    def __getitem__(self, idx):

        ref = self.load_refs(idx)
        image, image_gaze, mask, gt_box, mask_id, iid, head = self.load_img_feats(idx)
        image, mask, box, info, image_gaze, head_img, face, gaze_heatmap, gaze_pts = self.preprocess_info(image, image_gaze, mask, gt_box.copy(), iid, head)

        return torch.from_numpy(ref).long(), image, image_gaze, face, head_img, \
               torch.from_numpy(mask).float(), torch.from_numpy(box).float(), torch.from_numpy(gt_box).float(), \
               mask_id, np.array(info), gaze_heatmap, torch.from_numpy(gaze_pts).float()

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)
