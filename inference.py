import os
import time
import re
import yaml

import numpy as np
import cv2
import random
import en_vectors_web_lg
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from code.models.model import GVGNet_KD
from code.models.gaze_teacher import GazeModel_Teacher
from code.models.gaze_student import GazeModel_Student, SimKD

# change test

# 计算头部图像的二值图（224，224）
def get_head_box_channel(x_min, y_min, x_max, y_max, width, height, resolution):
    head_box = np.array([x_min/width, y_min/height, x_max/width, y_max/height])*resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution-1)

    head_channel = np.zeros((resolution,resolution), dtype=np.float32)
    head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


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


def process_data(img_rgb, image_gaze, head):
    h, w, _ = img_rgb.shape
    img_size = 416
    new_ar = w / h

    if new_ar < 1:
        nh = img_size
        nw = nh * new_ar
    else:
        nw = img_size
        nh = nw / new_ar

    nw, nh = int(nw), int(nh)

    dx = (img_size - nw) // 2
    dy = (img_size - nh) // 2

    # 原始图像
    ori_image = img_rgb
    ori_image = cv2.resize(ori_image, (nw, nh))
    sized_ori = np.ones((img_size, img_size, 3), dtype=np.uint8) * 127
    sized_ori[dy:dy + nh, dx:dx + nw, :] = ori_image

    # RGB图像
    transforms = get_img_transform()
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (nw, nh))
    sized = np.ones((img_size, img_size, 3), dtype=np.uint8) * 127
    sized[dy:dy + nh, dx:dx + nw, :] = img
    sized = transforms(sized)

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
    transforms_gaze = get_gaze_transform()
    width, height = image_gaze.size
    x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
    head_img = get_head_box_channel(x_min, y_min, x_max, y_max, width, height, 224).unsqueeze(0)
    face = image_gaze.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    image_gaze = transforms_gaze(image_gaze)
    face = transforms_gaze(face)

    # info
    info_img = (h, w, nh, nw, dx, dy, 0)

    return sized, image_gaze, info_img, sized_ori, head_img, face


def load_refs(sentence, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    for ix, word in enumerate(sentence):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def build_dict(txt_path):
    token_to_ix = {'PAD': 0, 'UNK': 1, 'CLS': 2}
    pretrained_emb = []
    spacy_tool = en_vectors_web_lg.load()
    pretrained_emb.append(spacy_tool('PAD').vector)
    pretrained_emb.append(spacy_tool('UNK').vector)
    pretrained_emb.append(spacy_tool('CLS').vector)

    with open(txt_path) as f:
        lines = f.readlines()

    for line in lines:
        line_data = line.split()
        ref = ""
        for i in range(5, len(line_data)):
            ref = ref + line_data[i] + " "

        words = re.sub(r"([.,'!?\"()*#:;])", '', ref.lower()).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    ix_to_token = {}
    for item in token_to_ix:
        ix_to_token[token_to_ix[item]] = item

    print("token_to_ix:\n", token_to_ix)

    return token_to_ix, ix_to_token, pretrained_emb


def nls(pred_seg, pred_box, weight_score=None, lamb_au=-1., lamb_bu=2, lamb_ad=1., lamb_bd=0):
    if weight_score is not None:
        # asnls
        mask = torch.ones_like(pred_seg) * weight_score.unsqueeze(1).unsqueeze(1) * lamb_ad + lamb_bd
        pred_box = pred_box[:, :4].long()
        for i in range(pred_seg.size()[0]):
            mask[i, pred_box[i, 1]:pred_box[i, 3] + 1, pred_box[i, 0]:pred_box[i, 2] + 1, ...] = weight_score[i].item() * lamb_au + lamb_bu
    else:
        # hard-nls
        mask = torch.zeros_like(pred_seg)
        pred_box = np.array(pred_box[:, :4], dtype=np.int32)
        for i in range(pred_seg.size()[0]):
            mask[i, pred_box[i][1]:pred_box[i][3] + 1, pred_box[i][0]:pred_box[i][2] + 1] = 1.
    return pred_seg * mask


if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda")

    with open('./configs/config.yaml', encoding='utf-8') as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    # 词向量预处理
    print("building dict...")
    txt_path = "./data/anns/train.txt"
    token_to_ix, ix_to_token, pretrained_emb = build_dict(txt_path)
    token_size = 17
    print("done!")

    # 模型搭建
    print('Building model ...')
    teacher_path = "./data/weights/model_teacher.pt"
    model_t = GazeModel_Teacher()
    model_t.load_state_dict(torch.load(teacher_path)['model'])
    print('Teacher model Done !')

    gvg_path = "./output/det_best.pth"
    model_s = GazeModel_Student()
    model_kd = SimKD(s_n=256, t_n=512, factor=2)
    model_s.load_state_dict(torch.load(gvg_path)['gaze_s_state'])
    model_kd.load_state_dict(torch.load(gvg_path)['gaze_kd_state'])
    print('Student model Done !')

    # MCN网络
    cfg["model"]["visual_pretrained"] = False
    model_mcn = GVGNet_KD(pretrained_emb, token_size, cfg)
    model_mcn.load_state_dict(torch.load(gvg_path)['mcn_state'])
    print('MCN model Done !')

    # 整理模型
    module_list = nn.ModuleList([])
    module_list.append(model_mcn)
    module_list.append(model_s)
    module_list.append(model_kd)
    module_list.append(model_t)
    module_list.cuda()

    for module in module_list:
        module.eval()

    with torch.no_grad():
        start = time.time()

        # 数据读取
        image_path = "./data/images/test/9.png"
        img_rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_gaze = Image.open(os.path.join(image_path))
        image_gaze = image_gaze.convert('RGB')
        head = [259, 45, 378, 167]
        sent = "glue stick"

        # 文本数据处理
        max_token = 3
        ref = load_refs(sent, token_to_ix, max_token)
        ref = torch.from_numpy(ref).long()

        # 图片数据处理
        img_rgb, image_gaze, info_img, ori_img, head_img, face = process_data(img_rgb, image_gaze, head)
        info_img = np.array(info_img)

        img_rgb = torch.unsqueeze(img_rgb, dim=0).to(device)
        ref = torch.unsqueeze(ref, dim=0).to(device)
        image_gaze = torch.unsqueeze(image_gaze, dim=0).to(device)
        head_img = torch.unsqueeze(head_img, dim=0).to(device)
        face = torch.unsqueeze(face, dim=0).to(device)

        # 模型调用
        # 凝视估计部分
        fusion_feat_s = model_s(image_gaze, head_img, face)
        with torch.no_grad():
            heatmap_t, fusion_feat_t, inout = model_t(image_gaze, head_img, face)
        cls_t = model_t.get_deconv()
        trans_feat_s, trans_feat_t, heatmap_s = model_kd(fusion_feat_s, fusion_feat_t, cls_t)

        heatmap_s = heatmap_s.squeeze(1)

        # MCN部分
        box, mask = model_mcn(img_rgb, ref, trans_feat_s)
        box = box.squeeze(1).cpu().numpy()
        pred_box_vis = box.copy()

        # 计算mask
        mask = nls(mask, box)
        mask = mask.cpu().numpy()
        mask_img = (mask[0, None]*255).astype(np.uint8).transpose((1, 2, 0))

        # cv2.imshow("seg", mask_img)
        # cv2.waitKey(0)

        end = time.time()
        print("time:", end-start)

        # 可视化
        left, top, right, bottom, _ = (pred_box_vis[0]).astype('int32')
        colors = [(255, 0, 0), (0, 255, 0), (0, 191, 255)]
        cv2.rectangle(ori_img, (left, top), (right, bottom), (0,0,255), 3)
        cv2.putText(ori_img, sent, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[2], 1)

        # cv2.imwrite("../output/det_real/{}.png".format(cnt), ori_img)

        cv2.imshow("det", ori_img)
        cv2.waitKey(0)

        mask_out = np.zeros((416, 416, 3), dtype=np.uint8)
        mask_np = np.array(mask_img, dtype=np.uint8)
        for m in range(416):
            for n in range(416):
                if mask_np[m][n] >= 128:
                    mask_out[m][n][0] = 0
                    mask_out[m][n][1] = 0
                    mask_out[m][n][2] = 255

        cv2.addWeighted(ori_img, 0.7, mask_out, 0.5, 0, ori_img)

        cv2.imshow("all", ori_img)
        cv2.waitKey(0)

        # cv2.imwrite("../output/all_s/{}.png".format(cnt), ori_img)

