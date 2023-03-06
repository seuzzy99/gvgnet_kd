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

import torch
import torch.nn as nn


class IOUWH_loss(nn.Module): #used for anchor guiding
    def __init__(self, reduction='none'):
        super(IOUWH_loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        orig_shape = pred.shape
        pred = pred.view(-1,4)
        target = target.view(-1,4)
        target[:,:2] = 0
        tl = torch.max((target[:, :2]-pred[:,2:]/2),
                      (target[:, :2] - target[:, 2:]/2))

        br = torch.min((target[:, :2]+pred[:,2:]/2),
                      (target[:, :2] + target[:, 2:]/2))

        area_p = torch.prod(pred[:,2:], 1)
        area_g = torch.prod(target[:,2:], 1)

        en = (tl< br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br-tl, 1) * en
        U = area_p+area_g-area_i+ 1e-16
        iou= area_i / U

        loss = 1-iou**2
        if self.reduction =='mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class IOUloss(nn.Module):
    def __init__(self, reduction='none'):
        super(IOUloss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        orig_shape = pred.shape
        pred = pred.view(-1,4)
        target = target.view(-1,4)
        tl = torch.max((pred[:, :2]-pred[:,2:]/2),
                      (target[:, :2] - target[:, 2:]/2))
        br = torch.min((pred[:, :2]+pred[:,2:]/2),
                      (target[:, :2] + target[:, 2:]/2))

        area_p = torch.prod(pred[:,2:], 1)
        area_g = torch.prod(target[:,2:], 1)

        en = (tl< br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br-tl, 1) * en
        iou= (area_i) / (area_p+area_g-area_i+ 1e-16)

        loss = 1-iou**2
        if self.reduction =='mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, reduction='none'):
        super(SoftDiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = logits
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num

        if self.reduction =='mean':
            score = score.mean()
        elif self.reduction == 'sum':
            score = score.sum()

        return score
