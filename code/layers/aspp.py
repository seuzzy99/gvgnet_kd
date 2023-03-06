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

from .blocks import darknet_conv


class aspp_decoder(nn.Module):
    def __init__(self, planes, hidden_planes, out_planes):
        super().__init__()
        self.conv0 = darknet_conv(planes, hidden_planes, ksize=1, stride=1)
        self.conv1 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=6)
        self.conv2 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=12)
        self.conv3 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=18)
        self.conv4 = darknet_conv(planes, hidden_planes, ksize=1, stride=1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        # self.out_proj = nn.Conv2d(hidden_planes * 5, out_planes, 1)

        self.low_feature = darknet_conv(512, 256, 1)
        self.conv_1 = nn.Conv2d(hidden_planes * 5, 256, 1)
        self.conv_2 = nn.Conv2d(512, 1, 3, padding=1)

        self.upsample_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, input):
        # mid [N, 512, 26, 26]
        # bot [N, 512, 52, 52]
        mid = input[0]
        bot = input[1]
        # encoder
        _, _, h, w = mid.size()
        b0 = self.conv0(mid)
        b1 = self.conv1(mid)
        b2 = self.conv2(mid)
        b3 = self.conv3(mid)
        b4 = self.conv4(self.pool(mid)).repeat(1, 1, h, w)
        mid = torch.cat([b0, b1, b2, b3, b4], 1)  # [N, 1280, 26, 26]

        # decoder部分
        mid = self.conv_1(mid)  # [N, 256, 26, 26]
        mid = self.upsample_2(mid)  # [N, 256, 52, 52]
        low_feature = self.low_feature(bot)  # [N, 256, 52, 52]
        bot = torch.cat([mid, low_feature], 1)  # [N, 512, 52, 52]
        bot = self.conv_2(bot)  # [N, 1, 52, 52]

        return bot

if __name__ == '__main__':

    model = aspp_decoder(512, 256, 1)

    mid = torch.randn(2, 512, 26, 26)
    bot = torch.randn(2, 512, 52, 52)

    out = model([mid, bot])

    print(out.shape)