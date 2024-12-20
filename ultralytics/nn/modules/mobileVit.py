# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "MobileViTBackbone",
)

import torch
import torch.nn as nn
import timm

class MobileViTBackbone(nn.Module):
    def __init__(self, step=1, layer1_r=1, layer2_r=1):
        super(MobileViTBackbone, self).__init__()
        # print("加载 MobileViT 模型。。。。。。。。。")
        # 加载 MobileViT 模型
        mobilevit = timm.create_model('mobilevit_s', pretrained=False)

        self.layer1_r = layer1_r #两个通道的比例
        self.layer2_r = layer2_r

        # 第一层
        self.layer1_1 = list(mobilevit.children())[0]
        self.layer1_2 = list(mobilevit.children())[1][0:3]

        # 第二层
        self.layer2_1 = list(mobilevit.children())[1][3]

        # 第三层
        self.layer3_1 = list(mobilevit.children())[1][4:]
        self.layer3_2 = list(mobilevit.children())[2]

        if step == 1:
            self.layer = nn.Sequential(
                self.layer1_1,
                *self.layer1_2,
            )
        elif step == 2:
            self.layer = self.layer2_1
        elif step == 3:
            self.layer = nn.Sequential(
                *self.layer3_1, self.layer3_2
            )

    def forward(self, x):

        maxChannelLen = x.shape[1]

        l1 = self.layer1_r
        lys = self.layer1_r + self.layer2_r

        n_l1 = maxChannelLen*l1 // lys
        # print("输入数据的形状", x.shape, maxChannelLen, n_l1)

        visible_x = x[:, :n_l1, :, :]
        lr_x = x[:, n_l1:, :, :]


        lr_x_out = self.layer(lr_x)
        visible_x_out = self.layer(visible_x)

        # print("lr_x的shape： ", lr_x_out.shape)
        # print("visible_x_out的shape： ", visible_x_out.shape)

        output = torch.concat((visible_x_out, lr_x_out), dim=1)
        # output = self.layer(x)
        # print("输出数据的形状", output.shape)
        # return self.layer(x)
        return output

# if __name__ == '__main__':
#     from torchinfo import summary
#     # model = CustomBackbone()
#     model = MobileViTBackbone(3)
#     # model = timm.create_model('mobilevit_s', pretrained=False)
#
#     summary(model, input_size=(1, 128, 40, 40))
#     pass