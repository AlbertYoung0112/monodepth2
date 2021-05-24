# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from math import ceil
import numpy as np

import torch
import torch as th
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from components import Identity, img2pc_bridge, depth_conv_1d, depthwise, pointwise, weights_init


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, with_pc=False, img_height=224):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.with_pc = with_pc
        if with_pc:
            self.register_pc_encoder()
        self.ecu_pc_channel = 128
        self.img_height = img_height
        ecu_feat_shape = int(ceil(self.img_height / 32))
        ecu_sample_size = 3
        self.ecu = ECUSmall(
            img_in_channel=self.num_ch_enc[-1],
            pc_in_channel=self.ecu_pc_channel,
            ecu_channel=self.num_ch_enc[-1],
            out_shape=ecu_feat_shape,
            concat_merge=False,
            sample_lb=int((ecu_feat_shape - ecu_sample_size) // 2),
            sample_size=ecu_sample_size
        )

    def register_pc_encoder(self):
        pc_channel = [16, 32, 64, 64, 64, 64, 128, 128]
        pc_stride = [1, 2, 1, 2, 1, 2, 2, 2]
        pc_layers = len(pc_channel)
        first_layer = nn.Sequential(
            nn.Conv1d(1, pc_channel[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(pc_channel[0]),
            nn.ReLU6(inplace=True)
        )
        setattr(self, f"pc_conv0", first_layer)
        for layer_idx in range(pc_layers - 1):
            setattr(self, f"pc_conv{layer_idx+1}",
                    depth_conv_1d(pc_channel[layer_idx], pc_channel[layer_idx + 1], 3, pc_stride[layer_idx]))
        for layer_idx in range(8, 11):
            setattr(self, f"pc_conv{layer_idx}", Identity())

    def forward(self, input_image, pc=None):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        img_feat = self.encoder.layer4(self.features[-1])
        if self.with_pc:
            pc_feat = pc
            for layer_idx in range(10):
                pc_layer = getattr(self, f'pc_conv{layer_idx}')
                pc_feat = pc_layer(pc_feat)
        else:
            n, _, _, w = img_feat.shape
            pc_feat = th.zeros((n, self.ecu_pc_channel, w), device=img_feat.device)
        enc_feat = self.ecu(img_feat, pc_feat)
        self.features.append(enc_feat)

        return self.features


class ECUSmall(nn.Module):
    def __init__(self, img_in_channel, pc_in_channel, ecu_channel,
                 sample_size, sample_lb, out_shape, concat_merge):
        super(ECUSmall, self).__init__()
        self.sample_size = sample_size
        self.sample_lb = sample_lb
        self.concat_merge = concat_merge
        self.out_shape = out_shape
        self.img_sample_layer = img2pc_bridge(img_in_channel, ecu_channel, sample_size)
        self.diff_extract_layer = nn.Sequential(
            nn.Conv1d(ecu_channel + pc_in_channel, ecu_channel, 1, 1),
            nn.BatchNorm1d(ecu_channel),
            nn.ReLU6(inplace=True),
            depth_conv_1d(ecu_channel, img_in_channel, 3, 1)
        )
        self.reduce_layer = nn.Sequential(
            depthwise(ecu_channel, 3),
            pointwise(ecu_channel, img_in_channel)
        )

    def forward(self, img_feat, pc_feat):
        crop_feat = th.narrow(img_feat, dim=2, start=self.sample_lb, length=self.sample_size)
        sample_feat = self.img_sample_layer(crop_feat)
        cat_feat = th.cat((sample_feat, pc_feat), dim=1)
        diff = self.diff_extract_layer(cat_feat)
        diff_up = th.stack([diff] * self.out_shape, dim=2)
        fixed_feat = th.cat((diff_up, img_feat), dim=1) if self.concat_merge else diff_up + img_feat
        return fixed_feat
