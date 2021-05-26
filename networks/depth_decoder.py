# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch as th
import torch.nn as nn

from collections import OrderedDict
from layers import *
from components import Identity, img2pc_bridge, depth_conv_1d, depthwise, pointwise, weights_init
from torch.nn.functional import interpolate


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


class MobileNetDecoder(nn.Module):
    def __init__(self, scales=range(4), num_output_channel=1):
        super(MobileNetDecoder, self).__init__()
        self.img_fw_layer_dst = [3, 2, 1]
        self.scales = scales
        self.register_decoder(out_channel=num_output_channel)
        self.outputs = {}
        self.sigmoid = nn.Sigmoid()

    def register_decoder(self, kernel_size=5, out_channel=1):
        decoder_in_channel = [512, 512, 512, 256, 128]
        decoder_out_channel = [512, 256, 128, 64, 32]
        for layer_idx in range(5):
            layer = nn.Sequential(
                depthwise(decoder_in_channel[layer_idx], kernel_size if layer_idx != 0 else 3),
                pointwise(decoder_in_channel[layer_idx], decoder_out_channel[layer_idx])
            )
            weights_init(layer)
            setattr(self, f"decode_conv{layer_idx}", layer)
        for inv_out_layer_idx in self.scales:
            out_layer_idx = 4 - inv_out_layer_idx
            layer = Conv3x3(decoder_out_channel[out_layer_idx], out_channel)
            # layer = pointwise(decoder_out_channel[out_layer_idx], out_channel)
            weights_init(layer)
            setattr(self, f"out_conv{out_layer_idx}", layer)

    def forward(self, features):
        self.outputs = {}
        dec_feat = features[-1]
        fw_idx = -2
        for layer_idx in range(5):
            dec_layer = getattr(self, f'decode_conv{layer_idx}')
            # print(layer_idx, dec_feat.shape, dec_layer)
            dec_feat = dec_layer(dec_feat)
            dec_feat = interpolate(dec_feat, scale_factor=2, mode='nearest')
            inv_layer_idx = 4 - layer_idx
            if inv_layer_idx in self.scales:
                out_layer = getattr(self, f'out_conv{layer_idx}')
                self.outputs[("disp", 4-layer_idx)] = self.sigmoid(out_layer(dec_feat))
            if layer_idx in self.img_fw_layer_dst:
                fw_feat = features[fw_idx]
                fw_idx -= 1
                dec_feat = th.cat((dec_feat, fw_feat), dim=1)

        return self.outputs
