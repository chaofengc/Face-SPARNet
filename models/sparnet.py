from models.blocks import *
import torch
from torch import nn
import numpy as np


class SPARNet(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body 
        - up_res_depth: depth of residual layers in each upsample block

    """
    def __init__(
        self,
        min_ch=32,
        max_ch=128,
        in_size=128,
        out_size=128,
        min_feat_size=16,
        res_depth=10,
        relu_type='leakyrelu',
        norm_type='bn',
        att_name='spar',
        bottleneck_size=4,
    ):
        super(SPARNet, self).__init__()
        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}

        ch_clip = lambda x: max(min_ch, min(x, max_ch))

        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))
        n_ch = ch_clip(max_ch // int(np.log2(in_size // min_feat_size) + 1))

        # ------------ define encoder --------------------
        self.encoder = []
        self.encoder.append(ConvLayer(3, n_ch, 3, 1))
        hg_depth = int(np.log2(64 / bottleneck_size))
        for i in range(down_steps):
            cin, cout = ch_clip(n_ch), ch_clip(n_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', hg_depth=hg_depth, att_name=att_name, **nrargs))

            n_ch = n_ch * 2
            hg_depth = hg_depth - 1
        hg_depth = hg_depth + 1
        self.encoder = nn.Sequential(*self.encoder)

        # ------------ define residual layers --------------------
        self.res_layers = []
        for i in range(res_depth + 3 - down_steps):
            channels = ch_clip(n_ch)
            self.res_layers.append(ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs))
        self.res_layers = nn.Sequential(*self.res_layers)

        # ------------ define decoder --------------------
        self.decoder = []
        for i in range(up_steps):
            hg_depth = hg_depth + 1
            cin, cout = ch_clip(n_ch), ch_clip(n_ch // 2)
            self.decoder.append(ResidualBlock(cin, cout, scale='up', hg_depth=hg_depth, att_name=att_name, **nrargs))
            n_ch = n_ch // 2

        self.decoder = nn.Sequential(*self.decoder)
        self.out_conv = ConvLayer(ch_clip(n_ch), 3, 3, 1)
    
    def forward(self, input_img):
        out = self.encoder(input_img)
        out = self.res_layers(out)
        out = self.decoder(out)
        out_img = self.out_conv(out)
        return out_img

