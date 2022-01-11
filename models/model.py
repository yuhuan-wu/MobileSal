import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import BatchNorm2d
import math

from models.MobileNetV2 import mobilenet_v2
from torch.nn import Parameter

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, leaky_relu=False, use_bn=True, frozen=False, spectral_norm=False, prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            elif spectral_norm:
                self.bn = SpectralNorm(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            if leaky_relu is True:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu is True:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                               dilation=1, groups=groups, bias=bias,
                               use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x


class ReceptiveConv(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=24, scale=4, dilation=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width*scale)
        #self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        #if self.scale > 1:
        #    out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileSal(nn.Module):
    def __init__(self, pretrained=True, use_carafe=True,
                 enc_channels=[16, 24, 32, 96, 320],
                 dec_channels=[16, 24, 32, 96, 320]):
        super(MobileSal, self).__init__()
        self.backbone = mobilenet_v2(pretrained)
        self.depthnet = DepthNet()

        self.depth_fuse = DepthFuseNet(inchannels=320)

        self.idr = IDR(enc_channels)

        self.fpn = CPRDecoder(enc_channels, dec_channels)

        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)

    def loss(self, input, target):
        pass

    def forward(self, input, depth=None, test=True):

        # generate backbone features
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)
        
        # RGB-D fuse & implicit depth restoration
        if depth is not None:
            depth_features = self.depthnet(depth)
            conv5 = self.depth_fuse(conv5, depth_features[-1])
            if test:
                depth_pred = None
            else:
                depth_pred = self.idr([conv1, conv2, conv3, conv4, conv5], input=input) # implicit depth restoration
        else:
            depth_pred = None

        features = self.fpn([conv1, conv2, conv3, conv4, conv5])

        saliency_maps = []
        for idx, feature in enumerate(features[:5]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )
            if test:
                break
        saliency_maps = torch.sigmoid(torch.cat(saliency_maps, dim=1))

        if test:
            return saliency_maps
        else:   
            return saliency_maps, depth_pred


class DepthNet(nn.Module):
    def __init__(self, pretrained=None,
                 use_gan=False):
        super(DepthNet, self).__init__()
        block = InvertedResidual
        input_channel = 1
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 2, 2, 1],
            [4, 32, 2, 2, 1],
            [4, 64, 2, 2, 1],
            [4, 96, 2, 2, 1],
            [4, 320, 2, 2, 1],
        ]
        features = []
        # building inverted residual blocks
        for t, c, n, s, d in inverted_residual_setting:
            output_channel = int(c * 1.0)
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in [1, 3, 5, 7, 9]:
                feats.append(x)
        return feats


class DepthFuseNet(nn.Module):
    def __init__(self, inchannels=320):
        super(DepthFuseNet, self).__init__()
        self.d_conv1 = InvertedResidual(inchannels, inchannels, residual=True)
        self.d_linear = nn.Sequential(
            nn.Linear(inchannels, inchannels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(inchannels, inchannels, bias=True),
        )
        self.d_conv2 = InvertedResidual(inchannels, inchannels, residual=True)

    def forward(self, x, x_d):
        x_f = self.d_conv1(x * x_d)
        x_d1 = self.d_linear(x.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=3)
        x_f1 = self.d_conv2(torch.sigmoid(x_d1) * x_f * x_d)
        return x_f1

class IDR(nn.Module):
    def __init__(self, enc_channels, channels=256, size_idx=3):
        super(IDR, self).__init__()
        self.inners = nn.ModuleList()
        for i in range(len(enc_channels)):
            self.inners.append(
                ConvBNReLU(enc_channels[i], channels, ksize=1, pad=0)
            )
        self.reduce = ConvBNReLU(channels * 5, channels, ksize=1)
        self.fuse = nn.Sequential(
                InvertedResidual(channels, channels, expand_ratio=6, residual=True),
                InvertedResidual(channels, channels, expand_ratio=6, residual=True),
                InvertedResidual(channels, channels, expand_ratio=6, residual=True),
                InvertedResidual(channels, channels, expand_ratio=6, residual=True),
                nn.Conv2d(channels, 1, 1, stride=1, padding=0)
            )
        self.size_idx = size_idx

    def forward(self, x, input=None):
        xx = []
        size = x[self.size_idx].shape[2:]
        for each_x in x:
            xx.append(
                F.interpolate(each_x, size=size, mode="bilinear")
            )
        xxx = []
        for i, each_xx in enumerate(xx):
            xxx.append(self.inners[i](each_xx))
        xxx = self.fuse(self.reduce(torch.cat(xxx, dim=1)))
        return torch.sigmoid(F.interpolate(xxx, size=input.shape[2:], mode='bilinear'))

class CPR(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, dilation=[1,2,3], residual=True):
        super(CPR, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        self.conv1 = ConvBNReLU(inp, hidden_dim, ksize=1, pad=0, prelu=False)

        self.hidden_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[0], groups=hidden_dim, dilation=dilation[0])
        self.hidden_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[1], groups=hidden_dim, dilation=dilation[1])
        self.hidden_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[2], groups=hidden_dim, dilation=dilation[2])
        self.hidden_bnact = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        m = self.conv1(x)
        m = self.hidden_conv1(m) + self.hidden_conv2(m) + self.hidden_conv3(m)
        m = self.hidden_bnact(m)
        if self.use_res_connect:
            return x + self.out_conv(m)
        else:
            return self.out_conv(m)



class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, input_num=2):
        super(Fusion, self).__init__()
        if input_num == 2:
            self.channel_att = nn.Sequential(nn.Linear(in_channels, in_channels),
                                             nn.ReLU(),
                                             nn.Linear(in_channels, in_channels),
                                             nn.Sigmoid()
                                             )
        self.fuse = nn.Sequential( CPR(in_channels, in_channels, expand_ratio=expansion, residual=True),
                                      ConvBNReLU(in_channels, in_channels, ksize=1, pad=0, stride=1)
                                      )


    def forward(self, low, high=None):
        if high is None:
            final = self.fuse(low)
        else:
            high_up = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
            fuse = torch.cat((high_up, low), dim=1)

            final = self.channel_att(fuse.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=2) * self.fuse(fuse)

        return final

class CPRDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, teacher=False):
        super(CPRDecoder, self).__init__()
        #assert in_channels[-1] == out_channels[-1]
        self.inners_a = nn.ModuleList()
        self.inners_b = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))

        self.fuse = nn.ModuleList()
        for i in range(len(in_channels)):
            if i == len(in_channels) - 1:
                self.fuse.append(Fusion(out_channels[i], out_channels[i], input_num=1))
            else:
                self.fuse.append(
                    ConvBNReLU(out_channels[i], out_channels[i]) if teacher else Fusion(out_channels[i], out_channels[i])
                    )

    def forward(self, features, att=None):
        stage_result = self.fuse[-1](self.inners_a[-1](features[-1]))
        results = [stage_result]
        for idx in range(len(features) - 2, -1, -1):
            #inner_top_down = F.interpolate(self.inners_b[idx](stage_result),
            #                               size=features[idx].shape[2:],
            #                               mode='bilinear',
            #                               align_corners=False)
            inner_top_down = self.inners_b[idx](stage_result)
            inner_lateral = self.inners_a[idx](features[idx])
            stage_result = self.fuse[idx](inner_lateral, inner_top_down)#(torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)

        return results
    
class FPNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNDecoder, self).__init__()
        #assert in_channels[-1] == out_channels[-1]
        self.inners = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners.append(ConvBNReLU(in_channels[i], out_channels[i], ksize=1, pad=0))

        self.fuse = nn.ModuleList()
        for i in range(len(out_channels)):
            self.fuse.append(
                ConvBNReLU(out_channels[i], out_channels[i]),
            )

    def forward(self, features, att=None):
        stage_result = self.fuse[-1](self.inners[-1](features[-1]))
        results = [stage_result]
        for idx in range(len(features) - 2, -1, -1):
            inner_top_down = F.interpolate(self.inners[idx](stage_result),
                                           size=features[idx].shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            inner_lateral = self.inners[idx-1](features[idx])
            stage_result = self.fuse[idx](inner_top_down + inner_lateral)
            results.insert(0, stage_result)

        return results
