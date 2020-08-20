"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable
from resnet import _ConvBatchNormReLU, _ResBlock
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock, _ASPPModule, _ConvReLU_

class MSCC(nn.Module):
    """Multi-scale inputs"""

    def __init__(self, scale, pyramids=[0.5, 0.75]):
        super(MSCC, self).__init__()
        self.scale = scale
        self.pyramids = pyramids

    def forward(self, x, mask):
        KLD, h0, h1 = self.scale(x, mask)
        logits_h0 = self.get_resized_logits(x, mask, h0, 1)
 
        return KLD, logits_h0, h0, h1

    def get_resized_logits(self, x, mask, logits, scale_return_index):
        # Original
        interp = lambda l: F.interpolate(l, size=logits.shape[2:], mode="bilinear", align_corners=False)

        # Scaled
        logits_pyramid = []
        for p in self.pyramids:
            size = [int(s * p) for s in x.shape[2:]]
            h = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.scale(h, mask)[scale_return_index])

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max

    def freeze_bn(self):
        self.scale.freeze_bn()


class DeepLabV2_local(nn.Sequential):
    """DeepLab v2"""

    def __init__(self, n_classes, n_blocks, pyramids, freeze_bn):
        super(DeepLabV2_local, self).__init__()

        self.add_module(
            "layer1",
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                        ("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                    ]
                )
            )
        )
        self.add_module("layer2", _ResBlock(n_blocks[0], 64, 64, 256, 1, 1))
        self.add_module("layer3", _ResBlock(n_blocks[1], 256, 128, 512, 2, 1))
        self.add_module("layer4", _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2))
        self.add_module("layer5", _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4))
        self.add_module("aspp", _ASPPModule(2048, n_classes, pyramids))

        self.add_module("contextual1", _ConvReLU_(n_classes, 256, 3, 1, 1, 1))
        self.add_module("contextual2", _ConvReLU_(256, 256, 3, 1, 2, 2))
        self.add_module("contextual3", _ConvReLU_(256, 256, 3, 1, 5, 5))
        self.add_module("fc_1", _ConvReLU_(3*256, 3, 3, 1, 1, 1))
        self.add_module("contextualpool", _ConvReLU_(3*256, n_classes, 1, 1, 0, 1))
        self.add_module("contextuallocalmu", _ConvReLU_(n_classes, n_classes, 3, 1, 1, 1, relu=False))
        self.add_module("contextuallocalsigma",_ConvReLU_(n_classes, n_classes, 3, 1, 1, 1))

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x, mask):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.aspp(h)

        # Contextual Module
        h1 = self.contextual1(torch.sigmoid(h))
        h2 = self.contextual2(h1)
        h3 = self.contextual3(h2)     
        f1 = torch.sigmoid(self.fc_1(torch.cat([h1, h2, h3], dim=1)))
        h1 = h1 * f1[:,0,:,:].unsqueeze(1)
        h2 = h2 * f1[:,1,:,:].unsqueeze(1)
        h3 = h3 * f1[:,2,:,:].unsqueeze(1)
        h1 = self.contextualpool(torch.cat([h1, h2, h3], dim=1))
        localmu = F.interpolate(self.contextuallocalmu(h1), size=h.size()[2:], mode="bilinear")
        localsigma = F.interpolate(self.contextuallocalsigma(h1), size=h.size()[2:], mode="bilinear")
        h1 = F.interpolate(self.reparameterize(localmu, localsigma), size=h.size()[2:], mode="bilinear") # contextual latent code      
        att = torch.sigmoid(h1)
        h_att = torch.mul(h, att)
        h0 = torch.sigmoid(h + h_att) # augmented feature

        # KL-Div loss
        localmask = F.interpolate(mask.float().unsqueeze(1).repeat(1, localmu.size(1), 1, 1), size=h.size()[2:], mode="nearest").bool()
        KLD = -0.5 * torch.sum((1 + localsigma - localmu.pow(2) - localsigma.exp())[localmask.bool()])/localmask.float().sum() if localmask.float().sum()>0 else localmask.float().mean()

        return KLD, h0, h1 # KLD loss, feature, contextual latent code

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def reparameterize(self, mu, logvar):
        """
        THE REPARAMETERIZATION IDEA:
        """
        if self.training:
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu


def DeepLabV2_ResNet101_local(hp):
    return DeepLabV2_local(
        n_classes=hp['n_classes'], 
        n_blocks=[3, 4, 23, 3], 
        pyramids=[6, 12, 18, 24], 
        freeze_bn=True
    )

def DeepLabV2_ResNet101_local_MSC(hp):
    return MSCC(
        scale=DeepLabV2_ResNet101_local(hp), 
        pyramids=[0.5, 0.75]
    )


class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.fc = Conv2dBlock(in_dim=hp['in_dim_fc'], 
                              out_dim=hp['out_dim_fc'], 
                              ks=1, 
                              st=1, 
                              padding=0, 
                              norm=hp['norm_fc'], 
                              activation=hp['activ_fc'], 
                              dropout=hp['drop_fc'])

        self.pred = nn.Conv2d(hp['out_dim_fc'], 1, 1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.cls = nn.Conv2d(hp['out_dim_fc'], hp['out_dim_cls'], 1, stride=1, padding=0)

    def forward(self, feat, mode):
        pred_map = self.fc(feat)
        if mode == 'gan':
            gan_score = self.pred(pred_map)
            gan_score_sigmoid = self.sigmoid(gan_score)
            return gan_score_sigmoid
        elif mode == 'cls':
            cls_score = self.cls(pred_map)
            return cls_score
        else:
            raise NotImplementedError('Invalid mode {} for discriminator.' % mode)


class Generator(nn.Module):
    def __init__(self, hp):
        super(Generator, self).__init__()

        self.mlp = nn.Sequential(
            Conv2dBlock(
                in_dim=hp['in_dim_mlp'], 
                out_dim=1024, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.1
            ),
            Conv2dBlock(
                in_dim=1024, 
                out_dim=960, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.1
            ),
            Conv2dBlock(
                in_dim=960, 
                out_dim=864, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.3
            ),
            Conv2dBlock(
                in_dim=864, 
                out_dim=784, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.5
            ),
            Conv2dBlock(
                in_dim=784, 
                out_dim=720, 
                ks=1, 
                st=1, 
                padding=0, 
                norm='none', 
                activation='lrelu', 
                dropout=0.5
            ),
            nn.Conv2d(720, hp['out_dim_mlp'], 1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, sample):
        feat = self.mlp(sample)
        feat_sigmoid = self.sigmoid(feat)
        return feat_sigmoid
