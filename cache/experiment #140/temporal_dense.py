import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *
from models.backbone.densenet import *
from models.GapBlock import gap_block

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def init_pretrained_weight(model, model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url, model_dir='./')
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class tem_dense(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len):
        super(tem_dense, self).__init__()
        self.in_planes = 2048
        self.base = ResNet()

        if pretrain_choice == 'imagenet':
            init_pretrained_weight(self.base, model_urls[model_name])
            print('Loading pretrained ImageNet model......')

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.plances = 1024

        self.avg_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.plances),
            self.relu
        )

        self.cat_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)

        t = seq_len
        self.layer1 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t / 2)
        t = t / 2
        self.layer2 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t / 2)
        t = t / 2
        self.layer3 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t / 2)

        self.bottleneck = nn.BatchNorm1d(self.plances)
        self.classifier = nn.Linear(self.plances, self.num_classes, bias=False)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)

    def forward(self, x, pids=None, camids=None):

        b, t, c, w, h = x.size()
        x = x.view(b * t, c, w, h)
        feat_map = self.base(x)  # (b * t, c, 16, 8)
        w = feat_map.size(2)
        h = feat_map.size(3)
        feat_map = self.down_channel(feat_map)

        feat_map_0 = feat_map.view(b, t, -1, w, h)  # (32, seq_len, 1048, 16, 8)
        feat_map_1, feature_1 = self.layer1(feat_map_0)
        feat_map_2, feature_2 = self.layer2(feat_map_1)
        feat_map_3, feature_3 = self.layer3(feat_map_2)

        cat_feat_3 = torch.stack([feature_1, feature_2, feature_3], 1)
        feature_ord3 = self.cat_conv(cat_feat_3)
        feature_ord3 = self.sigmoid(feature_ord3).view(b, -1)

        feature_list = []
        feature_list.append(feature_ord3)
        BN_feature_ord3 = self.bottleneck(feature_ord3)

        cls_score_list = []
        torch.cuda.empty_cache()

        if self.training:
            cls_score_list.append(self.classifier(BN_feature_ord3))
            return cls_score_list, feature_list
        else:
            return feature_ord3, BN_feature_ord3, pids, camids

