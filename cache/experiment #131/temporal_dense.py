import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *
from models.backbone.densenet import *


class gap_block(nn.Module):

    def __init__(self, inplanes, mid_planes, seq_len):
        super(gap_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.conv2 = nn.Conv2d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = nn.Conv2d(in_channels=mid_planes, out_channels=inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)

        self.para_fc0_0 = nn.Linear(in_features=inplanes, out_features=int(inplanes / 16))
        self.para_fc0_1 = nn.Linear(in_features=int(inplanes / 16), out_features=inplanes)

        self.para_fc1_0 = nn.Linear(in_features=2 * inplanes, out_features=int(2 * inplanes / 16))
        self.para_fc1_1 = nn.Linear(in_features=int(2 * inplanes / 16), out_features=inplanes)

        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        if seq_len != 1:
            self.cat_vect = nn.Conv1d(in_channels = int(seq_len), out_channels = 1, kernel_size =1 )

        self.para_fc0_0.apply(weights_init_kaiming)
        self.para_fc0_1.apply(weights_init_kaiming)

        self.para_fc1_0.apply(weights_init_kaiming)
        self.para_fc1_1.apply(weights_init_kaiming)

        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        self.bn3.apply(weights_init_kaiming)

        self.conv1.apply(weights_init_kaiming)
        self.conv2.apply(weights_init_kaiming)
        self.conv3.apply(weights_init_kaiming)

    def forward(self, feat_map):
        b, t, c, w, h = feat_map.size()
        feat_para = self.avg(feat_map.view(b * t, c, w, h)).view(b, t, -1)
        gap_feat_map0 = []

        for idx in range(0, t, 2):
            para_00 = self.sigmoid(self.para_fc0_0(feat_para[:, idx, :]))
            para_01 = self.sigmoid(self.para_fc0_0(feat_para[:, idx + 1, :]))
            para_00 = self.relu(self.para_fc0_1(para_00)).view(b, -1, 1, 1)
            para_01 = self.relu(self.para_fc0_1(para_01)).view(b, -1, 1, 1)
            gap_map0 = para_01 * feat_map[:, idx, :, :, :] + para_00 * feat_map[:, idx + 1, :, :, :]
            gap_map0 = gap_map0 ** 2
            gap_feat_map0.append(gap_map0)

        gap_feat_map0 = torch.stack(gap_feat_map0, 1)
        seq_len = gap_feat_map0.size(1)
        gap_feat_vect0 = self.avg(gap_feat_map0.view(b * seq_len, -1, w, h)).view(b, seq_len, -1)
        feat_vect = feat_para.view(b, t, -1)

        gap_feat_map = []
        for i in range(seq_len):
            idx = 2 * i
            para_0 = self.sigmoid(self.para_fc1_0(torch.cat([gap_feat_vect0[:, i, :], feat_vect[:, idx + 1, :]], 1)))
            para_1 = self.sigmoid(self.para_fc1_0(torch.cat([gap_feat_vect0[:, i, :], feat_vect[:, idx, :]], 1)))
            para_0 = self.relu(self.para_fc1_1(para_0)).view(b, -1, 1, 1)
            para_1 = self.relu(self.para_fc1_1(para_1)).view(b, -1, 1, 1)
            gap_map = gap_feat_map0[:, i, :, :, :] + para_0 * feat_map[:, idx, :, :, :] + para_1 * feat_map[:, idx + 1, :, :, :]
            gap_feat_map.append(gap_map)

        gap_feat_map = torch.stack(gap_feat_map, 1)
        gap_feat_map = gap_feat_map.view(b * seq_len, -1, w, h)

        gap_feat_map = self.conv1(gap_feat_map)
        gap_feat_map = self.bn1(gap_feat_map)
        gap_feat_map = self.relu(gap_feat_map)

        gap_feat_map = self.conv2(gap_feat_map)
        gap_feat_map = self.bn2(gap_feat_map)
        gap_feat_map = self.relu(gap_feat_map)

        gap_feat_map = self.conv3(gap_feat_map)
        gap_feat_map = self.bn3(gap_feat_map)
        gap_feat_map = self.relu(gap_feat_map)

        gap_feat_vect = self.avg(gap_feat_map).view(b, seq_len, -1)
        if seq_len != 1:
            feature = self.cat_vect(gap_feat_vect)
            # feature = self.relu(feature)
        else:
            feature = gap_feat_vect

        feature = feature.view(b, -1)
        gap_feat_map = gap_feat_map.view(b, seq_len, -1, w, h)

        return gap_feat_map, feature


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

        self.conv1 = nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances, kernel_size=1)
        self.cat_vect = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=1)
        self.cat_conv = nn.ModuleList([nn.Conv1d(in_channels=i + 2, out_channels=1, kernel_size=1) for i in range(3)])

        t = seq_len
        self.layer1 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t / 2)
        t = t / 2
        self.layer2 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t / 2)
        t = t / 2
        self.layer3 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t / 2)

        self.bottleneck = nn.BatchNorm1d(self.plances)
        self.classifier = nn.ModuleList([nn.Linear(self.plances, self.num_classes, bias=False) for i in range(4)])
        self.bottleneck.bias.requires_grad_(False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)
        self.cat_conv.apply(weights_init_kaiming)
        self.cat_vect.apply(weights_init_kaiming)
        self.conv1.apply(weights_init_kaiming)

    def forward(self, x, pids=None, camids=None):

        b, t, c, w, h = x.size()
        x = x.view(b * t, c, w, h)
        feat_map = self.base(x)  # (b * t, c, 16, 8)
        w = feat_map.size(2)
        h = feat_map.size(3)
        feat_map = self.conv1(feat_map)
        feat_map = self.relu(feat_map)

        feat_map_0 = feat_map.view(b, t, -1, w, h)  # (32, seq_len, 1048, 16, 8)
        feat_map_1, feature_1 = self.layer1(feat_map_0)
        feat_map_2, feature_2 = self.layer2(feat_map_1)
        feat_map_3, feature_3 = self.layer3(feat_map_2)

        cat_feat_2 = torch.stack([feature_1, feature_2], 1)
        cat_feat_3 = torch.stack([feature_1, feature_2, feature_3], 1)

        feature_ord1 = feature_1
        feature_ord2 = self.cat_conv[0](cat_feat_2).view(b, -1)
        feature_ord3 = self.cat_conv[1](cat_feat_3).view(b, -1)

        feature_list = []
        feature_list.append(feature_ord2)
        feature_list.append(feature_ord3)

        BN_feature_ord1 = self.bottleneck(feature_ord1)
        BN_feature_ord2 = self.bottleneck(feature_ord2)
        BN_feature_ord3 = self.bottleneck(feature_ord3)

        cls_score_list = []
        torch.cuda.empty_cache()

        if self.training:
            cls_score_list.append(self.classifier[1](BN_feature_ord1))
            cls_score_list.append(self.classifier[2](BN_feature_ord2))
            cls_score_list.append(self.classifier[3](BN_feature_ord3))
            return cls_score_list, feature_list
        else:
            return feature_ord3, BN_feature_ord3, pids, camids

