import torch
import torch.nn as nn
from torch.nn import functional as F

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

class AATM(nn.Module):

    def __init__(self, inplanes, mid_planes, spatial_method,
                 is_mutual_channel_attention,is_mutual_spatial_attention,
                 is_appearance_channel_attention,is_appearance_spatial_attention, **kwargs):

        super(AATM, self).__init__()

        self.inplanes = inplanes
        self.mid_planes = 256
        # self.seq_len = seq_len
        self.spatial_method = spatial_method
        self.is_mutual_channel_attention = is_mutual_channel_attention
        self.is_mutual_spatial_attention = is_mutual_spatial_attention
        self.is_appearance_channel_attention = is_appearance_channel_attention
        self.is_appearance_spatial_attention = is_appearance_spatial_attention

        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.Embeding = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            # nn.Conv2d(in_channels=mid_planes, out_channels=128,
            #           kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(128),
            # self.relu
        )
        self.Embeding.apply(weights_init_kaiming)

        if self.is_mutual_spatial_attention == 'yes':
            print('Build mutual sptial attention!')

            self.gamma_temporal = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes/8),
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(int(inplanes/8)),
                self.relu
            )
            self.gamma_temporal.apply(weights_init_kaiming)

            self.beta_temporal = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes/8),
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(int(inplanes/8)),
                self.relu
            )
            self.beta_temporal.apply(weights_init_kaiming)

            self.gg_temporal = nn.Sequential(
                nn.Conv2d(in_channels=2 * 16 * 8, out_channels=128,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                self.relu,
            )
            self.gg_temporal.apply(weights_init_kaiming)

            self.tte_para = nn.Sequential(
                nn.Conv2d(in_channels=2 * 128, out_channels=128,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                self.relu,
            )
            self.tte_para.apply(weights_init_kaiming)

            self.te_para = nn.Sequential(
                nn.Conv2d(in_channels=2 * 128, out_channels = 1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                self.sigmoid
            )
            self.te_para.apply(weights_init_kaiming)

        if self.is_mutual_channel_attention == 'yes':
            print('Build mutual channel attention!')

            self.theta_channel = nn.Sequential(
                nn.Conv1d(in_channels=inplanes, out_channels=int(inplanes / 8),
                          kernel_size=1, stride=1, padding=0, bias=False),
                self.relu,
            )
            self.theta_channel.apply(weights_init_kaiming)

            self.channel_para = nn.Sequential(
                nn.Linear(in_features=int(inplanes / 4), out_features=int(inplanes / 8)),
                self.relu,
                nn.Linear(in_features=int(inplanes / 8), out_features=inplanes),
                self.sigmoid
            )
            self.channel_para.apply(weights_init_kaiming)

        if self.is_appearance_spatial_attention :
            print('Build appearance spatial attention!')

            self.alphi_appearance = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8),
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(int(inplanes / 8)),
                self.relu
            )
            self.alphi_appearance.apply(weights_init_kaiming)

            self.delta_appearance = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8),
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(int(inplanes / 8)),
                self.relu
            )
            self.delta_appearance.apply(weights_init_kaiming)

            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=2 * 16 * 8, out_channels=128,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                self.relu,
            )
            self.gg_spatial.apply(weights_init_kaiming)

            self.spa_para = nn.Sequential(
                nn.Conv2d(in_channels=2 * 128, out_channels=128,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                self.relu,
                nn.Conv2d(in_channels=128, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                self.sigmoid
            )
            self.spa_para.apply(weights_init_kaiming)

        if self.is_appearance_channel_attention == 'yes':
            print('Build appearacne channel attention!')
            self.app_channel = nn.Sequential(
                nn.Linear(in_features=inplanes, out_features=int(inplanes / 8)),
                self.relu,
                nn.Linear(in_features=int(inplanes / 8), out_features=inplanes),
                self.sigmoid
            )
            self.app_channel.apply(weights_init_kaiming)

        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(mid_planes),
        #     self.relu,
        #     nn.Conv2d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(mid_planes),
        #     self.relu,
        #     nn.Conv2d(in_channels=mid_planes, out_channels=inplanes, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(inplanes),
        #     self.relu
        # )
        # self.conv_block.apply(weights_init_kaiming)

    def forward(self, feat_map):

        b, t, c, h, w = feat_map.size()
        reshape_map = feat_map.view(b * t, c, h, w)
        feat_para = self.avg(reshape_map).view(b, t, -1)
        embed_feat = self.Embeding(reshape_map).view(b, t, -1, h, w)

        if self.is_mutual_spatial_attention == 'yes' :
            gamma_feat = self.gamma_temporal(reshape_map).view(b, t, -1, h * w)
            beta_feat = self.beta_temporal(reshape_map).view(b, t, -1, h * w)

        if self.is_mutual_channel_attention == 'yes':
            channel_para = self.theta_channel(feat_para.permute(0, 2, 1))

        gap_feat_map0 = []

        for idx in range(0, t, 2):

             if self.is_mutual_channel_attention == 'yes':
                 para0 = torch.cat((channel_para[:, :, idx], channel_para[:, :, idx + 1]), 1)
                 para_00 = self.channel_para(para0).view(b, -1, 1, 1)
                 para1 = torch.cat((channel_para[:, :, idx + 1], channel_para[:, :, idx]), 1)
                 para_01 = self.channel_para(para1).view(b, -1, 1, 1)

             if self.is_mutual_spatial_attention == 'yes':
                 embed_feat0 = embed_feat[:, idx, :, :, :]
                 embed_feat1 = embed_feat[:, idx + 1, :, :, :]

                 gamma_feat0 = gamma_feat[:, idx, :, :].permute(0, 2, 1)
                 beta_feat0 = beta_feat[:, idx + 1, :, :]
                 Gs0 = torch.matmul(gamma_feat0, beta_feat0)
                 Gs_in0 = Gs0.permute(0, 2, 1).view(b, h * w, h, w)
                 Gs_out0 = Gs0.view(b, h * w, h, w)
                 Gs_joint0 = torch.cat((Gs_in0, Gs_out1), 1)
                 Gs_joint0 = self.gg_temporal(Gs_joint0)
                 para_alpha = self.tte_para(torch.cat((embed_feat0, embed_feat1), 1))
                 para_alpha = self.te_para(torch.cat((para_alpha, Gs_joint0), 1))

                 gamma_feat1 = gamma_feat[:, idx + 1, :, :].permute(0, 2, 1)
                 beta_feat1 = beta_feat[:, idx, :, :]
                 Gs1 = torch.matmul(gamma_feat1, beta_feat1)
                 Gs_in1 = Gs1.permute(0, 2, 1).view(b, h * w, h, w)
                 Gs_out1 = Gs1.view(b, h * w, h, w)
                 Gs_joint1 = torch.cat((Gs_in1, Gs_out0), 1)
                 Gs_joint1 = self.gg_temporal(Gs_joint1)
                 para_beta = self.tte_para(torch.cat((embed_feat1, embed_feat0), 1))
                 para_beta = self.te_para(torch.cat((para_beta, Gs_joint1), 1))

             if self.is_mutual_spatial_attention == 'yes' and self.is_mutual_channel_attention == 'yes':
                 para_00 = para_00 * para_alpha
                 para_01 = para_01 * para_beta

             elif self.is_mutual_channel_attention == 'yes':
                 para_00 = para_00
                 para_01 = para_01

             elif self.is_mutual_spatial_attention == 'yes':
                 para_00 = para_alpha
                 para_01 = para_beta

             else:
                 para_00 = 1
                 para_01 = 1

             gap_map0 = para_00 * feat_map[:, idx, :, :, :] + para_01 * feat_map[:, idx+1, :, :, :]
             # gap_map0 = self.relu(gap_map0)
             gap_map0 = gap_map0 ** 2
             gap_feat_map0.append(gap_map0)

        gap_feat_map0 = torch.stack(gap_feat_map0, 1)
        seq_len = gap_feat_map0.size(1)
        feat_vect = feat_para.view(b, t, -1)

        if self.is_appearance_spatial_attention == 'yes':
            embed_feat = embed_feat.view(b * t, -1, h, w)
            alphi_feat = self.alphi_appearance(reshape_map).view(b * t, -1, h * w)
            delta_feat = self.delta_appearance(reshape_map).view(b * t, -1, h * w)
            alphi_feat = alphi_feat.permute(0, 2, 1)
            Gs = torch.matmul(alphi_feat, delta_feat)
            Gs_in = Gs.permute(0, 2, 1).view(b * t, h * w, h, w)
            Gs_out = Gs.view(b * t, h * w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)
            para_spa = torch.cat((embed_feat, Gs_joint), 1)
            para_spa = self.spa_para(para_spa).view(b, t, -1, h, w)

        gap_feat_map = []
        for i in range(seq_len):
            idx = 2 * i

            if self.is_appearance_channel_attention == 'yes':
                para_0 = self.app_channel(feat_vect[:, idx, :]).view(b, -1, 1, 1)
                para_1 = self.app_channel(feat_vect[:, idx+1, :]).view(b, -1, 1, 1)

            if self.is_appearance_spatial_attention == 'yes' and self.is_appearance_channel_attention == 'yes':
                para_0 = para_0 * para_spa[:, idx, :, :, :]
                para_1 = para_1 * para_spa[:, idx+1, :, :, :]

            elif self.is_appearance_channel_attention == 'yes':
                para_0 = para_0
                para_1 = para_1

            elif self.is_appearance_spatial_attention == 'yes':
                para_0 = para_spa[:, idx, :, :, :]
                para_1 = para_spa[:, idx+1, :, :, :]

            else:
                para_0 = 1
                para_1 = 1

            gap_map = gap_feat_map0[:, i, :, :, :] + self.relu(para_0 * feat_map[:, idx, :, :, :] + para_1 * feat_map[:, idx + 1, :, :, :])
            gap_feat_map.append(gap_map)

        gap_feat_map = torch.stack(gap_feat_map, 1)
        gap_feat_map = gap_feat_map.view(b * seq_len, -1, h, w)
        # gap_feat_map = self.conv_block(gap_feat_map)

        if self.spatial_method == 'max':
            gap_feat_vect = F.max_pool2d(gap_feat_map, gap_feat_map.size()[2:])

        elif self.spatial_method == 'avg':
            gap_feat_vect = F.avg_pool2d(gap_feat_map, gap_feat_map.size()[2:])

        gap_fea_vect = gap_feat_vect.view(b, seq_len, -1)
        feature = gap_fea_vect.mean(1)
        feature = feature.view(b, -1)
        gap_feat_map = gap_feat_map.view(b, seq_len, -1, h, w)
        torch.cuda.empty_cache()

        return gap_feat_map, feature





