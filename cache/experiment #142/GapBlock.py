import torch
import torch.nn as nn


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


class gap_block(nn.Module):

    def __init__(self, inplanes, mid_planes, seq_len):
        super(gap_block, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.theta_channel = nn.Sequential(
            nn.Conv1d(in_channels=inplanes,out_channels=int(inplanes/8),
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(int(inplanes/8)),
            self.relu,
        )
        self.theta_channel.apply(weights_init_kaiming)

        self.channel_para_0 = nn.Sequential(
            nn.Linear(in_features=int(inplanes / 4), out_features=int(inplanes / 8)),
            self.relu,
            nn.Linear(in_features=int(inplanes / 8), out_features=inplanes),
            self.sigmoid
        )
        self.channel_para_0.apply(weights_init_kaiming)

        self.channel_para_1 = nn.Sequential(
            nn.Linear(in_features=int(inplanes / 4), out_features=int(inplanes / 8)),
            self.relu,
            nn.Linear(in_features=int(inplanes / 8), out_features=inplanes),
            self.sigmoid
        )
        self.channel_para_1.apply(weights_init_kaiming)

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
            nn.Conv2d(in_channels = 3*16*8, out_channels=128,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(in_channels=128, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            self.sigmoid
        )
        self.gg_temporal.apply(weights_init_kaiming)

        self.alphi_appearance = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels= int(inplanes/8),
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes/8)),
            self.relu
        )
        self.alphi_appearance.apply(weights_init_kaiming)

        self.delta_appearance = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes/8),
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes/8)),
            self.relu
        )
        self.delta_appearance.apply(weights_init_kaiming)

        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=3 * 16 * 8, out_channels=128,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(in_channels=128, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            self.sigmoid,
        )
        self.gg_spatial.apply(weights_init_kaiming)

        self.para_fc1_0 = nn.Linear(in_features=inplanes, out_features=int(inplanes / 8))
        self.para_fc1_1 = nn.Linear(in_features=int(inplanes / 8), out_features=inplanes)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
            self.relu
        )
        self.conv_block.apply(weights_init_kaiming)

        if seq_len != 1:
            self.cat_vect = nn.Conv1d(in_channels = int(seq_len), out_channels = 1, kernel_size =1 )

        self.para_fc1_0.apply(weights_init_kaiming)
        self.para_fc1_1.apply(weights_init_kaiming)


    def forward(self, feat_map):

        b, t, c, h, w = feat_map.size()
        reshape_map = feat_map.view(b*t, c, h, w)
        feat_para = self.avg(reshape_map).view(b, t, -1)

        gamma_feat = self.gamma_temporal(reshape_map).view(b, t, -1, w*h)
        beta_feat = self.beta_temporal(reshape_map).view(b, t, -1, w*h)
        channel_para = self.theta_channel(feat_para.permute(0, 2, 1))
        gap_feat_map0 = []

        for idx in range(0, t, 2):

            para = torch.cat((channel_para[:, :, idx], channel_para[:, :, idx+1]),1)
            para_00 = self.channel_para_0(para).view(b, -1, 1, 1)
            para_01 = self.channel_para_1(para).view(b, -1, 1, 1)

            gamma_feat0 = gamma_feat[:, idx, :, :].permute(0, 2, 1)
            beta_feat0 = beta_feat[:, idx+1, :, :]
            Gs0 = torch.matmul(gamma_feat0, beta_feat0)
            gamma_feat0 = gamma_feat0.view(b, -1, h, w)
            Gs_in0 = Gs0.permute(0, 2, 1).view(b, h*w, h, w)
            Gs_out0 = Gs0.view(b, h*w, h, w)
            Gs_joint0 = torch.cat((Gs_in0,Gs_out0),1)
            Gs_joint0 = torch.cat((Gs_joint0, gamma_feat0), 1)
            Gs_joint0 = self.gg_temporal(Gs_joint0)

            gamma_feat1 = gamma_feat[:, idx+1, :, :].permute(0, 2, 1)
            beta_feat1 = beta_feat[:, idx, :, :]
            Gs1 = torch.matmul(gamma_feat1, beta_feat1)
            gamma_feat1 = gamma_feat1.view(b, -1, h, w)
            Gs_in1 = Gs1.permute(0, 2, 1).view(b, h*w, h, w)
            Gs_out1 = Gs1.view(b, h*w, h, w)
            Gs_joint1 = torch.cat((Gs_in1, Gs_out1),1)
            Gs_joint1 = torch.cat((Gs_joint1, gamma_feat1),1)
            Gs_joint1 = self.gg_temporal(Gs_joint1)

            para_00 = para_00 * Gs_joint0
            para_01 = para_01 * Gs_joint1

            gap_map0 = para_00 * feat_map[:, idx, :, :, :] + para_01 * feat_map[:, idx + 1, :, :, :]
            gap_map0 = gap_map0 ** 2
            gap_feat_map0.append(gap_map0)

        gap_feat_map0 = torch.stack(gap_feat_map0, 1)
        seq_len = gap_feat_map0.size(1)
        feat_vect = feat_para.view(b, t, -1)
        alphi_feat = self.alphi_appearance(reshape_map).view(b*t, -1, h*w)
        delta_feat = self.delta_appearance(reshape_map).view(b*t, -1, h*w)
        alphi_feat = alphi_feat.permute(0, 2, 1)
        Gs = torch.matmul(delta_feat, alphi_feat)
        alphi_feat = alphi_feat.view(b*t, -1, h, w)
        Gs_in = Gs.permute(0, 2, 1).view(b*t, h*w, h, w)
        Gs_out = Gs.view(b*t, h*w, h, w)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)
        Gs_joint = torch.cat((Gs_joint, alphi_feat), 1)
        Gs_joint = self.gg_spatial(Gs_joint).view(b, t, -1, h, w)

        gap_feat_map = []
        for i in range(seq_len):
            idx = 2 * i

            para_0 = self.relu(self.para_fc1_0(feat_vect[:, idx, :]))
            para_1 = self.relu(self.para_fc1_0(feat_vect[:, idx+1, :]))

            para_0 = self.sigmoid(self.para_fc1_1(para_0)).view(b, -1, 1, 1)
            para_1 = self.sigmoid(self.para_fc1_1(para_1)).view(b, -1, 1, 1)

            para_0 = para_0 * Gs_joint[:, idx, :, :, :]
            para_1 = para_1 * Gs_joint[:, idx+1, :, :, :]

            gap_map = gap_feat_map0[:, i, :, :, :] + para_0 * feat_map[:, idx, :, :, :] + para_1 * feat_map[:, idx + 1, :, :, :]
            gap_feat_map.append(gap_map)

        gap_feat_map = torch.stack(gap_feat_map, 1)
        gap_feat_map = gap_feat_map.view(b * seq_len, -1, w, h)

        gap_feat_map = self.conv_block(gap_feat_map)
        gap_feat_vect = self.avg(gap_feat_map).view(b, seq_len, -1)

        if seq_len != 1:
            feature = self.cat_vect(gap_feat_vect)
            feature = self.sigmoid(feature)
        else:
            feature = gap_feat_vect

        feature = feature.view(b, -1)
        gap_feat_map = gap_feat_map.view(b, seq_len, -1, w, h)

        return gap_feat_map, feature