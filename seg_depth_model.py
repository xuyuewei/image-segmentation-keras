from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        print('SegDepthLoss inialized')

    def forward(self, seg_pred, seg_truth):
        seg_pred = torch.reshape(seg_pred, (-1, ))
        seg_truth = torch.reshape(seg_truth, (-1,))

        intersection = torch.sum(seg_pred * seg_truth)
        score = (2.0 * intersection + self.smooth) / (torch.sum(seg_pred) + torch.sum(seg_truth) + self.smooth)
        loss = 1 - score
        return loss


class SegDepthModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=10, n_depth=1):
        super().__init__()
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_depth = n_depth
        self.kernel_size = 3
        self.light_kernel_size = 1

        self.uconv1 = self.conv_block(self.n_channels, 32)
        self.uconv2 = self.conv_block(32, 64)
        self.uconv3 = self.conv_block(64, 128)
        self.uconv4 = self.conv_block(128, 256)
        self.uconv5 = self.conv_block(256, 512)

        self.ucenter = self.conv_block(512, 1024)

        self.deconv5 = self.deconv_block(1024, 512)
        self.uconv5_ = self.conv_block(1024, 512)
        self.deconv4 = self.deconv_block(512, 256)
        self.uconv4_ = self.conv_block(512, 256)
        self.deconv3 = self.deconv_block(256, 128)
        self.uconv3_ = self.conv_block(256, 128)
        self.deconv2 = self.deconv_block(128, 64)
        self.uconv2_ = self.conv_block(128, 64)
        self.deconv1 = self.deconv_block(64, 32)
        self.uconv1_ = self.conv_block(64, 32)

        self.res_conv1 = self.res_conv(32, 32)
        self.res_conv2 = self.res_conv(64, 64)
        self.res_conv3 = self.res_conv(128, 128)
        self.res_conv4 = self.res_conv(256, 256)
        self.res_conv5 = self.res_conv(512, 512)
        self.res_dconv1_ = self.res_conv(32, 32)
        self.res_dconv2_ = self.res_conv(64, 64)
        self.res_dconv3_ = self.res_conv(128, 128)
        self.res_dconv4_ = self.res_conv(256, 256)
        self.res_dconv5_ = self.res_conv(512, 512)

        self.conv_drop = nn.Dropout2d(p=0.2)

    def conv_block(self, in_dim, ou_dim, kernel_size=3):
        return nn.Sequential(nn.Conv2d(in_dim, ou_dim, kernel_size=kernel_size, padding=1, dilation=1),
                             nn.BatchNorm2d(ou_dim),
                             nn.ReLU(True)
                             )

    def res_conv(self, in_dim, ou_dim, kernel_size=1):
        return nn.Sequential(nn.Conv2d(in_dim, ou_dim, kernel_size=kernel_size, dilation=2),
                             nn.BatchNorm2d(ou_dim),
                             nn.ReLU(True),
                             nn.Conv2d(ou_dim, ou_dim, kernel_size=kernel_size, dilation=1),
                             nn.BatchNorm2d(ou_dim),
                             nn.ReLU(True)
                             )

    def res_block(self, x, in_dim, ou_dim):
        cx = self.conv_block(in_dim, ou_dim)(x)
        rx = self.res_conv(ou_dim, ou_dim)(cx)
        rx = cx + rx
        cx = self.conv_block(ou_dim, ou_dim)(rx)
        return cx

    def deconv_block(self, in_dim, ou_dim):
        return nn.Sequential(nn.ConvTranspose2d(in_dim, ou_dim, kernel_size=2, stride=2),
                             nn.BatchNorm2d(ou_dim),
                             nn.ReLU(True)
                             )

    def res_unet(self, x):
        cx1 = self.uconv1(x)
        rcx1 = cx1 + self.res_conv1(cx1)
        pcx1 = nn.MaxPool2d(2, stride=2)(rcx1)

        cx2 = self.uconv2(pcx1)
        rcx2 = cx2 + self.res_conv2(cx2)
        pcx2 = nn.MaxPool2d(2, stride=2)(rcx2)

        cx3 = self.uconv3(pcx2)
        rcx3 = cx3 + self.res_conv3(cx3)
        pcx3 = nn.MaxPool2d(2, stride=2)(rcx3)

        cx4 = self.uconv4(pcx3)
        rcx4 = cx4 + self.res_conv4(cx4)
        pcx4 = nn.MaxPool2d(2, stride=2)(rcx4)

        cx5 = self.uconv5(pcx4)
        rcx5 = cx1 + self.res_conv5(cx5)
        pcx5 = nn.MaxPool2d(2, stride=2)(rcx5)

        center = self.ucenter(pcx5)

        dx5 = self.deconv5(center)
        res_x5 = dx5 + cx5
        c_5 = torch.cat((dx5, cx5, res_x5), dim=0)
        resc_5 = self.res_dconv5_(c_5)
        resc_5 = resc_5 + c_5
        cc_5 = self.uconv5_(resc_5)

        dx4 = self.deconv4(cc_5)
        res_x4 = dx4 + cx4
        c_4 = torch.cat((dx4, cx4, res_x4), dim=0)
        resc_4 = self.res_dconv5_(c_4)
        resc_4 = resc_4 + c_4
        cc_4 = self.uconv4_(resc_4)

        dx3 = self.deconv3(cc_4)
        res_x3 = dx3 + cx3
        c_3 = torch.cat((dx3, cx3, res_x3), dim=0)
        resc_3 = self.res_dconv3_(c_3)
        resc_3 = resc_3 + c_3
        cc_3 = self.uconv3_(resc_3)

        dx2 = self.deconv2(cc_3)
        res_x2 = dx2 + cx2
        c_2 = torch.cat((dx2, cx2, res_x2), dim=0)
        resc_2 = self.res_dconv2_(c_2)
        resc_2 = resc_2 + c_2
        cc_2 = self.uconv2_(resc_2)

        dx1 = self.deconv1(cc_2)
        res_x1 = dx1 + cx1
        c_1 = torch.cat((dx1, cx1, res_x1), dim=0)
        resc_1 = self.res_dconv1_(c_1)
        resc_1 = resc_1 + c_1
        cc_1 = self.uconv1_(resc_1)

        return cc_1

    def seg_rescnn(self, x):
        x = self.res_block(x, 32, 64)
        x = self.res_block(x, 64, 128)
        x = self.res_block(x, 128, 64)
        x = self.res_block(x, 64, 32)
        out_x = nn.Conv2d(32, self.n_classes, kernel_size=1, dilation=1)(x)
        out_x = nn.Softmax2d()(out_x)
        out_multiclass_x = out_x.view(self.n_classes, self.height*self.width)
        out_multiclass_x.permute(1, 0)
        return out_multiclass_x

    def depth_rescnn(self, x):
        x = self.res_block(x, 32, 64)
        x = self.res_block(x, 64, 128)
        x = self.res_block(x, 128, 256)
        x = self.res_block(x, 256, 128)
        x = self.res_block(x, 128, 64)
        x = self.res_block(x, 64, 32)
        out_x = self.conv_block(32, self.n_depth, kernel_size=1)(x)
        return out_x

    def forward(self, xl, xr):
        ul = self.res_unet(xl)
        ur = self.res_unet(xr)

        seg_softmax = self.seg_rescnn(ul)

        diff = ul + ur
        lr_merge = torch.cat((ul, ur, diff), dim=0)
        depth_relu = self.depth_rescnn(lr_merge)

        return seg_softmax, depth_relu
