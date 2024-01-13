import math
import torch
from torch import nn
import torch.nn.functional as F

def conv_block(in_chan, out_chan, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm3d(out_chan),
        nn.ReLU(inplace=True)
    )


def conv_stage(in_chan, out_chan):
    return nn.Sequential(
        conv_block(in_chan, out_chan),
        conv_block(out_chan, out_chan),
    )


class UNetPP_3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc00 = conv_stage(1, 16)
        self.enc10 = conv_stage(16, 32)
        self.enc20 = conv_stage(32, 64)
        self.enc30 = conv_stage(64, 128)
        self.enc40 = conv_stage(128, 128)
        self.pool = nn.MaxPool3d(2, 2)

        self.dec41 = conv_stage(256, 64)
        self.dec32 = conv_stage(64+32+64, 32)
        self.dec23 = conv_stage(32+16+16+32, 16)
        self.dec14 = conv_stage(16+16+16+16+16, 16)
        self.conv_out1 = nn.Conv3d(16, 1, 1)

        self.dec31 = conv_stage(64+128, 32)
        self.dec22 = conv_stage(32+16+32, 16)
        self.dec13 = conv_stage(16+16+16+16, 16)
        self.conv_out2 = nn.Conv3d(16, 1, 1)

        self.dec21 = conv_stage(32+64, 16)
        self.dec12 = conv_stage(16+16+16, 16)
        self.conv_out3 = nn.Conv3d(16, 1, 1)

        self.dec11 = conv_stage(16+32, 16)
        self.conv_out4 = nn.Conv3d(16, 1, 1)

        self.conv_out5 = nn.Conv3d(16, 1, 1)
        

    def forward(self, x):
        # x: [batch_size, 1, depth, height, width]
        # out: [batch_size, 1, depth, height, width]
        x10 = self.enc00(x)
        x20 = self.enc10(self.pool(x10))
        x30 = self.enc20(self.pool(x20))
        x40 = self.enc30(self.pool(x30))
        x50 = self.enc40(self.pool(x40))

        x41 = self.dec41(torch.cat((x40, F.upsample(x50, x40.size()[2:], mode='trilinear')), 1))
        x31 = self.dec31(torch.cat((x30, F.upsample(x40, x30.size()[2:], mode='trilinear')), 1))
        x21 = self.dec21(torch.cat((x20, F.upsample(x30, x20.size()[2:], mode='trilinear')), 1))
        x11 = self.dec11(torch.cat((x10, F.upsample(x20, x10.size()[2:], mode='trilinear')), 1))
        out1 = self.conv_out1(x10)
        
        x32 = self.dec32(torch.cat((x30, x31, F.upsample(x41, x31.size()[2:], mode='trilinear')), 1))
        x22 = self.dec22(torch.cat((x20, x21, F.upsample(x31, x21.size()[2:], mode='trilinear')), 1))
        x12 = self.dec12(torch.cat((x10, x11, F.upsample(x21, x11.size()[2:], mode='trilinear')), 1))
        out2 = self.conv_out2(x11)

        x23 = self.dec23(torch.cat((x20, x21, x22, F.upsample(x32, x22.size()[2:], mode='trilinear')), 1))
        x13 = self.dec13(torch.cat((x10, x11, x12, F.upsample(x22, x12.size()[2:], mode='trilinear')), 1))
        out3 = self.conv_out3(x12)

        x14 = self.dec14(torch.cat((x10, x11, x12, x13, F.upsample(x23, x13.size()[2:], mode='trilinear')), 1))
        out4 = self.conv_out4(x13)

        out5 = self.conv_out5(x14)

        
        # out = F.sigmoid(out)
        return (out1 + out2 + out3 + out4 + out5) / 5