""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__=[
    "Res_UNet"
]

class FirstConv(nn.Module):
    """(convolution => [BN] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.first_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.first_conv(x)
    

class DownConv(nn.Module):
    """(convolution => [BN] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),  padding=(1, 1, 1)),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResDownConv(nn.Module):
    """(convolution => [BN] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),  padding=(1, 1, 1)),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.InstanceNorm3d(out_channels, affine=True)
            )
        self.relu = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        residual = x
        out = self.double_conv(x)
        
        residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        
        return out
    


class UpConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.conv = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                        nn.InstanceNorm3d(out_channels, affine=True),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),  padding=(1, 1, 1)),
                        nn.InstanceNorm3d(out_channels, affine=True),
                        nn.LeakyReLU(inplace=True)
                    )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
    

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class nnUNet(nn.Module):
    def __init__(self, in_channels, num_classes, channels=32, deep_supervision=False):
        super(nnUNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.channels = channels # default = 64
        self.deep_supervision = deep_supervision # default = 64

        self.inc = FirstConv(in_channels, channels)
        self.down1 = DownConv(channels, channels*2)
        self.down2 = DownConv(channels*2, channels*4)
        self.down3 = DownConv(channels*4, channels*8)
        self.down4 = DownConv(channels*8, channels*16)
        self.down5 = DownConv(channels*16, channels*32)
        self.up1 = UpConv(channels*32, channels*16)
        self.up2 = UpConv(channels*16, channels*8)
        self.up3 = UpConv(channels*8, channels*4)
        self.up4 = UpConv(channels*4, channels*2)
        self.up5 = UpConv(channels*2, channels)
        
        self.out1 = Conv1x1(channels*16, num_classes)
        self.out2 = Conv1x1(channels*8, num_classes)
        self.out3 = Conv1x1(channels*4, num_classes)
        self.out4 = Conv1x1(channels*2, num_classes)
        self.out5 = Conv1x1(channels, num_classes)
    
    def forward(self, x):
        seg_outputs = []
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        seg_outputs.append(self.out1(x))
        x = self.up2(x, x4)
        seg_outputs.append(self.out2(x))
        x = self.up3(x, x3)
        seg_outputs.append(self.out3(x))
        x = self.up4(x, x2)
        seg_outputs.append(self.out4(x))
        x = self.up5(x, x1)
        seg_outputs.append(self.out5(x))
        
        if self.deep_supervision:
            return [seg for seg in seg_outputs[::-1]]
        else:
            return seg_outputs[-1]

class Res_UNet(nn.Module):
    def __init__(self, in_channels, num_classes, channels=32, deep_supervision=False):
        super(Res_UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.channels = channels # default = 64
        self.deep_supervision = deep_supervision # default = 64

        self.inc = FirstConv(in_channels, channels)
        self.down1 = ResDownConv(channels, channels*2)
        self.down2 = ResDownConv(channels*2, channels*4)
        self.down3 = ResDownConv(channels*4, channels*8)
        self.down4 = ResDownConv(channels*8, channels*16)
        self.down5 = ResDownConv(channels*16, channels*32)
        self.up1 = UpConv(channels*32, channels*16)
        self.up2 = UpConv(channels*16, channels*8)
        self.up3 = UpConv(channels*8, channels*4)
        self.up4 = UpConv(channels*4, channels*2)
        self.up5 = UpConv(channels*2, channels)
        
        self.out1 = Conv1x1(channels*16, num_classes)
        self.out2 = Conv1x1(channels*8, num_classes)
        self.out3 = Conv1x1(channels*4, num_classes)
        self.out4 = Conv1x1(channels*2, num_classes)
        self.out5 = Conv1x1(channels, num_classes)
    
    def forward(self, x):
        seg_outputs = []
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        seg_outputs.append(self.out1(x))
        x = self.up2(x, x4)
        seg_outputs.append(self.out2(x))
        x = self.up3(x, x3)
        seg_outputs.append(self.out3(x))
        x = self.up4(x, x2)
        seg_outputs.append(self.out4(x))
        x = self.up5(x, x1)
        seg_outputs.append(self.out5(x))
        
        if self.deep_supervision:
            return [seg for seg in seg_outputs[::-1]]
        else:
            return seg_outputs[-1]
        