#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from copy import deepcopy
from torch import nn
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
def findConv2dOutShape(H_in, W_in, D_in, kernel_size, stride, padding, dilation, pool=2):
    if stride is None: stride=1
    if padding is None: padding=0
    if dilation is None: dilation=1
    if type(kernel_size) == int: kernel_size = [kernel_size,kernel_size,kernel_size]
    if type(stride) == int: stride = [stride,stride,stride]
    if type(padding) == int: padding = [padding,padding,padding]
    if type(dilation) == int: dilation = [dilation,dilation,dilation]

    H_out = np.floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1)-1) / stride[0] + 1)
    W_out = np.floor((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1)-1) / stride[1] + 1)
    D_out = np.floor((D_in + 2*padding[2] - dilation[2]*(kernel_size[2]-1)-1) / stride[2] + 1)

    if pool:
        H_out /= pool
        W_out /= pool
        D_out /= pool
    
    print('in', H_in, W_in, D_in, 'out', int(H_out), int(W_out), int(D_out))

class DDense_Block(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dropout_rate, padding, dilation, check_size=False):
        super(DDense_Block, self).__init__()
        self.conv_list = nn.ModuleList()
        self.inplanes = inplanes
        self.planes = planes
        self.check_size = check_size         
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation

        self.conv_1 = nn.Conv3d(
            in_channels=inplanes, out_channels=planes, 
            kernel_size=kernel_size, padding=padding[0], dilation=dilation[0])
        # self.conv_2 = nn.Conv3d(in_channels=(inplanes + planes), out_channels=planes, kernel_size=kernel_size, padding=2,  dilation=2)
        # self.conv_3 = nn.Conv3d(in_channels=(inplanes + 2 * planes), out_channels=planes, kernel_size=kernel_size, padding=3, dilation=3)
        self.conv_2 = nn.Conv3d(
            in_channels=(inplanes + planes), out_channels=planes, 
            kernel_size=kernel_size, padding=padding[1], dilation=dilation[1])
        self.conv_3 = nn.Conv3d(
            in_channels=(inplanes + 2 * planes), out_channels=planes, 
            kernel_size=kernel_size, padding=padding[2], dilation=dilation[2])

        self.norm = nn.BatchNorm3d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv_1_1 = nn.Conv3d(
            (inplanes + 3 * planes), planes, 
            kernel_size=[1 for _ in kernel_size],
            padding=[0 for i in kernel_size])
        self.norm_1_1 = nn.BatchNorm3d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin_1_1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
        # dropout
        self.dropout = nn.Dropout3d(dropout_rate, inplace=True)

    def forward(self, x): # x : (4,32,96,96,64)
        out = self.conv_1(x)
        if self.check_size:
            findConv2dOutShape(
                x.shape[-3],x.shape[-2],x.shape[-1],pool=1,
                kernel_size=self.kernel_size,stride=None,padding=self.padding[0],dilation=self.dilation[0])
            
        out = self.dropout(out)
        out = self.norm(out)
        out = self.nonlin(out)  # out_1 : (4,32,94,94,62)
        residual_1 = torch.cat((x, out), dim=1)  # 32 + 32
        
        # print('here1', residual_1.shape, self.inplanes+self.planes)

        out = self.conv_2(residual_1)
        if self.check_size:
            findConv2dOutShape(
                residual_1.shape[-3],residual_1.shape[-2],residual_1.shape[-1],pool=1,
                kernel_size=self.kernel_size,stride=None,padding=self.padding[1],dilation=self.dilation[1])
        
        out = self.dropout(out)
        out = self.norm(out)
        out = self.nonlin(out)  # 32
        residual_1 = torch.cat((out, residual_1), dim=1)  # 32 + 64
        
        # print('here2', residual_2.shape, self.inplanes+2*self.planes)

        out = self.conv_3(residual_1)
        if self.check_size:
            findConv2dOutShape(
                residual_1.shape[-3],residual_1.shape[-2],residual_1.shape[-1],pool=1,
                kernel_size=self.kernel_size,stride=None,padding=self.padding[2],dilation=self.dilation[2])
            
        out = self.dropout(out)
        out = self.norm(out)
        out = self.nonlin(out)  # 32
        out = torch.cat((residual_1, out), dim=1)  # 96 + 32

        # print('here3', out.shape, self.inplanes+3*self.planes)

        out = self.norm_1_1(self.conv_1_1(out))
        out = self.nonlin_1_1(out)  # 32
        return out


class DDenseDownBlock_first(nn.Module):

    def __init__(self, in_planes, kernel_size, dropout_rate, padding, dilation, check_size=False):
        super().__init__()
        self.conv1 = DDense_Block(in_planes, in_planes, 
            kernel_size=kernel_size, dropout_rate=dropout_rate, 
            padding=padding, dilation=dilation, check_size=check_size)
        self.conv2 = DDense_Block(in_planes*2, in_planes, 
            kernel_size=kernel_size, dropout_rate=dropout_rate, 
            padding=padding, dilation=dilation, check_size=check_size)
        self.pool_op = nn.MaxPool3d(2,stride=2)

    def forward(self, x):
        residual_1 = x  # 32
        x = self.conv1(x)
        concat_1 = torch.cat((x, residual_1), dim=1)  # 32 * 2
        residual_1 = self.conv2(concat_1)  # 32 * 2
        out = self.pool_op(residual_1)
        return out, residual_1



# 여기서 DDenseBlock 구현

class DDenseDownBlock_2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dropout_rate, padding, dilation, check_size=False):

        super().__init__()

        self.kernel_size = kernel_size
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = out_planes // 4

        # maxpooling 구현
        self.pool_op = nn.MaxPool3d(2,stride=2)

        self.conv1 = DDense_Block(in_planes, in_planes, 
            kernel_size=kernel_size, dropout_rate=dropout_rate, 
            padding=padding, dilation=dilation, check_size=check_size)
        self.conv2 = DDense_Block(in_planes*2, in_planes*2, 
            kernel_size=kernel_size, dropout_rate=dropout_rate, 
            padding=padding, dilation=dilation, check_size=check_size)

        # conv3 = 1*1 conv
        self.conv3 = nn.Conv3d(in_planes * 4, in_planes * 2, [1 for _ in kernel_size], padding=[0 for i in kernel_size])
        self.norm3 = nn.BatchNorm3d(in_planes * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    
    def forward(self, x):
        residual_1 = x  # 32
        out_1 = self.conv1(x)  # 32
        residual_2 = out_1
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 32 * 2

        out = self.conv2(concat_1)  # 32 * 2

        concat_1 = torch.cat((out, residual_1), dim=1)  # 32*2 + 32*1 = 32 * 3
        concat_1 = torch.cat((concat_1, residual_2), dim = 1) # 32*3 + 32* = 32 * 4

        out = self.norm3(self.conv3(concat_1))
        residual_out = self.nonlin3(out)
        out = self.pool_op(residual_out)

        del residual_1, residual_2, out_1, concat_1
        return out, residual_out


class DDenseDownLayer_first(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, drop_rate, padding, dilation, check_size=False, block=DDenseDownBlock_first):
        super().__init__()
        self.convs = nn.Sequential(
            block(in_channel, kernel_size, drop_rate, padding, dilation, check_size)
        )
    def forward(self, x):
        return self.convs(x)

class DDenseDownLayer_2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, drop_rate, padding, dilation, check_size=False, block=DDenseDownBlock_2):
        super().__init__()
        self.convs = nn.Sequential(
            block(in_channel, out_channel, kernel_size, drop_rate, padding, dilation, check_size)
        )
    def forward(self, x):
        return self.convs(x)



# 여기는 Dense_Up_Block 구현하기

class DenseUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dropout_rate, padding, dilation, check_size=False):

        super().__init__()

        aim_planes = in_planes // 2  # 256
        self.conv0 = nn.Conv3d(in_planes, aim_planes, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size])

        self.norm0 = nn.BatchNorm3d(aim_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin0 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv1 = DDense_Block(aim_planes, aim_planes, 
            kernel_size=kernel_size, dropout_rate=dropout_rate, 
            padding=padding, dilation=dilation, check_size=check_size)
        self.conv2 = DDense_Block(aim_planes*2, aim_planes, 
            kernel_size=kernel_size, dropout_rate=dropout_rate, 
            padding=padding, dilation=dilation, check_size=check_size)

        self.conv3 = nn.Conv3d(aim_planes * 3, aim_planes, [1 for _ in kernel_size],padding=[0 for i in kernel_size])
        self.norm3 = nn.BatchNorm3d(aim_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)


    def forward(self, x):
        x = self.norm0(self.conv0(x))
        x = self.nonlin0(x)  # 256
        residual_1 = x  # 256

        out = self.conv1(x)  # 256
        residual_2 = out  # 256
        concat_1 = torch.cat((out, residual_1), dim=1)  # 512

        out = self.conv2(concat_1)  # 256

        concat_2 = torch.cat((out, residual_1), dim=1)  # 512
        concat_2 = torch.cat((concat_2,residual_2), dim = 1) # 512 + 256

        out = self.norm3(self.conv3(concat_2))
        out = self.nonlin3(out)

        return out


class DenseUpLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, drop_rate, padding, dilation, block=DenseUpBlock):
        super().__init__()

        self.convs = nn.Sequential(
            block(in_channel, out_channel, kernel_size, drop_rate, padding, dilation)
        )

    def forward(self, x):
        return self.convs(x)







# Coordinate Attention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True): # True):
        super(h_sigmoid, self).__init__()
        # self.relu = nn.ReLU6(inplace=inplace)
        self.relu = nn.ReLU6(inplace=False)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):  # True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()


        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_d = nn.AdaptiveAvgPool3d((1, 1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x   #(1,512,24,22,4)

        b, c, h, w, d = x.size()
        x_h = self.pool_h(x)  # (1,512,24,1,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2, 4) # (1,512,22,1,1)
        x_d = self.pool_d(x).permute(0, 1, 4, 3, 2) # (1,512,4,1,1)

        y = torch.cat([x_h, x_w, x_d], dim=2)   # (1,512,50,1,1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) # (1,16,50,1,1)

        x_h, x_w, x_d = torch.split(y, [h, w, d], dim=2)
        x_w = x_w.permute(0, 1, 3, 2, 4)
        x_d = x_d.permute(0, 1, 4, 3, 2)

        a_d = self.conv_d(x_d).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_d * a_w * a_h

        return out