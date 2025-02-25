import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict


from Resources.Segmentator_3D.JEpark.CommonCT_custom.core.custom_blocks.BaseModelClass_MedicalZoo import BaseModel

"""
Implementations based on the HyperDenseNet paper: https://arxiv.org/pdf/1804.02967.pdf
"""


class _HyperDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_channels, drop_rate):
        super(_HyperDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features,
                                           num_output_channels, kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_HyperDenseLayer, self).forward(x)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)

        return torch.cat([x, new_features], 1)


class _HyperDenseBlock(nn.Sequential):
    """
    Constructs a series of dense-layers based on in and out kernels list
    """

    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlock, self).__init__()
        out_kernels = [1, 25, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 9

        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])

        print("out:", out_kernels)
        print("in:", in_kernels)

        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _HyperDenseBlockEarlyFusion(nn.Sequential):
    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlockEarlyFusion, self).__init__()
        out_kernels = [1, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 8

        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])

        print("out:", out_kernels)
        print("in:", in_kernels)

        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)




class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop_layer = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.drop_layer(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=2, stride=2))


class SkipDenseNet3D(BaseModel):
    """Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Based on the implementation of https://github.com/tbuikr/3D-SkipDenseSeg
    Paper here : https://arxiv.org/pdf/1709.03199.pdf

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        classes (int) - number of classification classes
    """

    def __init__(self, 
                 in_channels=2, 
                 classes=4, 
                 growth_rate=16, 
                 block_config=(4, 4, 4, 4), 
                 num_init_features=32, 
                 drop_rate=0.1,
                 bn_size=4):

        super(SkipDenseNet3D, self).__init__()
        self.num_classes = classes
        # First three convolutions
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        self.features_bn = nn.Sequential(OrderedDict([
            ('norm2', nn.BatchNorm3d(num_init_features)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.conv_pool_first = nn.Conv3d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0,
                                         bias=False)

        # Each denseblock
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList([])
        self.transit_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            
            up_block = nn.ConvTranspose3d(num_features, classes, kernel_size=2 ** (i + 1) + 2,
                                          stride=2 ** (i + 1),
                                          padding=1, bias=False) # I removed group=classes. Please check how the perfomance is changed. 

            self.upsampling_blocks.append(up_block)

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                # self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        # self.bn4 = nn.BatchNorm3d(num_features)

        # ----------------------- classifier -----------------------
        self.bn_class = nn.BatchNorm3d(classes * 4 + num_init_features)
        self.conv_class = nn.Conv3d(classes * 4 + num_init_features, classes, kernel_size=1, padding=0)
        self.relu_last = nn.ReLU()
        # ----------------------------------------------------------

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                # nn.Conv3d.bias.data.fill_(-0.1)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, self.input_channels, 12, 12, 12)
        ideal_out = torch.rand(1, self.num_classes, 12, 12, 12)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        print("SkipDenseNet3D-1 test is complete")

    def forward(self, x):
        first_three_features = self.features(x)
        first_three_features_bn = self.features_bn(first_three_features)
        out = self.conv_pool_first(first_three_features_bn)

        out = self.dense_blocks[0](out)
        up_block1 = self.upsampling_blocks[0](out)
        out = self.transit_blocks[0](out)

        out = self.dense_blocks[1](out)
        up_block2 = self.upsampling_blocks[1](out)
        out = self.transit_blocks[1](out)

        out = self.dense_blocks[2](out)
        up_block3 = self.upsampling_blocks[2](out)
        out = self.transit_blocks[2](out)

        out = self.dense_blocks[3](out)
        up_block4 = self.upsampling_blocks[3](out)

        out = torch.cat([up_block1, up_block2, up_block3, up_block4, first_three_features], 1)

        # ----------------------- classifier -----------------------
        out = self.conv_class(self.relu_last(self.bn_class(out)))
        # ----------------------------------------------------------
        return out
    

class SinglePathDenseNet(BaseModel):
    def __init__(self, in_channels, classes=4, drop_rate=0.1, return_logits=True, early_fusion=False):
        super(SinglePathDenseNet, self).__init__()
        self.return_logits = return_logits
        self.features = nn.Sequential()
        self.num_classes = classes
        self.input_channels = in_channels

        if early_fusion:
            block = _HyperDenseBlockEarlyFusion(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 52:
                total_conv_channels = 477
            else:
                if in_channels == 3:
                    total_conv_channels = 426
                else:
                    total_conv_channels = 503

        else:
            block = _HyperDenseBlock(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 2:
                total_conv_channels = 452
            else:
                total_conv_channels = 451

        self.features.add_module('denseblock1', block)

        self.features.add_module('conv1x1_1', nn.Conv3d(total_conv_channels,
                                                        400, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_1', nn.Dropout(p=0.5))

        self.features.add_module('conv1x1_2', nn.Conv3d(400,
                                                        200, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_2', nn.Dropout(p=0.5))

        self.features.add_module('conv1x1_3', nn.Conv3d(200,
                                                        150, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_3', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(150,
                                                           self.num_classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))

    def forward(self, x):
        features = self.features(x)
        if self.return_logits:
            out = self.classifier(features)
            return out

        else:
            return features

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, self.input_channels, 12, 12, 12)
        ideal_out = torch.rand(1, self.num_classes, 12, 12, 12)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        print("DenseNet3D-1 test is complete")


class DualPathDenseNet(BaseModel):
    def __init__(self, in_channels, classes=4, drop_rate=0, fusion='concat'):
        """
        2-stream and 3-stream implementation with late fusion
        :param in_channels: 2 or 3 (dual or triple path based on paper specifications).
        Channels are the input modalities i.e T1,T2 etc..
        :param drop_rate:  dropout rate for dense layers
        :param classes: number of classes to segment
        :param fusion: 'concat or 'sum'
        """
        super(DualPathDenseNet, self).__init__()
        self.input_channels = in_channels
        self.num_classes = classes

        self.fusion = fusion
        if self.fusion == "concat":
            in_classifier_channels = self.input_channels * 150
        else:
            in_classifier_channels = 150

        if self.input_channels == 2:
            # here!!!!
            self.stream_1 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False, early_fusion=True)
            self.stream_2 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False, early_fusion=True)

        if self.input_channels == 3:
            self.stream_1 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False)
            self.stream_2 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False)
            self.stream_3 = SinglePathDenseNet(in_channels=1, drop_rate=drop_rate, classes=classes,
                                               return_logits=False)

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(in_classifier_channels,
                                                           classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))

    def forward(self, multi_channel_medical_img):
        """
        :param multi_channel_medical_img: shape of [batch, input_channels, height, width, depth]
        :return: late fusion classification predictions
        """
        channels = multi_channel_medical_img.shape[1]
        if channels != self.input_channels:
            print("Network channels does not match input channels, check your model/input!")
            return None
        else:
            if self.input_channels == 2:
                in_stream_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
                in_stream_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
                output_features_t1 = self.stream_1(in_stream_1)
                output_features_t2 = self.stream_2(in_stream_2)

                if self.fusion == 'concat':
                    concat_features = torch.cat((output_features_t1, output_features_t2), dim=1)
                    return self.classifier(concat_features)
                else:
                    features = output_features_t1 + output_features_t2
                    return self.classifier(features)
            elif self.input_channels == 3:
                in_stream_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
                in_stream_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
                in_stream_3 = multi_channel_medical_img[:, 2, ...].unsqueeze(dim=1)
                output_features_t1 = self.stream_1(in_stream_1)
                output_features_t2 = self.stream_2(in_stream_2)
                output_features_t3 = self.stream_3(in_stream_3)
                if self.fusion == 'concat':
                    concat_features = torch.cat((output_features_t1, output_features_t2, output_features_t3), dim=1)
                    return self.classifier(concat_features)
                else:
                    features = output_features_t1 + output_features_t2 + output_features_t3
                    return self.classifier(features)


class DualSingleDenseNet(BaseModel):
    """
    2-stream and 3-stream implementation with early fusion
    dual-single-densenet OR Disentangled modalities with early fusion in the paper
    """

    def __init__(self, in_channels, classes=4, drop_rate=0.5,):
        """

        :param input_channels: 2 or 3 (dual or triple path based on paper specifications).
        Channels are the input modalities i.e T1,T2 etc..
        :param drop_rate:  dropout rate for dense layers
        :param classes: number of classes to segment
        :param fusion: 'concat or 'sum'
        """
        super(DualSingleDenseNet, self).__init__()
        self.input_channels = in_channels
        self.num_classes = classes

        if self.input_channels == 2:
            self.early_conv_1 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=drop_rate)
            self.early_conv_2 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=drop_rate)
            single_path_channels = 52
            self.stream_1 = SinglePathDenseNet(in_channels=single_path_channels, drop_rate=drop_rate,
                                               classes=classes, return_logits=True, early_fusion=True)
            self.classifier = nn.Sequential()

        if self.input_channels == 3:
            self.early_conv_1 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_2 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            self.early_conv_3 = _HyperDenseLayer(num_input_features=1, num_output_channels=25, drop_rate=0)
            single_path_channels = 78
            self.stream_1 = SinglePathDenseNet(in_channels=single_path_channels, drop_rate=drop_rate,
                                               classes=classes, return_logits=True, early_fusion=True)

    def forward(self, multi_channel_medical_img):
        """
        :param multi_channel_medical_img: shape of [batch, input_channels, height, width, depth]
        :return: late fusion classification predictions
        """
        channels = multi_channel_medical_img.shape[1]
        if channels != self.input_channels:
            print("Network channels does not match input channels, check your model/input!")
            return None
        else:
            if self.input_channels == 2:
                in_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
                in_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
                y1 = self.early_conv_1(in_1)
                y2 = self.early_conv_1(in_2)
                print(y1.shape)
                print(y2.shape)
                in_stream = torch.cat((y1, y2), dim=1)
                logits = self.stream_1(in_stream)
                return logits

            elif self.input_channels == 3:
                in_1 = multi_channel_medical_img[:, 0, ...].unsqueeze(dim=1)
                in_2 = multi_channel_medical_img[:, 1, ...].unsqueeze(dim=1)
                in_3 = multi_channel_medical_img[:, 2, ...].unsqueeze(dim=1)
                y1 = self.early_conv_1(in_1)
                y2 = self.early_conv_2(in_2)
                y3 = self.early_conv_3(in_3)
                in_stream = torch.cat((y1, y2, y3), dim=1)
                logits = self.stream_1(in_stream)
                return logits