import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseResBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout=0.0, dilation_rate=1, act='leaky'):
        super(DenseResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.norm1 = nn.BatchNorm3d(growth_rate)
        self.relu = nn.LeakyReLU(inplace=True) if act == 'leaky' else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
        
        # Update shortcut to match the output channels after dense growth
        self.shortcut = nn.Conv3d(in_channels + growth_rate, in_channels + growth_rate, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        if self.dropout:
            out = self.dropout(out)
        
        out = torch.cat([x, out], dim=1)  # Dense connection
        residual = self.shortcut(out)  # Adjusted shortcut to match output channels
        out += residual
        out = self.relu(out)
        return out

class DenseResUNet3D(nn.Module):
    def __init__(self, 
                 in_channels=1, out_channels=1, 
                 base_filters=16, num_blocks=4, 
                 growth_rate=16, dropout=0.0,
                 act='leaky', dilation_rate=1, 
                 deep_supervision=False
        ):
        super(DenseResUNet3D, self).__init__()
        self.num_blocks = num_blocks
        self.base_filters = base_filters
        self.growth_rate = growth_rate
        self.deep_supervision = deep_supervision

        # Contracting path
        self.contracting_layers = nn.ModuleList()
        in_channels = in_channels
        channel_list = [in_channels]
        for i in range(num_blocks):
            block = DenseResBlock(in_channels, growth_rate, dropout=dropout, dilation_rate=2**i, act=act)
            self.contracting_layers.append(block)
            in_channels += growth_rate
            if i < num_blocks - 1:  # No pooling after the last block
                self.contracting_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            channel_list.append(in_channels)

        # Expansive path
        self.expansive_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        in_channels = channel_list[-1]
        for i in range(num_blocks - 1, 0, -1):
            # Reduce channels for upconv, matching the output from the corresponding encoder block
            upconv = nn.ConvTranspose3d(in_channels, channel_list[i], kernel_size=2, stride=2)
            self.expansive_layers.append(upconv)
            # print(i, 0, in_channels)            
            # After upsampling, concatenation will double the number of input channels
            block = DenseResBlock(channel_list[i] * 2, growth_rate, dropout=dropout, dilation_rate=2**(i - 1), act=act)
            self.expansive_layers.append(block)
            if deep_supervision:
                out = nn.Conv3d(channel_list[i] * 2, out_channels, kernel_size=1)
                self.out_layers.append(out)
            in_channels = channel_list[i] * 2 + growth_rate
            # print(i, 1, in_channels, channel_list[i])
        # Final output layer
        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.moduels = nn.ModuleList(
            self.contracting_layers + self.expansive_layers + self.out_layers + [self.out_conv])
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight) # xavier_normal_
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Contracting path
        skips = []
        for i, layer in enumerate(self.contracting_layers):
            x = layer(x)
            if isinstance(layer, DenseResBlock):
                skips.append(x)
            # print(f"Contracting : {i}, Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}")
        # Expansive path
        skips = skips[::-1]
        deep_supervision_outputs = []
        # print('Skips:', [s.shape for s in skips])
        for i in range(0, len(self.expansive_layers), 2):
            x = self.expansive_layers[i](x)
            # print(f"Expansive : {i}, 0, {x.shape}, Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}")
            skip = skips[i//2+1]
            # print(f"Expansive : {i}, 1, {skip.shape}, Min: {skip.min():.4f}, Max: {skip.max():.4f}, Mean: {skip.mean():.4f}")
            x = torch.cat([x, skip], dim=1)
            if self.deep_supervision:
                deep_supervision_outputs.append(self.out_layers[math.ceil(i/2)](x))
            # print(f"Expansive : {i}, 2, {x.shape}, Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}")
            x = self.expansive_layers[i+1](x)
            # print(f"Expansive : {i}, 3, {x.shape}, Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}")
        
        # Final output layer
        if self.deep_supervision:
            deep_supervision = [self.out_conv(x)] + deep_supervision_outputs
            return deep_supervision
        else:
            outputs = self.out_conv(x)
            # print(f"Final : Min: {outputs.min():.4f}, Max: {outputs.max():.4f}, Mean: {outputs.mean():.4f}")
            return outputs