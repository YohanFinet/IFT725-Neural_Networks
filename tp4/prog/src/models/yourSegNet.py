# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from models.CNNBaseModel import CNNBaseModel

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourSegNet.  Un réseau très différent du UNet.
Soyez originaux et surtout... amusez-vous!

'''

class YourSegNet(CNNBaseModel):
    """
     Class that implements a brand new segmentation CNN
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds YourSegNet  model.
        Args:
            num_classes(int): number of classes.
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super().__init__(num_classes, init_weights)

        # Load pre-trained ResNet
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the first convolution of the ResNet to accept single-channel inputs
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # remove last layers (avg pool and fc) of ResNet usually used for classification
        modules = list(base.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Atrous Spatial Pyramid Pooling
        self.aspp = ASPP(2048, [12, 24, 36])

        # Final layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        input_shape = x.shape[-2:]  # height and width of the input tensor
        backbone_block = self.backbone(x)
        aspp_block = self.aspp(backbone_block)
        final_block = self.final_layer(aspp_block)
        result = nn.functional.interpolate(
            final_block, size=input_shape, mode="bilinear", align_corners=False
        )  # upsampling
        return result


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) block that performs parallel atrous convolutions and pooling
    on the input tensor and concatenates the resulting features.
    Args:
        in_channels (int): Number of channels in the input tensor.
        atrous_rates (list of int): A list of atrous rates for the atrous convolutions.
        out_channels (int): Number of output channels for the final 1x1 convolution. Default is 256.
    """
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        self.convolution_branches = nn.ModuleList()

        # Add a 1x1 convolution
        self.convolution_branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        # Add atrous convolutions
        for rate in atrous_rates:
            self.convolution_branches.append(self.aspp_conv(in_channels, out_channels, rate))

        # Add pooling
        self.convolution_branches.append(self.aspp_pooling(in_channels, out_channels))

        # Final 1x1 convolution to get out_channels channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(len(self.convolution_branches) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        """
        Forward pass of the ASPP block
        Args:
            x: Tensor
        """
        # Compute the features for all branches
        branch_features = []
        for branch in self.convolution_branches[:-1]:
            branch_features.append(branch(x))

        # Upsample the features obtained from the pooling branch
        input_size = x.shape[-2:]
        pooling_features = self.convolution_branches[-1](x)
        upsampled_pooling_features = nn.functional.interpolate(
            pooling_features, size=input_size, mode="bilinear", align_corners=False
        )
        branch_features.append(upsampled_pooling_features)

        # Concatenate the features obtained from all the branches
        concatenated_features = torch.cat(branch_features, dim=1)

        # Final 1x1 convolution to get out_channels channels
        return self.final_conv(concatenated_features)

    def aspp_conv(self, in_channels, out_channels, dilation):
        """
        Applies a 3x3 convolution with a specified dilation rate.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def aspp_pooling(self, in_channels, out_channels):
        """
        Applies a global pooling operation followed by a 1x1 convolution.
        """
        block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

'''
Fin de votre code.
'''