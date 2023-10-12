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
import torch.nn.functional as F


class SimpleCNNBlock(nn.Module):
    """
    this block is an example of a simple conv-relu-conv-relu block
    with 3x3 convolutions
    """

    def __init__(self, in_channels):
        super(SimpleCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        return output


""" 
TODO

Suivant l'example ci-haut, vous devez rédiger les classes permettant de créer des :

1- Un bloc résiduel
2- Un bloc dense
3- Un bloc Bottleneck

Ces blocks seront utilisés dans le fichier YouNET.py
"""


class ResidualBlock(nn.Module):
    """
    this block is the building block of the residual network. it takes an 
    input with in_channels, applies some blocks of convolutional layers
    to reduce it to out_channels and sum it up to the original input,
    """

    def __init__(self, in_channels, out_channels, middle_block):
        super(ResidualBlock, self).__init__()
        self.middle_block = middle_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
            self.conv_x = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0)
            
        
    def forward(self, x):
        output = None
        if self.in_channels != self.out_channels:
            output = self.conv_out(x)
            output = self.middle_block(output)
            output += self.conv_x(x)
        else:
            output = self.middle_block(x)
            output += x
        return F.relu(output)


class DenseBlock(nn.Module):
    """
    This block is the building block of the Dense network. It takes an
    input with in_channels, applies some blocks of convolutional, batchnorm layers
    and then concatenate the output with the original input
    """

    def __init__(self, in_channels, middle_block):
        super(DenseBlock, self).__init__()
        self.middle_block = middle_block
        
    def forward(self, x):
        output = self.middle_block(x)
        output = torch.cat((x, output), 1)
        return output


class BottleneckBlock(nn.Module):
    """
    This block takes an input with in_channels reduces number of channels by a certain
    parameter "downsample" through kernels of size 1x1, 3x3, 1x1 respectively.
    """

    def __init__(self, in_channels, downsample):
        super(BottleneckBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=downsample, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(downsample),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=downsample, out_channels=downsample, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(downsample),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=downsample, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        output = self.conv_layers(x)
        return output
        
