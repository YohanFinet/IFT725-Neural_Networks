# -*- coding:utf-8 -*- 

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from model.CNNBaseModel import CNNBaseModel
from layers.CNNBlocks import ResidualBlock, DenseBlock, BottleneckBlock

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourNet.  Le réseau est constitué de

    1) quelques blocs d'opérations de base du type «conv-batch-norm-relu»
    2) 1 (ou plus) bloc dense inspiré du modèle «denseNet)
    3) 1 (ou plus) bloc résiduel inspiré de «resNet»
    4) 1 (ou plus) bloc de couches «bottleneck» avec ou sans connexion résiduelle, c’est au choix
    5) 1 (ou plus) couches pleinement connectées 
    
    NOTE : le code des blocks résiduel, dense et bottleneck doivent être mis dans le fichier CNNBlocks.py
    Aussi, vous pouvez vous inspirer du code de CNNVanilla.py pour celui de *YourNet*

'''


class YourNet(CNNBaseModel):

    def __init__(self, num_classes=10, init_weights=True):
        super(YourNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Nx64x16x16

            ResidualBlock(64, 64, BottleneckBlock(64, 32)),# Nx64x16x16
            DenseBlock(64, BottleneckBlock(64, 32)),# Nx128x16x16
            BottleneckBlock(128, 64),# Nx128x16x16
            ResidualBlock(128, 256, BottleneckBlock(256, 128)),# Nx256x8x8
            nn.MaxPool2d(kernel_size=2, stride=2) #Nx256x4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # reshape feature maps
        x = self.fc_layers(x)
        return x
'''
FIN DE VOTRE CODE
'''
