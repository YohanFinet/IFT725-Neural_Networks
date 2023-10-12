import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from torchvision import datasets
import torchvision.transforms as transforms
from manage.HDF5Dataset import HDF5Dataset
import matplotlib.pyplot as plt
import numpy as np
import random



acdc_hdf5_file = '../../data/ift780_acdc.hdf5'

acdc_augment_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    ]
)

acdc_base_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set_1 = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform)
train_set_2 = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform)

index_image = random.randrange(len(train_set_1))

array_1 = np.transpose(train_set_1[0][0], (1, 2, 0)).squeeze()
array_2 = acdc_augment_transform(array_1.numpy())
array_2 = np.transpose(array_2, (1, 2, 0)).squeeze()

f = plt.figure(figsize=(10, 10))
ax1 = f.add_subplot(221)
ax1.imshow(array_1)
ax1.set_title('base image')
ax1.axis('off')

ax1 = f.add_subplot(222)
ax1.imshow(array_2)
ax1.set_title('modified image')
ax1.axis('off')

plt.show()