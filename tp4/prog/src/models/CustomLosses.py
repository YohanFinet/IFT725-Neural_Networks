import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import mean_dice
from torchmetrics.classification import MulticlassJaccardIndex

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        dice = mean_dice(input, target)
        return 1 - dice


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        labels = torch.unique(target[target.nonzero(as_tuple=True)])  # Identify classes (that are not background)

        # Compute the jaccard index for each individual class
        jaccard_indices = torch.stack([self.jaccard(input, target, label) for label in labels])

        mean_jaccard = torch.mean(jaccard_indices, dim=0)  # Compute the mean jaccard over all classes

        loss = 1 - mean_jaccard

        return loss

    def jaccard(self, input, target, label):
        reduce_axis = (0, 1)

        input = F.softmax(input, dim=1)[:, label,
                ...]  # For the input, extract the probabilities of the requested label
        target = torch.eq(target, label)  # For the target, extract the boolean mask of the requested label

        # Flatten the tensors to facilitate broadcasting
        input = torch.flatten(input, start_dim=1)
        target = torch.flatten(target, start_dim=1)

        # Compute jaccard index
        intersect = input * target
        intersection = torch.sum(intersect, 1, keepdim=True)
        union = torch.sum(input + target - intersect, 1, keepdim=True)
        jaccard = torch.mean((intersect + 1) / (union + 1), dim=reduce_axis)

        return jaccard


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = nn.MultiMarginLoss()
        return loss(input.permute(0, 2, 3, 1).reshape(-1, 4), target.reshape(-1))