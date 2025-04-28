import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

def get_model(num_classes, pretrained_backbone=False):
    model = deeplabv3_resnet50(
        weights=None,
        aux_loss=None,
        num_classes=num_classes
    )
    return model
