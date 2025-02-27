import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import logging

logger = logging.getLogger(__name__)

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(ResNetBackbone, self).__init__()
        
        try:
        
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            resnet = resnet34(weights=weights)
            
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            
            if pretrained:
                for param in list(self.features.parameters())[:-3]:
                    param.requires_grad = False
            
            logger.info("ResNet backbone initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ResNet backbone: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.features(x)
        except Exception as e:
            logger.error(f"Error in backbone forward pass: {e}")
            raise

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info("All backbone layers unfrozen")