import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.ops import roi_pool
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class RCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(RCNN, self).__init__()
        
        logger.info("Initializing RCNN model...")
        
        weights = ResNet34_Weights.DEFAULT
        resnet = resnet34(weights=weights)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        for param in list(self.backbone.parameters())[:-3]:
            param.requires_grad = False
        
        self.roi_pool_size = (7, 7)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()
        logger.info("Model initialized successfully")

    def forward(self, x: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, None]:
        try:
            batch_size = x.size(0)
            
            features = self.backbone(x)
            
            if boxes is None:
                boxes = torch.tensor([[0, 0, 223, 223]], device=x.device).float()
                boxes = boxes.repeat(batch_size, 1, 1)
            
            rois = []
            for i in range(batch_size):
                batch_boxes = boxes[i]
                batch_rois = torch.cat([
                    torch.full((batch_boxes.size(0), 1), i, device=x.device),
                    batch_boxes
                ], dim=1)
                rois.append(batch_rois)
            rois = torch.cat(rois, dim=0)
            
            roi_features = roi_pool(features, rois, self.roi_pool_size)
            
            roi_features = roi_features.view(roi_features.size(0), -1)
            
            class_scores = self.classifier(roi_features)
            
            return class_scores, None
            
        except Exception as e:
            logger.error(f"Error in RCNN forward pass: {e}")
            raise

    def _initialize_weights(self):
        try:
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
            logger.info("Weights initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing weights: {e}")
            raise

    def save_model(self, path: str):
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str):
        try:
            self.load_state_dict(torch.load(path))
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise