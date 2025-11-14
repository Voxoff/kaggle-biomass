import torch
import torch.nn as nn
import timm

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=5)
        # MLP regression head
        in_features = self.backbone.num_features
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5)
        )
    
    def forward(self, x):
        return self.backbone(x)

