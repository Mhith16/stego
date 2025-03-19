import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class EnhancedDiscriminator(nn.Module):
    def __init__(self, image_channels=1, use_logits=True):
        super(EnhancedDiscriminator, self).__init__()
        
        # Flag to indicate whether to return logits or probabilities
        self.use_logits = use_logits
        
        # Use spectral normalization for stability
        self.model = nn.Sequential(
            # First layer
            spectral_norm(nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second layer
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third layer
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth layer
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Classification layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(128, 1))
        )
        
        # Sigmoid activation (only used if use_logits=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.model(x)
        features = self.avg_pool(features)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        
        if self.use_logits:
            # Return raw logits for BCEWithLogitsLoss
            return logits
        else:
            # Return probabilities for regular BCE
            return self.sigmoid(logits)


# Original implementation (for compatibility with older code)
class Discriminator(nn.Module):
    def __init__(self, image_channels=1):
        super(Discriminator, self).__init__()
        
        # Simpler CNN-based discriminator
        self.model = nn.Sequential(
            # First layer
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second layer
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third layer
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth layer
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate the size after convolutions
        # Assuming 256x256 input, the output of the last conv layer is 16x16
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.model(x)
        features = self.avg_pool(features)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output