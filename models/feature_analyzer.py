import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv(F.relu(self.norm(x)))
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        return self.pool(self.conv(F.relu(self.norm(x))))


class MedicalFeatureAnalyzer(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super(MedicalFeatureAnalyzer, self).__init__()
        
        # Use pretrained DenseNet as backbone
        densenet = models.densenet121(pretrained=pretrained)
        
        # Extract DenseNet features and modify for grayscale input
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Feature extraction layers
        self.features = nn.Sequential(
            self.initial_conv,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DenseBlock(64, 32, 6),
            TransitionLayer(64 + 6 * 32, 128),
            DenseBlock(128, 32, 12),
            TransitionLayer(128 + 12 * 32, 256),
            DenseBlock(256, 32, 24),
            TransitionLayer(256 + 24 * 32, 512),
            DenseBlock(512, 32, 16)
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(512 + 16 * 32)
        self.spatial_attention = SpatialAttention()
        
        # Final layers
        self.final_norm = nn.BatchNorm2d(512 + 16 * 32)
        self.final_conv = nn.Conv2d(512 + 16 * 32, 1, kernel_size=1, bias=False)
        
        # Sigmoid activation for [0,1] output
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply attention
        features = self.channel_attention(features) * features
        features = self.spatial_attention(features) * features
        
        # Final processing
        features = F.relu(self.final_norm(features))
        features = self.final_conv(features)
        
        # Upscale to match input image size
        if features.size()[2:] != x.size()[2:]:
            features = F.interpolate(features, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # Apply sigmoid to get values in [0,1] range
        features = self.sigmoid(features)
        
        return features


# Simplified version for quicker training
class SimpleFeatureAnalyzer(nn.Module):
    def __init__(self, in_channels=1):
        super(SimpleFeatureAnalyzer, self).__init__()
        
        # Simplified architecture
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Attention
        self.channel_attention = ChannelAttention(128)
        self.spatial_attention = SpatialAttention()
        
        # Final layer
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.features(x)
        
        # Apply attention
        features = self.channel_attention(features) * features
        features = self.spatial_attention(features) * features
        
        # Get feature weights
        features = self.final_conv(features)
        features = self.sigmoid(features)
        
        # Upscale to match input size
        if features.size()[2:] != x.size()[2:]:
            features = F.interpolate(features, size=x.size()[2:], mode='bilinear', align_corners=False)
            
        return features


# Original implementation (for compatibility with older code)
class FeatureAnalysisDenseNet(nn.Module):
    def __init__(self, in_channels=1, growth_rate=16, block_config=(3, 6, 12, 8), num_init_features=32):
        super(FeatureAnalysisDenseNet, self).__init__()
        
        # Initial convolution - now using the parameter instead of hardcoded value
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Downsampling path (encoder)
        num_features = num_init_features
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.blocks.append(block)
            num_features += num_layers * growth_rate
            
            # Add transition layer except after the last block
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.transitions.append(trans)
                num_features = num_features // 2
        
        # Final normalization and convolution
        self.final_norm = nn.BatchNorm2d(num_features)
        self.final_conv = nn.Conv2d(num_features, 1, kernel_size=1, bias=False)
        
        # Sigmoid activation for [0,1] output
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial features
        features = self.features(x)
        
        # Pass through dense blocks and transitions
        for i, (block, transition) in enumerate(zip(self.blocks[:-1], self.transitions)):
            features = block(features)
            features = transition(features)
        
        # Last dense block (no transition after it)
        features = self.blocks[-1](features)
        
        # Final processing
        features = F.relu(self.final_norm(features))
        features = self.final_conv(features)
        
        # Upscale to match input image size
        if features.size()[2:] != x.size()[2:]:
            features = F.interpolate(features, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # Apply sigmoid to get values in [0,1] range
        features = self.sigmoid(features)
        
        return features