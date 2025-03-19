import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class EnhancedSteganographyEncoder(nn.Module):
    def __init__(self, image_channels=1, message_length=256):
        super(EnhancedSteganographyEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Message preparation network
        self.prep_msg = nn.Sequential(
            nn.Linear(message_length, message_length * 2),
            nn.ReLU(inplace=True),
            nn.Linear(message_length * 2, message_length * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)  # For regularization
        )
        
        # Message embedding layer
        self.embed_msg = nn.Linear(message_length * 4, 64*64)
        
        # Residual blocks for feature processing
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64 + 1) for _ in range(6)  # Increased from 4 to 6 for better quality
        ])
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.BatchNorm2d(64 + 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, image_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1,1] for fine-grained control
        )
        
        # Adaptive embedding strength - learnable parameter with higher initial value
        self.embed_strength = nn.Parameter(torch.tensor(0.12))
        
        # Perceptual scaling factor - controls embedding based on feature importance
        self.perceptual_scale = nn.Parameter(torch.tensor(1.5))
        
    def forward(self, image, message, feature_weights=None):
        # Initial image processing
        x = F.relu(self.bn1(self.conv1(image)))  # [B, 64, H, W]
        
        # Process the message
        msg = self.prep_msg(message)  # [B, message_length*4]
        
        # Embed message into spatial feature map
        msg_spatial = self.embed_msg(msg)  # [B, 64*64]
        msg_spatial = msg_spatial.view(-1, 1, 64, 64)  # [B, 1, 64, 64]
        
        # Resize to match feature map dimensions
        msg_spatial = F.interpolate(msg_spatial, size=(x.size(2), x.size(3)), 
                                   mode='bilinear', align_corners=False)
        
        # Apply feature weights if provided with perceptual scaling
        if feature_weights is not None:
            # Ensure feature_weights has the right dimensions
            if feature_weights.size()[2:] != x.size()[2:]:
                feature_weights = F.interpolate(feature_weights, size=(x.size(2), x.size(3)), 
                                              mode='bilinear', align_corners=False)
            
            # Apply perceptual scaling: embed more in less important areas
            # (1 - feature_weights) gives higher values in less important areas
            perceptual_mask = torch.pow(1 - feature_weights, self.perceptual_scale)
            
            # Weight the message embedding
            msg_spatial = msg_spatial * perceptual_mask
        
        # Concatenate message with image features
        x = torch.cat([x, msg_spatial], dim=1)  # [B, 65, H, W]
        
        # Process through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final processing
        modifications = self.final_layers(x)
        
        # Apply adaptive embedding strength with minimum threshold for reliability
        effective_strength = torch.clamp(torch.abs(self.embed_strength), min=0.08, max=0.25)
        
        # Use feature weights to control embedding strength in different regions
        if feature_weights is not None:
            # Invert the weights: embed more strongly in less important areas
            inv_weights = 1.0 - feature_weights
            # Scale to [0.5, 1.5] range to avoid extreme variations
            embedding_scale = 0.5 + inv_weights
            effective_strength = effective_strength * embedding_scale
        
        stego_image = image + modifications * effective_strength
        
        # Ensure output values are in valid range [0, 1]
        stego_image = torch.clamp(stego_image, 0, 1)
        
        return stego_image


# Original implementation (for compatibility with older code)
class SteganographyEncoder(nn.Module):
    def __init__(self, image_channels=1, growth_rate=16, num_dense_layers=4):
        super(SteganographyEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, padding=1)
        
        # Process the message
        self.prep_msg = nn.Sequential(
            nn.Linear(256, 256),  # Default message length of 256
            nn.ReLU(inplace=True)
        )
        
        # Message embedding layer 
        self.embed_msg = nn.Linear(256, 64*64)  # Reshape to spatial dimension
        
        # Dense block for feature extraction
        self.dense_layers = nn.ModuleList()
        current_channels = 32 + 1  # Initial channels + message channel
        for i in range(num_dense_layers):
            self.dense_layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        
        # Final normalization and convolution
        self.final_norm = nn.BatchNorm2d(current_channels)
        self.final_conv = nn.Conv2d(current_channels, image_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, image, message, feature_weights=None):
        # Initial image processing
        x = self.conv1(image)  # [B, 32, H, W]
        
        # Process the message
        msg = self.prep_msg(message)  # [B, 256]
        
        # Embed message into spatial feature map
        msg_spatial = self.embed_msg(msg)  # [B, 64*64]
        msg_spatial = msg_spatial.view(-1, 1, 64, 64)  # [B, 1, 64, 64]
        
        # Resize to match feature map dimensions
        msg_spatial = F.interpolate(msg_spatial, size=(x.size(2), x.size(3)), 
                                   mode='bilinear', align_corners=False)
        
        # Apply feature weights if provided
        if feature_weights is not None:
            # Ensure feature_weights has the right dimensions
            if feature_weights.size()[2:] != x.size()[2:]:
                feature_weights = F.interpolate(feature_weights, size=(x.size(2), x.size(3)), 
                                              mode='bilinear', align_corners=False)
            # Weight the message embedding
            msg_spatial = msg_spatial * feature_weights
        
        # Concatenate message with image features
        x = torch.cat([x, msg_spatial], dim=1)  # [B, 33, H, W]
        
        # Process through dense layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        
        # Final processing
        x = F.relu(self.final_norm(x))
        x = self.final_conv(x)
        
        # Add residual connection for better image quality
        stego_image = image + torch.tanh(x) * 0.1  # Scale down the modifications
        
        # Ensure output values are in valid range [0, 1]
        stego_image = torch.clamp(stego_image, 0, 1)
        
        return stego_image


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv(F.relu(self.norm(x)))
        return torch.cat([x, out], 1)