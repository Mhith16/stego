# Define the EXACT architecture used during training
# These should match the trained weights EXACTLY
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import necessary components from your existing modules
from models.decoder import EnhancedErrorCorrectionModule
from models.feature_analyzer import DenseBlock, TransitionLayer, ChannelAttention, SpatialAttention
from models.encoder import ResidualBlock

class ExactFeatureAnalyzer(nn.Module):
    def __init__(self, in_channels=1):
        super(ExactFeatureAnalyzer, self).__init__()
        
        # Exact architecture from trained model with 1024 final features
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks and transitions to match the exact shape in the saved model
        self.blocks = nn.ModuleList([
            DenseBlock(64, 16, 3),      # Match block configuration
            DenseBlock(112, 16, 6),     # Numbers adjusted to match saved weights
            DenseBlock(208, 16, 12),    
            DenseBlock(400, 16, 24)     # This produces exactly 1024 final features
        ])
        
        self.transitions = nn.ModuleList([
            TransitionLayer(112, 112),
            TransitionLayer(208, 208),
            TransitionLayer(400, 400)
        ])
        
        # Final layers with EXACT 1024 features
        self.final_norm = nn.BatchNorm2d(1024)  # Exact 1024 features to match saved model
        self.final_conv = nn.Conv2d(1024, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
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


class ExactEnhancedEncoder(nn.Module):
    """Encoder with EXACT architecture matching saved weights"""
    def __init__(self, image_channels=1, message_length=256):
        super(ExactEnhancedEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Message preparation network
        self.prep_msg = nn.Sequential(
            nn.Linear(message_length, message_length * 2),
            nn.ReLU(inplace=True),
            nn.Linear(message_length * 2, message_length * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Message embedding layer
        self.embed_msg = nn.Linear(message_length * 4, 64*64)
        
        # Residual blocks for feature processing
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64 + 1) for _ in range(6)
        ])
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.BatchNorm2d(64 + 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, image_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Adaptive embedding strength - exact value to match saved model
        self.embed_strength = nn.Parameter(torch.tensor(0.12))
        self.perceptual_scale = nn.Parameter(torch.tensor(1.5))
        
    def forward(self, image, message, feature_weights=None):
        # Initial image processing
        x = F.relu(self.bn1(self.conv1(image)))
        
        # Process the message
        msg = self.prep_msg(message)
        
        # Embed message into spatial feature map
        msg_spatial = self.embed_msg(msg)
        msg_spatial = msg_spatial.view(-1, 1, 64, 64)
        
        # Resize to match feature map dimensions
        msg_spatial = F.interpolate(msg_spatial, size=(x.size(2), x.size(3)), 
                                   mode='bilinear', align_corners=False)
        
        # Apply feature weights if provided with perceptual scaling
        if feature_weights is not None:
            if feature_weights.size()[2:] != x.size()[2:]:
                feature_weights = F.interpolate(feature_weights, size=(x.size(2), x.size(3)), 
                                              mode='bilinear', align_corners=False)
            perceptual_mask = torch.pow(1 - feature_weights, self.perceptual_scale)
            msg_spatial = msg_spatial * perceptual_mask
        
        # Concatenate message with image features
        x = torch.cat([x, msg_spatial], dim=1)
        
        # Process through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final processing
        modifications = self.final_layers(x)
        
        # Apply adaptive embedding strength
        effective_strength = torch.clamp(torch.abs(self.embed_strength), min=0.08, max=0.25)
        
        # Use feature weights to control embedding strength in different regions
        if feature_weights is not None:
            inv_weights = 1.0 - feature_weights
            embedding_scale = 0.5 + inv_weights
            effective_strength = effective_strength * embedding_scale
        
        stego_image = image + modifications * effective_strength
        stego_image = torch.clamp(stego_image, 0, 1)
        
        return stego_image


class ExactEnhancedDecoder(nn.Module):
    """Decoder with EXACT architecture matching saved weights"""
    def __init__(self, image_channels=1, message_length=256):
        super(ExactEnhancedDecoder, self).__init__()
        
        # Initial layers
        self.initial_layers = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(128)
        self.spatial_attention = SpatialAttention()
        
        # Residual blocks for feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(8)
        ])
        
        # Global average pooling and fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten_size = 128 * 8 * 8
        
        # Fully connected layers for message extraction
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, message_length),
        )
        
        # Final sigmoid for binary output
        self.sigmoid = nn.Sigmoid()
        
        # Error correction module
        self.error_correction = EnhancedErrorCorrectionModule(message_length)
        
    def forward(self, stego_image):
        # Extract features
        x = self.initial_layers(stego_image)
        
        # Apply attention
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        
        # Process through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Decode message
        message_logits = self.fc_layers(x)
        
        # Apply sigmoid to get probabilities
        message_probs = self.sigmoid(message_logits)
        
        # Apply error correction
        corrected_message = self.error_correction(message_probs)
        
        return corrected_message