import torch
import torch.nn as nn
import torch.nn.functional as F

# Import required components from your existing files
# If you don't have these classes defined, you'll need to include their definitions here

# DenseBlock and TransitionLayer from feature_analyzer.py
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

# Define attention mechanisms
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

# Define ResidualBlock for the encoder
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

# Define EnhancedErrorCorrectionModule for the decoder
class EnhancedErrorCorrectionModule(nn.Module):
    def __init__(self, message_length):
        super(EnhancedErrorCorrectionModule, self).__init__()
        
        # Deeper network for better error correction
        hidden_dim1 = min(message_length * 2, 1024)
        hidden_dim2 = min(message_length * 4, 2048)
        
        self.correction_network = nn.Sequential(
            nn.Linear(message_length, hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim1, message_length),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(message_length, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, message_length),
            nn.Sigmoid()
        )
        
    def forward(self, message_probs):
        # Apply learned correction
        corrected = self.correction_network(message_probs)
        
        # Hard decision threshold for binary values when in evaluation mode
        if not self.training:
            # Get confidence estimates
            confidence = self.confidence_estimator(message_probs)
            
            # Adaptive thresholding based on confidence
            base_threshold = 0.5
            adaptive_offset = (confidence - 0.5) * 0.2
            
            # Apply different thresholds for each bit based on confidence
            thresholds = base_threshold + adaptive_offset
            
            # Make hard decisions
            hard_decisions = (corrected > thresholds).float()
            return hard_decisions
        
        return corrected

# Feature Analyzer with 784 features
class ExactFeatureAnalyzer(nn.Module):
    def __init__(self, in_channels=1):
        super(ExactFeatureAnalyzer, self).__init__()
        
        # Calculate the exact feature counts based on your network
        num_init_features = 64
        growth_rate = 16
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Initialize with fewer features to match your expected output
        # Block configuration creates a total of 784 features
        num_features = num_init_features  # Start with 64
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        # Block 1: 64 -> 64 + (3 * 16) = 112 features
        self.blocks.append(DenseBlock(num_features, growth_rate, 3))
        num_features += 3 * growth_rate  # 112
        self.transitions.append(TransitionLayer(num_features, num_features))
        
        # Block 2: 112 -> 112 + (6 * 16) = 208 features
        self.blocks.append(DenseBlock(num_features, growth_rate, 6))
        num_features += 6 * growth_rate  # 208
        self.transitions.append(TransitionLayer(num_features, num_features))
        
        # Block 3: 208 -> 208 + (12 * 16) = 400 features
        self.blocks.append(DenseBlock(num_features, growth_rate, 12))
        num_features += 12 * growth_rate  # 400
        self.transitions.append(TransitionLayer(num_features, num_features))
        
        # Block 4: 400 -> 400 + (24 * 16) = 784 features
        self.blocks.append(DenseBlock(num_features, growth_rate, 24))
        num_features += 24 * growth_rate  # 784
        
        # Use the actual number of features (784) instead of hardcoded 1024
        self.final_norm = nn.BatchNorm2d(num_features)  # 784 features
        self.final_conv = nn.Conv2d(num_features, 1, kernel_size=1, bias=False)
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

# Encoder with exact architecture
class ExactEnhancedEncoder(nn.Module):
    def __init__(self, image_channels=1, message_length=512):
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

# Decoder with exact architecture
class ExactEnhancedDecoder(nn.Module):
    def __init__(self, image_channels=1, message_length=512):
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