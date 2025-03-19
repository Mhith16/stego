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


class ErrorCorrectionModule(nn.Module):
    def __init__(self, message_length):
        super(ErrorCorrectionModule, self).__init__()
        
        # A simple learned error correction module
        hidden_dim = min(message_length * 2, 1024)  # Cap hidden dimension
        
        self.correction = nn.Sequential(
            nn.Linear(message_length, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, message_length),
            nn.Sigmoid()
        )
        
    def forward(self, message_probs):
        # Apply learned correction
        corrected = self.correction(message_probs)
        
        # Hard decision threshold for binary values when in evaluation mode
        if not self.training:
            hard_decisions = (corrected > 0.5).float()
            return hard_decisions
        
        return corrected


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
        
        # Confidence estimator - predicts how confident we are in each bit
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
            # High confidence bits use threshold closer to 0.5
            # Low confidence bits are biased toward their probable value
            base_threshold = 0.5
            adaptive_offset = (confidence - 0.5) * 0.2  # Scale to ±0.1
            
            # Apply different thresholds for each bit based on confidence
            thresholds = base_threshold + adaptive_offset
            
            # Make hard decisions
            hard_decisions = (corrected > thresholds).float()
            return hard_decisions
        
        return corrected


class EnhancedSteganographyDecoder(nn.Module):
    def __init__(self, image_channels=1, message_length=256):
        super(EnhancedSteganographyDecoder, self).__init__()
        
        # Initial layers
        self.initial_layers = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # Reduce spatial dimension
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(128)
        self.spatial_attention = SpatialAttention()
        
        # Residual blocks for feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(8)  # Increased from 6 to 8 for better extraction
        ])
        
        # Global average pooling and fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten_size = 128 * 8 * 8
        
        # Fully connected layers for message extraction with more capacity
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


# Original implementation (for compatibility with older code)
class SteganographyDecoder(nn.Module):
    def __init__(self, image_channels=1, growth_rate=16, num_layers=4, message_length=256):
        super(SteganographyDecoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, padding=1)
        
        # Dense block for feature extraction
        self.dense_block = DenseBlock(32, growth_rate, num_layers)
        
        # Transition layer to reduce spatial dimensions
        self.transition = nn.Sequential(
            nn.BatchNorm2d(self.dense_block.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dense_block.out_channels, 64, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Global average pooling and fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten_size = 64 * 8 * 8
        
        # Fully connected layers to decode the message
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, message_length),
            nn.Sigmoid()  # Binary output
        )
        
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
                nn.init.constant_(m.bias, 0)
    
    def forward(self, stego_image):
        # Extract features
        x = self.conv1(stego_image)
        x = self.dense_block(x)
        x = self.transition(x)
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Decode message
        message = self.fc(x)
        
        return message


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # Simplified architecture to avoid dimension issues
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
        
        self.out_channels = current_channels
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x