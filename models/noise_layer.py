import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnhancedNoiseLayer(nn.Module):
    """
    Enhanced noise layer for medical image steganography that simulates
    real-world transformations and distortions specific to medical imaging.
    """
    def __init__(self, noise_types=None):
        super(EnhancedNoiseLayer, self).__init__()
        self.noise_types = noise_types or [
            'dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper',
            'contrast', 'gamma', 'intensity_shift', 'windowing', 'zoom'
        ]
        
        # Pre-compute DCT basis (for JPEG simulation)
        self.register_buffer('dct_basis', None)
        self.register_buffer('idct_basis', None)
        self.register_buffer('quant_matrix', None)
        
        # Initialize convolution kernels for blur operations
        self.blur_kernels = {}
        
    def _init_dct_basis(self, device):
        """Initialize DCT basis matrices for JPEG simulation"""
        dct_basis = torch.zeros(64, 64, device=device)
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    for l in range(8):
                        if i == 0:
                            alpha_i = 1.0 / np.sqrt(8)
                        else:
                            alpha_i = np.sqrt(2) / np.sqrt(8)
                            
                        if j == 0:
                            alpha_j = 1.0 / np.sqrt(8)
                        else:
                            alpha_j = np.sqrt(2) / np.sqrt(8)
                            
                        basis_index = i * 8 + j
                        pixel_index = k * 8 + l
                        dct_basis[basis_index, pixel_index] = alpha_i * alpha_j * np.cos((2*k+1)*i*np.pi/16) * np.cos((2*l+1)*j*np.pi/16)
        
        idct_basis = dct_basis.transpose(0, 1)
        
        # JPEG quantization matrix (luminance)
        quant_matrix = torch.tensor([
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
        ], device=device).reshape(8, 8).to(device)
        
        self.register_buffer('dct_basis', dct_basis)
        self.register_buffer('idct_basis', idct_basis)
        self.register_buffer('quant_matrix', quant_matrix)
    
    def forward(self, x, noise_type=None, noise_intensity=1.0):
        """
        Apply noise or transformation to the input images
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            noise_type: Type of noise to apply, if None, randomly choose one
            noise_intensity: Control the strength of the applied noise (0.0 to 1.0)
            
        Returns:
            Noisy/transformed tensor of the same shape
        """

        x = torch.clamp(x, 0, 1)

        # If no specific noise type is specified, randomly choose one
        if noise_type is None:
            noise_type = np.random.choice(self.noise_types)
        
        # Apply the selected noise with controlled intensity
        if noise_type == 'dropout':
            result = self.dropout(x, dropout_prob=0.5 * noise_intensity)
        elif noise_type == 'jpeg':
            result = self.jpeg_compression(x, quality_factor=50 + 50 * (1 - noise_intensity))
        elif noise_type == 'gaussian':
            result = self.gaussian_noise(x, std=0.05 * noise_intensity)
        elif noise_type == 'blur':
            result = self.gaussian_blur(x, kernel_size=3, sigma=1.0 * noise_intensity)
        elif noise_type == 'salt_pepper':
            result = self.salt_pepper_noise(x, density=0.1 * noise_intensity)
        elif noise_type == 'contrast':
            result = self.contrast_adjustment(x, factor=1.0 + noise_intensity)
        elif noise_type == 'gamma':
            result = self.gamma_correction(x, gamma=0.8 + 0.4 * noise_intensity)
        elif noise_type == 'intensity_shift':
            result = self.intensity_shift(x, shift=0.1 * noise_intensity)
        elif noise_type == 'windowing':
            result = self.windowing(x, center=0.5, width=0.5, intensity=noise_intensity)
        elif noise_type == 'zoom':
            result = self.zoom(x, factor=1.0 + 0.2 * noise_intensity)
        else:
            result = x  # No noise
        
        return torch.clamp(result, 0, 1)

    
    def dropout(self, x, dropout_prob=0.5):
        """Random pixel dropout"""
        mask = torch.rand_like(x) > dropout_prob
        return x * mask
    
    def jpeg_compression(self, x, quality_factor=50):
        """
        Simulate JPEG compression with DCT and quantization
        
        Args:
            x: Input tensor [B, C, H, W]
            quality_factor: JPEG quality (0-100, higher is better)
            
        Returns:
            Compressed tensor of the same shape
        """
        # Initialize DCT basis if not already done
        if self.dct_basis is None:
            self._init_dct_basis(x.device)
            
        batch_size, channels, height, width = x.shape
        
        # Pad if necessary to make dimensions divisible by 8
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0)
            
        # Get padded dimensions
        _, _, padded_h, padded_w = x.shape
        
        # Scale quality factor to get quantization scaling
        scale = 1.0
        if quality_factor < 50:
            scale = 5000 / quality_factor
        else:
            scale = 200 - 2 * quality_factor
            
        scale = scale / 100.0
        
        # Prepare quantization matrix
        quant = self.quant_matrix.view(-1) * scale
        quant = torch.clamp(quant, 1, 255)
        
        # Process each channel separately
        result = []
        for ch in range(channels):
            channel = x[:, ch:ch+1, :, :]  # Keep dimension
            
            # Extract 8x8 blocks using unfold
            blocks = F.unfold(channel, kernel_size=8, stride=8)
            blocks = blocks.view(batch_size, 64, -1)
            
            # Apply DCT to each block
            dct_coef = torch.matmul(self.dct_basis.to(x.device), blocks)
            
            # Quantize DCT coefficients
            quant_coef = torch.round(dct_coef / quant.view(-1, 1))
            
            # Dequantize
            dequant_coef = quant_coef * quant.view(-1, 1)
            
            # Apply inverse DCT
            idct_blocks = torch.matmul(self.idct_basis.to(x.device), dequant_coef)
            
            # Reshape back to image
            idct_blocks = idct_blocks.view(batch_size, 64, -1)
            recon_channel = F.fold(
                idct_blocks, 
                output_size=(padded_h, padded_w),
                kernel_size=8, 
                stride=8
            )
            
            # Remove padding if added
            if pad_h > 0 or pad_w > 0:
                recon_channel = recon_channel[:, :, :height, :width]
                
            result.append(recon_channel)
            
        # Combine channels
        recon_image = torch.cat(result, dim=1)
        
        # Clamp values to [0, 1]
        return torch.clamp(recon_image, 0, 1)
    
    def gaussian_noise(self, x, std=0.05):
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * std
        noisy = x + noise
        return torch.clamp(noisy, 0, 1)
    
    def gaussian_blur(self, x, kernel_size=3, sigma=1.0):
        """
        Apply Gaussian blur using separable convolution 
        for better performance
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Check if we already have this kernel
        key = f"blur_{kernel_size}_{sigma:.2f}"
        if key not in self.blur_kernels:
            # Create 1D Gaussian kernel
            half_size = kernel_size // 2
            one_dim_kernel = torch.exp(-torch.arange(-half_size, half_size+1, dtype=torch.float).pow(2) / (2 * sigma ** 2))
            one_dim_kernel = one_dim_kernel / one_dim_kernel.sum()
            
            # Create horizontal and vertical kernels for separable convolution
            h_kernel = one_dim_kernel.view(1, 1, 1, kernel_size).repeat(x.size(1), 1, 1, 1)
            v_kernel = one_dim_kernel.view(1, 1, kernel_size, 1).repeat(x.size(1), 1, 1, 1)
            
            self.blur_kernels[key] = (h_kernel, v_kernel)
        
        h_kernel, v_kernel = self.blur_kernels[key]
        
        # Move kernels to the right device
        h_kernel = h_kernel.to(x.device)
        v_kernel = v_kernel.to(x.device)
        
        # Apply padding
        padding = kernel_size // 2
        padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        # Apply horizontal blur (groups=channels for depthwise convolution)
        temp = F.conv2d(padded, h_kernel, groups=x.size(1))
        
        # Apply vertical blur
        blurred = F.conv2d(temp, v_kernel, groups=x.size(1))
        
        return blurred
    
    def salt_pepper_noise(self, x, density=0.1):
        """Add salt and pepper noise"""
        noise = torch.rand_like(x)
        
        # Salt (white) noise
        salt = (noise < density/2).float()
        
        # Pepper (black) noise
        pepper = (noise > 1 - density/2).float()
        
        # Apply salt and pepper noise
        noisy = x.clone()
        noisy[salt > 0] = 1.0
        noisy[pepper > 0] = 0.0
        
        return noisy
    
    def contrast_adjustment(self, x, factor=1.5):
        """
        Adjust image contrast - medical images often undergo
        contrast adjustments during review
        """
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        adjusted = mean + factor * (x - mean)
        return torch.clamp(adjusted, 0, 1)
    
    def gamma_correction(self, x, gamma=1.2):
        """
        Apply gamma correction - common in medical imaging
        """
        return torch.pow(x, gamma)
    
    def intensity_shift(self, x, shift=0.1):
        """
        Apply a global intensity shift - simulates changes
        in exposure settings
        """
        shifted = x + shift
        return torch.clamp(shifted, 0, 1)
    
    def windowing(self, x, center=0.5, width=0.5, intensity=1.0):
        """
        Simulate medical image windowing - a process where
        radiologists adjust contrast to focus on specific tissue types
        
        Args:
            x: Input tensor
            center: Center of the window (0-1)
            width: Width of the window (0-1)
            intensity: Control the strength of the windowing effect
            
        Returns:
            Windowed image
        """
        # Scale intensity to control effect strength
        scaled_width = width * (1.0 + intensity)
        
        # Apply windowing
        min_val = center - scaled_width / 2
        max_val = center + scaled_width / 2
        
        # Rescale values within the window to full range
        windowed = torch.clamp(x, min_val, max_val)
        normalized = (windowed - min_val) / (max_val - min_val)
        
        # Blend with original based on intensity
        return normalized * intensity + x * (1 - intensity)
    
    def zoom(self, x, factor=1.2):
        """
        Simulate zoom/crop transformation - common when radiologists
        focus on regions of interest
        
        Args:
            x: Input tensor
            factor: Zoom factor (>1 means zoom in)
            
        Returns:
            Zoomed and rescaled image
        """
        B, C, H, W = x.size()
        
        # Calculate new dimensions and padding
        new_H = int(H / factor)
        new_W = int(W / factor)
        
        # Ensure dimensions are even
        new_H = new_H - (new_H % 2)
        new_W = new_W - (new_W % 2)
        
        # Calculate padding needed
        pad_H = (H - new_H) // 2
        pad_W = (W - new_W) // 2
        
        # Crop center
        cropped = x[:, :, pad_H:pad_H+new_H, pad_W:pad_W+new_W]
        
        # Resize back to original dimensions
        zoomed = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        
        return zoomed


# Original implementation (for compatibility with older code)
class NoiseLayer(nn.Module):
    def __init__(self, noise_types=None):
        super(NoiseLayer, self).__init__()
        self.noise_types = noise_types or ['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper']
        
    def forward(self, x, noise_type=None):

        x = torch.clamp(x, 0, 1)

        # If no specific noise type is specified, randomly choose one
        if noise_type is None:
            noise_type = np.random.choice(self.noise_types)
        
        # Apply the selected noise
        if noise_type == 'dropout':
            result = self.dropout(x)
        elif noise_type == 'jpeg':
            result = self.jpeg_compression(x)
        elif noise_type == 'gaussian':
            result = self.gaussian_noise(x)
        elif noise_type == 'blur':
            result = self.gaussian_blur(x)
        elif noise_type == 'salt_pepper':
            result = self.salt_pepper_noise(x)
        else:
            result = x  # No noise
        
        return torch.clamp(result, 0, 1)
    
    def dropout(self, x, dropout_prob=0.5):
        """Random pixel dropout"""
        mask = torch.rand_like(x) > dropout_prob
        return x * mask
    
    def jpeg_compression(self, x, quality_factor=50):
        """Simulate JPEG compression"""
        # This is a simplified simulation
        # For more accurate JPEG simulation, consider using differentiable JPEG libraries

        x = torch.clamp(x, 0, 1)
        
        # Convert to YCbCr (approximate)
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 0:1] + 0.114 * x[:, 0:1]
        
        # DCT approximation (using fixed masks for high frequencies)
        batch_size, _, h, w = y.shape
        
        # Create blocks of 8x8
        y_blocks = y.unfold(2, 8, 8).unfold(3, 8, 8)
        y_blocks = y_blocks.contiguous().view(-1, 8, 8)
        
        # Simple simulation of quantization by zeroing out high frequencies
        mask = torch.ones(8, 8, device=y.device)
        threshold = max(1, int(8 * (1 - quality_factor / 100)))
        for i in range(8):
            for j in range(8):
                if i + j >= 8 - threshold:
                    mask[i, j] = 0
        
        # Apply mask (zero out high frequencies)
        y_blocks = y_blocks * mask
        
        # Reshape back
        y_blocks = y_blocks.view(batch_size, 1, h // 8, w // 8, 8, 8)
        y_compressed = y_blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, 1, h, w)
        
        # For grayscale, just return the Y channel
        return y_compressed
        
    def gaussian_noise(self, x, std=0.05):
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * std
        noisy = x + noise
        return torch.clamp(noisy, 0, 1)
    
    def gaussian_blur(self, x, kernel_size=5, sigma=1.0):
        """Apply Gaussian blur"""
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create 1D Gaussian kernel
        channels = x.shape[1]
        
        # Create Gaussian kernel manually
        # First create a 1D kernel
        half_size = kernel_size // 2
        kernel_1d = torch.exp(-torch.arange(-half_size, half_size+1, dtype=torch.float, device=x.device) ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
        
        # Apply padding
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        # Apply convolution with the Gaussian kernel
        # We need to group the channels since we want to apply the same Gaussian filter to each channel
        blurred = F.conv2d(x_padded, kernel_2d, groups=channels)
        
        return blurred
    
    def salt_pepper_noise(self, x, density=0.1):
        """Add salt and pepper noise"""
        noise = torch.rand_like(x)
        
        # Salt (white) noise
        salt = (noise < density/2).float()
        
        # Pepper (black) noise
        pepper = (noise > 1 - density/2).float()
        
        # Apply salt and pepper noise
        noisy = x.clone()
        noisy[salt > 0] = 1.0
        noisy[pepper > 0] = 0.0
        
        return noisy