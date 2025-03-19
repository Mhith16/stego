import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio between two images
    
    Args:
        img1: First image tensor or array
        img2: Second image tensor or array
        
    Returns:
        PSNR value in dB
    """
    if torch.is_tensor(img1) and torch.is_tensor(img2):
        mse = torch.mean((img1 - img2) ** 2).item()
    else:
        mse = np.mean((img1 - img2) ** 2)
        
    if mse < 1.0e-10:  # Avoid division by zero
        return float('inf')
        
    max_pixel = 1.0
    return 10 * np.log10((max_pixel ** 2) / mse)


def compute_ssim(img1, img2):
    """
    Compute Structural Similarity Index between two images
    
    Args:
        img1: First image tensor or array
        img2: Second image tensor or array
        
    Returns:
        SSIM value between 0 and 1
    """
    # Convert tensors to numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # If batched, compute average SSIM
    if img1.ndim == 4:  # [B, C, H, W]
        ssim_values = []
        for i in range(img1.shape[0]):
            if img1.shape[1] == 1:  # Grayscale
                ssim_val = ssim(img1[i, 0], img2[i, 0], data_range=1.0)
            else:  # RGB
                ssim_val = ssim(
                    img1[i].transpose(1, 2, 0), 
                    img2[i].transpose(1, 2, 0), 
                    multichannel=True, 
                    data_range=1.0
                )
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:  # [C, H, W] or [H, W]
        if img1.ndim == 3:
            if img1.shape[0] == 1:  # [1, H, W]
                return ssim(img1[0], img2[0], data_range=1.0)
            else:  # [3, H, W]
                return ssim(
                    img1.transpose(1, 2, 0), 
                    img2.transpose(1, 2, 0), 
                    multichannel=True, 
                    data_range=1.0
                )
        else:  # [H, W]
            return ssim(img1, img2, data_range=1.0)


def compute_bit_accuracy(original_msg, decoded_msg, threshold=0.5):
    """
    Compute bit accuracy between original and decoded messages
    
    Args:
        original_msg: Original binary message
        decoded_msg: Decoded binary message
        threshold: Threshold for binary decision (default 0.5)
        
    Returns:
        Accuracy as a value between 0 and 1
    """
    if torch.is_tensor(original_msg):
        original_bits = (original_msg > threshold).float()
    else:
        original_bits = (np.array(original_msg) > threshold).astype(float)
        
    if torch.is_tensor(decoded_msg):
        decoded_bits = (decoded_msg > threshold).float()
    else:
        decoded_bits = (np.array(decoded_msg) > threshold).astype(float)
    
    if torch.is_tensor(original_bits) and torch.is_tensor(decoded_bits):
        return torch.mean((original_bits == decoded_bits).float()).item()
    else:
        return np.mean(original_bits == decoded_bits)


def compute_embedding_capacity(image_shape, message_length):
    """
    Compute bits per pixel (bpp) embedding capacity
    
    Args:
        image_shape: Shape of the image [C, H, W]
        message_length: Length of the binary message
        
    Returns:
        Bits per pixel (bpp) value
    """
    h, w = image_shape[-2:]  # Get height and width
    total_pixels = h * w
    bits_per_pixel = message_length / total_pixels
    return bits_per_pixel


def compute_text_accuracy(original_text, decoded_text):
    """
    Compute character-level accuracy between original and decoded text
    
    Args:
        original_text: Original text string
        decoded_text: Decoded text string
        
    Returns:
        Character accuracy as a value between 0 and 1
    """
    # Make sure inputs are strings
    original_text = str(original_text)
    decoded_text = str(decoded_text)
    
    # Get the minimum length for comparison
    min_len = min(len(original_text), len(decoded_text))
    
    # Count matching characters
    matches = sum(1 for i in range(min_len) if original_text[i] == decoded_text[i])
    
    # Calculate accuracy (penalize for length difference)
    max_len = max(len(original_text), len(decoded_text))
    if max_len == 0:  # Handle empty strings
        return 1.0
        
    return matches / max_len


class PerformanceTracker:
    """
    Track metrics during training or evaluation
    """
    def __init__(self):
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'bit_accuracy': [],
            'text_accuracy': []
        }
        
    def update(self, original_images, stego_images, original_msgs, decoded_msgs, 
               original_text=None, decoded_text=None):
        """Update metrics with a new batch of data"""
        self.metrics['psnr'].append(compute_psnr(original_images, stego_images))
        self.metrics['ssim'].append(compute_ssim(original_images, stego_images))
        self.metrics['bit_accuracy'].append(compute_bit_accuracy(original_msgs, decoded_msgs))
        
        if original_text is not None and decoded_text is not None:
            self.metrics['text_accuracy'].append(compute_text_accuracy(original_text, decoded_text))
    
    def get_averages(self):
        """Get average values for all tracked metrics"""
        return {
            metric: np.mean(values) if values else 0.0
            for metric, values in self.metrics.items()
        }
        
    def reset(self):
        """Reset all metrics"""
        for key in self.metrics:
            self.metrics[key] = []
            
    def __str__(self):
        """String representation with metric averages"""
        averages = self.get_averages()
        return (
            f"PSNR: {averages['psnr']:.2f} dB, "
            f"SSIM: {averages['ssim']:.4f}, "
            f"Bit Accuracy: {averages['bit_accuracy']:.4f}, "
            f"Text Accuracy: {averages['text_accuracy']:.4f}"
        )