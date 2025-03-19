import os
import argparse
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import sys
import torch.nn.functional as F

# Add parent directory to path to access models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models from the main module
from models.feature_analyzer import MedicalFeatureAnalyzer, SimpleFeatureAnalyzer
from models.encoder import EnhancedSteganographyEncoder

# For backward compatibility
from models.feature_analyzer import FeatureAnalysisDenseNet
from models.encoder import SteganographyEncoder


def text_to_binary(text, max_length=256, with_error_correction=True):
    """
    Convert text to binary tensor representation with error correction
    
    Args:
        text: Input text string
        max_length: Maximum binary length
        with_error_correction: Whether to add error correction bits
        
    Returns:
        Binary tensor representation
    """
    # Convert each character to its ASCII binary representation
    binary = ''.join([format(ord(c), '08b') for c in text])
    
    # Add parity bits for simple error detection/correction (every 8 bits)
    if with_error_correction:
        with_parity = ''
        for i in range(0, len(binary), 8):
            block = binary[i:i+8]
            if len(block) == 8:  # Only add parity for complete bytes
                parity = (sum(int(bit) for bit in block) % 2)  # Even parity
                with_parity += block + str(parity)
        binary = with_parity
    
    # Convert to tensor
    binary_tensor = torch.tensor([int(bit) for bit in binary], dtype=torch.float32)
    
    # Check length and adjust
    if len(binary_tensor) > max_length:
        print(f"Warning: Binary data length ({len(binary_tensor)}) exceeds max_length ({max_length}). Truncating.")
        binary_tensor = binary_tensor[:max_length]
    else:
        # Pad with zeros
        padding = torch.zeros(max_length - len(binary_tensor))
        binary_tensor = torch.cat([binary_tensor, padding])
    
    return binary_tensor


def preprocess_image(image_path, target_size=(512, 512)):
    """
    Preprocess image for the steganography model
    
    Args:
        image_path: Path to the input image
        target_size: Image size to resize to
        
    Returns:
        Preprocessed image tensor [1, 1, H, W]
    """
    # Load image
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Try with PIL if OpenCV fails
            image = Image.open(image_path).convert('L')
            image = np.array(image)
    except Exception as e:
        print(f"Error reading image: {e}")
        # Try with PIL as a fallback
        image = Image.open(image_path).convert('L')
        image = np.array(image)
    
    # Normalize and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),  # Scales to [0, 1]
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def embed_patient_data(image_path, text, model_path, output_path=None, message_length=256, image_size=(512, 512), use_enhanced_models=True):
    """Embed patient data into an X-ray image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set default output path if not provided
    if output_path is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(os.path.dirname(image_path), f"{name}_stego{ext}")
    
    try:
        # Import load_correct_models (add this import at the top of the file too)
        from model_config import load_correct_models
        
        # Create args object for model loading
        class Args:
            def __init__(self):
                self.model_path = model_path
                self.message_length = message_length
                self.use_enhanced_models = use_enhanced_models
        
        args = Args()
        
        # Load models
        print(f"Loading models from: {model_path}")
        feature_analyzer, encoder, _, _ = load_correct_models(args, device)
        
        # Set models to evaluation mode
        feature_analyzer.eval()
        encoder.eval()
    
    # try:
    #     # Load models
    #     print(f"Loading models from: {model_path}")
        
    #     if use_enhanced_models:
    #         # Enhanced models
    #         feature_analyzer = SimpleFeatureAnalyzer(in_channels=1).to(device)
    #         encoder = EnhancedSteganographyEncoder(image_channels=1, message_length=message_length).to(device)
            
    #         # Try to load state dictionaries
    #         try:
    #             feature_analyzer.load_state_dict(
    #                 torch.load(os.path.join(model_path, 'feature_analyzer.pth'), map_location=device)
    #             )
    #             encoder.load_state_dict(
    #                 torch.load(os.path.join(model_path, 'encoder.pth'), map_location=device)
    #             )
    #         except Exception as e:
    #             print(f"Warning: Error loading enhanced models: {e}")
    #             print("Falling back to original models")
    #             use_enhanced_models = False
        
    #     if not use_enhanced_models:
    #         # Original models for backward compatibility
    #         feature_analyzer = FeatureAnalysisDenseNet(in_channels=1).to(device)
    #         encoder = SteganographyEncoder(image_channels=1).to(device)
            
    #         feature_analyzer.load_state_dict(
    #             torch.load(os.path.join(model_path, 'feature_analyzer.pth'), map_location=device), 
    #             strict=False
    #         )
    #         encoder.load_state_dict(
    #             torch.load(os.path.join(model_path, 'encoder.pth'), map_location=device), 
    #             strict=False
    #         )
        
        # Set models to evaluation mode
        feature_analyzer.eval()
        encoder.eval()
        
        # Preprocess image
        image_tensor = preprocess_image(image_path, image_size).to(device)
        
        # Prepare text data
        binary_tensor = text_to_binary(text, message_length, with_error_correction=True).unsqueeze(0).to(device)
        
        # Generate stego image
        with torch.no_grad():
            # Get feature weights
            feature_weights = feature_analyzer(image_tensor)
            
            # Embed message
            stego_image = encoder(image_tensor, binary_tensor, feature_weights)
        
        # Calculate metrics
        mse = torch.mean((image_tensor - stego_image) ** 2).item()
        psnr = 10 * np.log10(1.0 / mse)
        
        # Save stego image
        stego_image_np = stego_image[0, 0].cpu().numpy() * 255
        cv2.imwrite(output_path, stego_image_np.astype(np.uint8))
        
        print(f"Stego image created successfully: {output_path}")
        print(f"Image quality (PSNR): {psnr:.2f} dB")
        
        return output_path
        
    except Exception as e:
        print(f"Error embedding data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed patient data into an X-ray image")
    parser.add_argument('--image', type=str, required=True, help='Path to the X-ray image')
    parser.add_argument('--text', type=str, required=True, help='Patient data text or path to text file')
    parser.add_argument('--model_path', type=str, default='./models/weights/final_models', help='Path to trained models')
    parser.add_argument('--output', type=str, help='Path to save the stego image')
    parser.add_argument('--message_length', type=int, default=256, help='Maximum binary message length')
    parser.add_argument('--original_models', action='store_true', help='Use original models instead of enhanced models')
    parser.add_argument('--image_size', type=int, default=512, help='Image size (square)')
    
    args = parser.parse_args()
    
    # Check if text is a file path
    if os.path.exists(args.text):
        with open(args.text, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read().strip()
    else:
        text = args.text
    
    # Embed the data
    image_size = (args.image_size, args.image_size)
    embed_patient_data(
        args.image, 
        text, 
        args.model_path, 
        args.output, 
        args.message_length, 
        image_size,
        not args.original_models
    )