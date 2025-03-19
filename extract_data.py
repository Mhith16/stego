import os
import argparse
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import sys

# Add parent directory to path to access models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models and model loading function
from model_config import load_correct_models


def binary_to_text(binary_tensor, threshold=0.5, with_error_correction=True):
    """
    Convert binary tensor to text with error correction
    
    Args:
        binary_tensor: Binary tensor from decoder
        threshold: Threshold for binary decision
        with_error_correction: Whether to apply error correction
        
    Returns:
        Extracted text string
    """
    # Convert tensor to binary string with adaptive threshold
    if isinstance(binary_tensor, torch.Tensor):
        binary_output = (binary_tensor > threshold).int().cpu().numpy()
    else:
        binary_output = (np.array(binary_tensor) > threshold).astype(int)
        
    binary_string = ''.join(['1' if bit else '0' for bit in binary_output])
    
    # Remove padding zeros from the end
    # Find the last '1' bit and keep everything before it plus 8 bits 
    # (to ensure we don't cut off in the middle of a character)
    try:
        last_one_index = binary_string.rindex('1')
        end_index = min(last_one_index + 9, len(binary_string))
        binary_string = binary_string[:end_index]
    except ValueError:
        # No '1' bits found, likely all zeros - keep a minimal amount
        binary_string = binary_string[:8]
    
    # Apply error correction if requested
    if with_error_correction:
        # Process 9-bit blocks (8 data bits + 1 parity bit)
        corrected = ''
        for i in range(0, len(binary_string), 9):
            if i + 9 <= len(binary_string):
                block = binary_string[i:i+8]
                parity_bit = binary_string[i+8]
                
                # Check parity (even parity)
                computed_parity = str(sum(int(bit) for bit in block) % 2)
                
                # If parity doesn't match, there's an error
                if computed_parity != parity_bit:
                    # Simple error handling: we just note it
                    # In a real system, more sophisticated error correction would be used
                    pass
                
                # Add the data bits (without parity) to the corrected string
                corrected += block
        
        binary_string = corrected
    
    # Convert binary to text (8 bits per character)
    text = ''
    for i in range(0, len(binary_string), 8):
        if i + 8 <= len(binary_string):
            byte = binary_string[i:i+8]
            try:
                char = chr(int(byte, 2))
                # Only add printable ASCII characters and common whitespace
                if char.isprintable() or char in ('\n', '\t', ' '):
                    text += char
                else:
                    # Replace unprintable characters with a placeholder
                    text += '?'
            except ValueError:
                # Skip invalid binary sequences
                pass
    
    return text


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


def extract_patient_data(stego_path, model_path, message_length=256, image_size=(512, 512), use_enhanced_models=True):
    """
    Extract patient data from a stego image
    
    Args:
        stego_path: Path to the stego image
        model_path: Directory containing trained models
        message_length: Length of the binary message
        image_size: Input image size (width, height)
        use_enhanced_models: Whether to use enhanced models or original models
    
    Returns:
        The extracted patient data as text
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create args object for model loading
        class Args:
            def __init__(self):
                self.model_path = model_path
                self.message_length = message_length
                self.use_enhanced_models = use_enhanced_models
        
        args = Args()
        
        # Load decoder model using the correct architecture
        _, _, decoder, _ = load_correct_models(args, device)
        
        # Get the actual message length from the decoder
        if hasattr(decoder, 'fc_layers') and hasattr(decoder.fc_layers, '-1'):
            actual_message_length = decoder.fc_layers[-1].out_features
            print(f"Using message length {actual_message_length} based on decoder architecture")
        else:
            actual_message_length = message_length
            print(f"Using specified message length: {message_length}")
        
        # Preprocess image
        stego_tensor = preprocess_image(stego_path, image_size).to(device)
        
        # Extract binary data
        with torch.no_grad():
            binary_output = decoder(stego_tensor)
        
        # Try multiple thresholds for better extraction
        thresholds = [0.5, 0.4, 0.6, 0.3, 0.7]
        extracted_texts = []
        
        for threshold in thresholds:
            # Convert to text
            text = binary_to_text(binary_output[0], threshold=threshold, with_error_correction=True)
            extracted_texts.append(text)
        
        # Choose the best result (the one with the most printable characters)
        best_text = max(extracted_texts, key=lambda t: sum(1 for c in t if c.isprintable()))
        
        print("\nSuccessfully extracted patient data:")
        print("-" * 40)
        print(best_text)
        print("-" * 40)
        
        return best_text
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patient data from a stego image")
    parser.add_argument('--image', type=str, required=True, help='Path to the stego image')
    parser.add_argument('--model_path', type=str, default='./models/weights/final_models', help='Path to trained models')
    parser.add_argument('--output', type=str, help='Path to save the extracted text (optional)')
    parser.add_argument('--message_length', type=int, default=256, help='Binary message length')
    parser.add_argument('--use_enhanced_models', action='store_true', help='Use enhanced models')
    parser.add_argument('--image_size', type=int, default=512, help='Image size (square)')
    
    args = parser.parse_args()
    
    # Extract the data
    image_size = (args.image_size, args.image_size)
    extracted_text = extract_patient_data(
        args.image, 
        args.model_path, 
        args.message_length, 
        image_size,
        args.use_enhanced_models
    )
    
    # Save to file if requested
    if extracted_text and args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"Extracted text saved to: {args.output}")