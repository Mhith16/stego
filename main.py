import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import project modules
from models.feature_analyzer import MedicalFeatureAnalyzer, SimpleFeatureAnalyzer
from models.encoder import EnhancedSteganographyEncoder
from models.decoder import EnhancedSteganographyDecoder
from models.discriminator import EnhancedDiscriminator
from models.noise_layer import EnhancedNoiseLayer
from utils.data_loader import get_data_loaders
from utils.metrics import compute_psnr, compute_ssim, compute_bit_accuracy, PerformanceTracker

# For backward compatibility
from models.feature_analyzer import FeatureAnalysisDenseNet
from models.encoder import SteganographyEncoder
from models.decoder import SteganographyDecoder
from models.discriminator import Discriminator
from models.noise_layer import NoiseLayer


def evaluate(args):
    """
    Evaluate trained steganography models on test data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    _, val_loader = get_data_loaders(
        args.xray_dir, 
        args.label_dir,
        batch_size=args.batch_size,
        val_split=1.0,  # Use all data for evaluation
        image_size=(args.image_size, args.image_size)
    )
    
    # Import the load_correct_models function
    from model_config import load_correct_models
    
    # Initialize models with the correct architecture based on saved weights
    print("Initializing models...")
    feature_analyzer, encoder, decoder, noise_layer = load_correct_models(args, device)
    
    # Set models to evaluation mode
    feature_analyzer.eval()
    encoder.eval()
    decoder.eval()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize performance tracker
    tracker = PerformanceTracker()
    
    # Track metrics by noise type
    noise_type_metrics = {noise_type: PerformanceTracker() for noise_type in args.noise_types}
    
    # Process dataset
    print("Evaluating on dataset...")
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader)):
            # Get data
            images = data['image'].to(device)
            messages = data['patient_data'].to(device)
            patient_text = data['patient_text']
            
            # Ensure messages have the right dimensions
            if messages.dim() == 3:  # [B, 1, L]
                messages = messages.squeeze(1)  # [B, L]
            
            # Generate feature weights
            feature_weights = feature_analyzer(images)
            
            # Generate stego images
            stego_images = encoder(images, messages, feature_weights)
            
            # Extract messages from clean stego images
            decoded_messages = decoder(stego_images)
            
            # Update metrics
            tracker.update(images, stego_images, messages, decoded_messages)
            
            # Test on different noise types
            for noise_type in args.noise_types:
                # Apply noise
                try:
                    noisy_stego = noise_layer(stego_images, noise_type)
                
                    # Decode from noisy image
                    noisy_decoded = decoder(noisy_stego)
                    
                    # Update noise-specific metrics
                    noise_type_metrics[noise_type].update(images, stego_images, messages, noisy_decoded)
                except Exception as e:
                    print(f"Error with noise type {noise_type}: {e}")
                    # Use a default tracker in case of error
                    dummy_decoded = torch.ones_like(messages) * 0.5
                    noise_type_metrics[noise_type].update(images, stego_images, messages, dummy_decoded)
            
            # Save example visualizations
            if batch_idx == 0:
                save_visualizations(
                    images, stego_images, feature_weights, noise_layer, 
                    decoder, messages, decoded_messages, args
                )
    
    # Print and save evaluation results
    print_evaluation_results(tracker, noise_type_metrics, args)
    

def save_visualizations(images, stego_images, feature_weights, noise_layer, decoder, messages, decoded_messages, args):
    """
    Save visualization of steganography results
    """
    # Take the first image from the batch
    image = images[0]
    stego_image = stego_images[0]
    feature_weight = feature_weights[0]
    
    # Apply different noise types
    noise_images = {}
    decoded_under_noise = {}
    
    for noise_type in args.noise_types:
        noise_img = noise_layer(stego_image.unsqueeze(0), noise_type)[0]
        noise_images[noise_type] = noise_img
        
        # Decode under noise
        decoded = decoder(noise_img.unsqueeze(0))[0]
        decoded_under_noise[noise_type] = compute_bit_accuracy(messages[0], decoded)
    
    # Calculate the difference
    diff = torch.abs(image - stego_image)
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(image[0].cpu().numpy(), cmap='gray')
    plt.title("Original X-ray")
    plt.axis('off')
    
    # Feature weights
    plt.subplot(2, 4, 2)
    plt.imshow(feature_weight[0].cpu().numpy(), cmap='plasma')
    plt.title("Feature Importance Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Stego image
    plt.subplot(2, 4, 3)
    plt.imshow(stego_image[0].cpu().numpy(), cmap='gray')
    psnr = compute_psnr(image.unsqueeze(0), stego_image.unsqueeze(0))
    plt.title(f"Stego Image (PSNR: {psnr:.2f}dB)")
    plt.axis('off')
    
    # Difference
    plt.subplot(2, 4, 4)
    plt.imshow(diff[0].cpu().numpy() * 10, cmap='hot')  # Amplify for visibility
    plt.title("Difference (x10)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Noise examples (4 types)
    for i, (noise_type, noise_img) in enumerate(list(noise_images.items())[:4]):
        plt.subplot(2, 4, 5 + i)
        plt.imshow(noise_img[0].cpu().numpy(), cmap='gray')
        plt.title(f"{noise_type} (Acc: {decoded_under_noise[noise_type]:.2f})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'steganography_results.png'), dpi=200)
    
    # If interactive mode is enabled, show the plot
    if args.interactive:
        plt.show()


def print_evaluation_results(tracker, noise_type_metrics, args):
    """
    Print and save evaluation results
    """
    # Get average metrics
    avg_metrics = tracker.get_averages()
    
    # Print overall results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"SSIM: {avg_metrics['ssim']:.4f}")
    print(f"Bit Accuracy (No Noise): {avg_metrics['bit_accuracy']:.4f}")
    
    # Print metrics by noise type
    print("\nPerformance under different noise types:")
    print("-"*50)
    print(f"{'Noise Type':<15} {'Bit Accuracy':<15} {'PSNR (dB)':<10} {'SSIM':<10}")
    print("-"*50)
    
    # Format and print results for each noise type
    for noise_type, noise_tracker in noise_type_metrics.items():
        noise_metrics = noise_tracker.get_averages()
        print(f"{noise_type:<15} {noise_metrics['bit_accuracy']:<15.4f} {avg_metrics['psnr']:<10.2f} {avg_metrics['ssim']:<10.4f}")
    
    # Save results to file
    with open(os.path.join(args.results_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"{'Noise Type':<15} {'PSNR (dB)':<10} {'SSIM':<10} {'Bit Accuracy':<15} {'Capacity (bpp)':<15}\n")
        f.write("-"*70 + "\n")
        
        # Calculate capacity in bits per pixel
        h, w = args.image_size, args.image_size
        total_pixels = h * w
        capacity = args.message_length / total_pixels
        
        # First write the no-noise results
        f.write(f"{'none':<15} {avg_metrics['psnr']:<10.2f} {avg_metrics['ssim']:<10.4f} {avg_metrics['bit_accuracy']:<15.4f} {capacity:<15.4f}\n")
        
        # Then write results for each noise type
        for noise_type, noise_tracker in noise_type_metrics.items():
            noise_metrics = noise_tracker.get_averages()
            effective_capacity = capacity * noise_metrics['bit_accuracy']  # Effective capacity accounts for bit errors
            f.write(f"{noise_type:<15} {avg_metrics['psnr']:<10.2f} {avg_metrics['ssim']:<10.4f} {noise_metrics['bit_accuracy']:<15.4f} {effective_capacity:<15.4f}\n")
    
    print(f"\nResults saved to {os.path.join(args.results_dir, 'evaluation_metrics.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Image Steganography Evaluation")
    
    # Data paths
    parser.add_argument('--xray_dir', type=str, required=True, help='Directory containing X-ray images')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing patient data text files')
    
    # Model parameters
    parser.add_argument('--load_models', action='store_true', help='Load pretrained models')
    parser.add_argument('--model_path', type=str, default='./models/weights/final_models', help='Path to model weights')
    parser.add_argument('--message_length', type=int, default=256, help='Length of binary message')
    parser.add_argument('--use_enhanced_models', action='store_true', help='Use enhanced models for evaluation')
    parser.add_argument('--image_size', type=int, default=512, help='Image size (square)')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--noise_types', nargs='+', default=['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper'],
                        help='Types of noise to evaluate on')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--interactive', action='store_true', help='Show interactive visualizations')
    
    args = parser.parse_args()
    evaluate(args)