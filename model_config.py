# model_config.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_analyzer import FeatureAnalysisDenseNet, SimpleFeatureAnalyzer, MedicalFeatureAnalyzer
from models.encoder import SteganographyEncoder, EnhancedSteganographyEncoder
from models.decoder import SteganographyDecoder, EnhancedSteganographyDecoder
from models.discriminator import Discriminator, EnhancedDiscriminator
from models.noise_layer import EnhancedNoiseLayer, NoiseLayer

# Import the exact architecture models
from exact_model_architecture import ExactFeatureAnalyzer, ExactEnhancedEncoder, ExactEnhancedDecoder

def get_training_feature_analyzer():
    """Returns feature analyzer with EXACT training configuration"""
    return FeatureAnalysisDenseNet(
        in_channels=1,
        growth_rate=16,
        block_config=(3, 6, 12, 24),  # This matches your trained model
        num_init_features=64  # Important: This was 64 in training, not 32
    )

def get_training_encoder():
    """Returns encoder with training configuration"""
    return SteganographyEncoder(
        image_channels=1,
        growth_rate=16,
        num_dense_layers=4
    )

def get_training_decoder():
    """Returns decoder with training configuration"""
    return SteganographyDecoder(
        image_channels=1,
        growth_rate=16,
        num_layers=4,
        message_length=256
    )

def get_enhanced_models(message_length=256):
    """Returns enhanced versions of all models with proper configuration"""
    feature_analyzer = ExactFeatureAnalyzer(in_channels=1)
    encoder = ExactEnhancedEncoder(image_channels=1, message_length=message_length)
    decoder = ExactEnhancedDecoder(image_channels=1, message_length=message_length)
    discriminator = EnhancedDiscriminator(image_channels=1, use_logits=True)
    
    return feature_analyzer, encoder, decoder, discriminator

def load_correct_models(args, device):
    """
    Load models with the correct architecture based on saved weights
    
    Args:
        args: Command line arguments
        device: Torch device (cuda/cpu)
        
    Returns:
        Tuple of (feature_analyzer, encoder, decoder, noise_layer)
    """
    print(f"Loading models from {args.model_path}...")
    
    # Whether we're using the enhanced models for inference
    use_enhanced = getattr(args, 'use_enhanced_models', False)
    
    # Create new models with the exact architecture matching the saved weights
    if use_enhanced:
        print("Initializing enhanced models with exact architecture...")
        feature_analyzer, encoder, decoder, _ = get_enhanced_models(args.message_length)
    else:
        print("Initializing original models with exact architecture...")
        feature_analyzer = get_training_feature_analyzer()
        encoder = get_training_encoder()
        decoder = get_training_decoder()
    
    # Move models to device
    feature_analyzer = feature_analyzer.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Initialize noise layer
    noise_layer = EnhancedNoiseLayer().to(device)
    
    # Try to load state dictionaries with strict=False to ignore missing keys
    try:
        # Load model weights
        feature_analyzer.load_state_dict(
            torch.load(os.path.join(args.model_path, 'feature_analyzer.pth'), 
                      map_location=device),
            strict=False
        )
        
        encoder.load_state_dict(
            torch.load(os.path.join(args.model_path, 'encoder.pth'), 
                      map_location=device),
            strict=False
        )
        
        decoder.load_state_dict(
            torch.load(os.path.join(args.model_path, 'decoder.pth'), 
                      map_location=device),
            strict=False
        )
        
        print("Models loaded successfully. Some layers may have been initialized with random weights.")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Running with untrained models. Results may be poor.")
    
    # Set models to evaluation mode
    feature_analyzer.eval()
    encoder.eval()
    decoder.eval()
    
    return feature_analyzer, encoder, decoder, noise_layer