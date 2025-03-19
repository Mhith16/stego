# model_config.py
from models.feature_analyzer import FeatureAnalysisDenseNet, SimpleFeatureAnalyzer, MedicalFeatureAnalyzer
from models.encoder import SteganographyEncoder, EnhancedSteganographyEncoder
from models.decoder import SteganographyDecoder, EnhancedSteganographyDecoder
from models.discriminator import Discriminator, EnhancedDiscriminator

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
    feature_analyzer = SimpleFeatureAnalyzer(in_channels=1)
    encoder = EnhancedSteganographyEncoder(image_channels=1, message_length=message_length)
    decoder = EnhancedSteganographyDecoder(image_channels=1, message_length=message_length)
    discriminator = EnhancedDiscriminator(image_channels=1)
    
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
    from models.noise_layer import EnhancedNoiseLayer, NoiseLayer
    import os
    import torch
    
    print(f"Loading models from {args.model_path}...")
    
    if args.use_enhanced_models:
        # Try with enhanced models first
        feature_analyzer, encoder, decoder, _ = get_enhanced_models(args.message_length)
        feature_analyzer = feature_analyzer.to(device)
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        noise_layer = EnhancedNoiseLayer().to(device)
        
        try:
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
            print("Enhanced models loaded successfully")
            return feature_analyzer, encoder, decoder, noise_layer
            
        except Exception as e:
            print(f"Error loading enhanced models: {e}")
            print("Falling back to original models with correct configuration")
    
    # Fall back to original models with correct configuration
    feature_analyzer = get_training_feature_analyzer().to(device)
    encoder = get_training_encoder().to(device)
    decoder = get_training_decoder().to(device)
    noise_layer = NoiseLayer().to(device)
    
    try:
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
        print("Original models loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Running with untrained models")
    
    return feature_analyzer, encoder, decoder, noise_layer