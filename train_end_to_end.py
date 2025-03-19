import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_analyzer import FeatureAnalysisDenseNet, MedicalFeatureAnalyzer, SimpleFeatureAnalyzer
from models.encoder import EnhancedSteganographyEncoder, SteganographyEncoder
from models.decoder import EnhancedSteganographyDecoder, SteganographyDecoder
from models.discriminator import Discriminator, EnhancedDiscriminator
from models.noise_layer import EnhancedNoiseLayer
from utils.data_loader import get_data_loaders
from utils.metrics import compute_psnr, compute_ssim, compute_bit_accuracy
from model_config import load_correct_models



# Custom SSIM loss function
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def forward(self, img1, img2):
        # Check if window needs to be moved to the same device as input
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


def preprocess_patient_data(text, max_length=256):
    """
    Preprocess patient data text to binary representation.
    Specifically optimized for the format:
    
    Name: XXX
    Age: XXY
    ID: CDSZZZZZ
    """
    # Convert each character to its ASCII binary representation
    binary = ''.join([format(ord(c), '08b') for c in text])
    
    # Add parity bits for simple error detection/correction (every 8 bits)
    with_parity = ''
    for i in range(0, len(binary), 8):
        block = binary[i:i+8]
        if len(block) == 8:  # Only add parity for complete bytes
            parity = (sum(int(bit) for bit in block) % 2)  # Even parity
            with_parity += block + str(parity)
    
    # Convert to tensor
    binary_tensor = torch.tensor([int(bit) for bit in with_parity], dtype=torch.float32)
    
    # Handle length - pad or truncate
    if len(binary_tensor) > max_length:
        binary_tensor = binary_tensor[:max_length]
    else:
        # Pad with zeros
        padding = torch.zeros(max_length - len(binary_tensor))
        binary_tensor = torch.cat([binary_tensor, padding])
    
    return binary_tensor

def train_phase2_with_fixes(feature_analyzer, encoder, decoder, discriminator, 
                           noise_layer, train_loader, val_loader, args, device, writer):
    """
    Phase 2 training with robust fixes for numerical stability
    """
    print("\n=== Phase 2: Training for robustness with noise ===\n")
    
    # Initialize optimizers with lower learning rate for stability
    lr_phase2 = args.lr * 0.5  # Half the learning rate for stability
    fa_optimizer = optim.Adam(feature_analyzer.parameters(), lr=lr_phase2)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr_phase2)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr_phase2)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr_phase2)
    
    # Use BCEWithLogitsLoss for better numerical stability (internal sigmoid + BCE)
    # For discriminator and adversarial loss
    bce_logits_loss = nn.BCEWithLogitsLoss()
    
    # Regular BCE for decoder where we want to keep the sigmoid
    bce_loss = nn.BCELoss()
    
    # Other losses
    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss()
    
    # Initialize gradient scaler for mixed precision (helps with stability)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Reduce epochs_phase2 if needed
    actual_epochs = min(args.epochs_phase2, 20)  # Cap at 20 for this run
    print(f"Running for {actual_epochs} epochs")
    
    for epoch in range(actual_epochs):
        feature_analyzer.train()
        encoder.train()
        decoder.train()
        discriminator.train()
        
        total_disc_loss = 0
        total_encoder_loss = 0
        total_decoder_loss = 0
        total_combined_loss = 0
        
        # Start with very low noise intensity and gradually increase
        # Use a sigmoid curve for smoother ramping
        noise_intensity = min(0.7, 0.7 / (1 + np.exp(-10 * (epoch / actual_epochs - 0.3))))
        print(f"Phase 2 - Epoch {epoch+1}/{actual_epochs} (Noise intensity: {noise_intensity:.2f})")
        
        for batch_idx, data in enumerate(tqdm(train_loader)):
            images = data['image'].to(device)
            messages = data['patient_data'].to(device)
            
            # Strict check and correction of input data
            images = torch.clamp(images, 0, 1)
            
            # Ensure messages have the right dimensions
            if messages.dim() == 3:  # [B, 1, L]
                messages = messages.squeeze(1)  # [B, L]
                
            # Ensure message values are in [0,1]
            messages = torch.clamp(messages, 0, 1)
            
            # -------------------------------------------
            # Train discriminator first
            # -------------------------------------------
            disc_optimizer.zero_grad()
            
            # Generate feature weights
            with torch.no_grad():  # No need to track gradients here
                feature_weights = feature_analyzer(images)
                feature_weights = torch.clamp(feature_weights, 0, 1)
            
                # Generate stego images
                stego_images = encoder(images, messages, feature_weights)
                stego_images = torch.clamp(stego_images, 0, 1)
            
            # Use raw logits for discriminator (BCEWithLogitsLoss applies sigmoid internally)
            # Remove final sigmoid from discriminator forward call if needed
            if hasattr(discriminator, 'use_logits') and discriminator.use_logits:
                # For discriminators that output logits
                real_logits = discriminator(images)
                fake_logits = discriminator(stego_images.detach())
                
                # Create labels with slight smoothing for better stability
                real_labels = torch.ones_like(real_logits).to(device) * 0.9  # 0.9 instead of 1
                fake_labels = torch.zeros_like(fake_logits).to(device) * 0.1  # 0.1 instead of 0
                
                disc_loss_real = bce_logits_loss(real_logits, real_labels)
                disc_loss_fake = bce_logits_loss(fake_logits, fake_labels)
            else:
                # For discriminators with sigmoid already applied
                real_preds = discriminator(images)
                fake_preds = discriminator(stego_images.detach())
                
                # Clamp outputs to valid range
                real_preds = torch.clamp(real_preds, 1e-7, 1 - 1e-7)
                fake_preds = torch.clamp(fake_preds, 1e-7, 1 - 1e-7)
                
                # Create labels with slight smoothing
                real_labels = torch.ones_like(real_preds).to(device) * 0.9
                fake_labels = torch.zeros_like(fake_preds).to(device) * 0.1
                
                disc_loss_real = bce_loss(real_preds, real_labels)
                disc_loss_fake = bce_loss(fake_preds, fake_labels)
            
            disc_loss = disc_loss_real + disc_loss_fake
            
            # Use gradient scaling if available
            if scaler is not None:
                scaler.scale(disc_loss).backward()
                scaler.step(disc_optimizer)
            else:
                disc_loss.backward()
                disc_optimizer.step()
            
            # -------------------------------------------
            # Train encoder, decoder, and feature analyzer
            # -------------------------------------------
            fa_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Generate feature weights
            feature_weights = feature_analyzer(images)
            feature_weights = torch.clamp(feature_weights, 0, 1)
            
            # Generate stego images
            stego_images = encoder(images, messages, feature_weights)
            stego_images = torch.clamp(stego_images, 0, 1)
            
            # Try-except block for noise application
            try:
                # Apply noise with current intensity (but handle potential errors)
                noise_type = np.random.choice(args.noise_types)
                noisy_stego_images = noise_layer(stego_images, noise_type, noise_intensity)
                noisy_stego_images = torch.clamp(noisy_stego_images, 0, 1)
            except Exception as e:
                print(f"WARNING: Error applying noise: {e}")
                print(f"Using original stego images without noise")
                noisy_stego_images = stego_images.clone()
            
            # Decode messages from noisy stego images
            decoded_messages = decoder(noisy_stego_images)
            decoded_messages = torch.clamp(decoded_messages, 1e-7, 1 - 1e-7)  # Ensure valid range
            
            # Get discriminator predictions for generator training
            if hasattr(discriminator, 'use_logits') and discriminator.use_logits:
                disc_logits = discriminator(stego_images)
                adv_loss = bce_logits_loss(disc_logits, real_labels)  # Fool the discriminator
            else:
                disc_preds = discriminator(stego_images)
                disc_preds = torch.clamp(disc_preds, 1e-7, 1 - 1e-7)
                adv_loss = bce_loss(disc_preds, real_labels)  # Fool the discriminator
            
            # Calculate losses
            # 1. Message loss
            message_loss = bce_loss(decoded_messages, messages)
            
            # 2. Image distortion loss (MSE + SSIM)
            img_mse_loss = mse_loss(stego_images, images)
            
            # Try-except for SSIM loss (can sometimes cause errors)
            try:
                img_ssim_loss = ssim_loss(stego_images, images)
            except Exception as e:
                print(f"WARNING: SSIM loss error: {e}")
                img_ssim_loss = torch.tensor(0.0).to(device)
            
            image_loss = img_mse_loss + args.lambda_ssim * img_ssim_loss
            
            # Combined loss with different weights early in training
            # Higher weight on image quality at start, more on message as training progresses
            message_weight = args.lambda_message * min(1.0, epoch / (actual_epochs * 0.3) + 0.5)
            image_weight = args.lambda_image
            adv_weight = args.lambda_adv * min(1.0, epoch / (actual_epochs * 0.5))
            
            combined_loss = message_weight * message_loss + \
                            image_weight * image_loss + \
                            adv_weight * adv_loss
            
            # Use gradient scaling if available
            if scaler is not None:
                scaler.scale(combined_loss).backward()
                
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(feature_analyzer.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                
                scaler.step(fa_optimizer)
                scaler.step(encoder_optimizer)
                scaler.step(decoder_optimizer)
                scaler.update()
            else:
                combined_loss.backward()
                
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(feature_analyzer.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                
                fa_optimizer.step()
                encoder_optimizer.step()
                decoder_optimizer.step()
            
            # Update running losses
            total_disc_loss += disc_loss.item()
            total_encoder_loss += image_loss.item()
            total_decoder_loss += message_loss.item()
            total_combined_loss += combined_loss.item()
            
            # Log every N batches
            if batch_idx % args.log_interval == 0:
                # Calculate metrics
                with torch.no_grad():  # Don't track gradients for metrics
                    psnr = compute_psnr(images, stego_images)
                    ssim = compute_ssim(images, stego_images)
                    bit_acc = compute_bit_accuracy(messages, decoded_messages)
                
                # Log to tensorboard
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Phase2/Loss/Discriminator', disc_loss.item(), step)
                writer.add_scalar('Phase2/Loss/Encoder', image_loss.item(), step)
                writer.add_scalar('Phase2/Loss/Decoder', message_loss.item(), step)
                writer.add_scalar('Phase2/Loss/Combined', combined_loss.item(), step)
                writer.add_scalar('Phase2/Metrics/PSNR', psnr, step)
                writer.add_scalar('Phase2/Metrics/SSIM', ssim, step)
                writer.add_scalar('Phase2/Metrics/BitAccuracy', bit_acc, step)
                writer.add_scalar('Phase2/Noise/Intensity', noise_intensity, step)
                
                # Print stats periodically
                if batch_idx % (args.log_interval * 10) == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}, "
                          f"D_loss: {disc_loss.item():.4f}, "
                          f"G_loss: {image_loss.item():.4f}, "
                          f"M_loss: {message_loss.item():.4f}, "
                          f"PSNR: {psnr:.2f}, "
                          f"Acc: {bit_acc:.4f}")
                
                # Add images to tensorboard periodically
                if batch_idx % (args.log_interval * 5) == 0:
                    writer.add_images('Phase2/Images/Original', images[:4], step)
                    writer.add_images('Phase2/Images/Stego', stego_images[:4], step)
                    writer.add_images(f'Phase2/Images/Noisy_{noise_type}', noisy_stego_images[:4], step)
        
        # End of epoch - calculate average losses
        avg_disc_loss = total_disc_loss / len(train_loader)
        avg_encoder_loss = total_encoder_loss / len(train_loader)
        avg_decoder_loss = total_decoder_loss / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)
        
        print(f"Phase 2 - Epoch {epoch+1} Average Losses:")
        print(f"  Discriminator: {avg_disc_loss:.6f}")
        print(f"  Encoder: {avg_encoder_loss:.6f}")
        print(f"  Decoder: {avg_decoder_loss:.6f}")
        print(f"  Combined: {avg_combined_loss:.6f}")
        
        # Run validation with error handling
        feature_analyzer.eval()
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        
        val_psnr = 0
        val_ssim = 0
        val_bit_acc = 0
        val_count = 0
        
        try:
            with torch.no_grad():
                for data in val_loader:
                    try:
                        images = data['image'].to(device)
                        messages = data['patient_data'].to(device)
                        
                        # Ensure correct shapes and value ranges
                        images = torch.clamp(images, 0, 1)
                        if messages.dim() == 3:  # [B, 1, L]
                            messages = messages.squeeze(1)  # [B, L]
                        messages = torch.clamp(messages, 0, 1)
                        
                        # Generate feature weights
                        feature_weights = feature_analyzer(images)
                        feature_weights = torch.clamp(feature_weights, 0, 1)
                        
                        # Generate stego images
                        stego_images = encoder(images, messages, feature_weights)
                        stego_images = torch.clamp(stego_images, 0, 1)
                        
                        # Calculate image quality metrics
                        val_psnr += compute_psnr(images, stego_images)
                        val_ssim += compute_ssim(images, stego_images)
                        
                        # Test on different noise types
                        noise_accuracies = []
                        for noise_type in args.noise_types:
                            try:
                                noisy_stego = noise_layer(stego_images, noise_type, noise_intensity)
                                noisy_stego = torch.clamp(noisy_stego, 0, 1)
                                decoded_msgs = decoder(noisy_stego)
                                noise_accuracies.append(compute_bit_accuracy(messages, decoded_msgs))
                            except Exception as e:
                                print(f"Warning: Error in validation with noise {noise_type}: {e}")
                                noise_accuracies.append(0.5)  # Default value on error
                        
                        # Average bit accuracy across noise types
                        val_bit_acc += sum(noise_accuracies) / len(noise_accuracies)
                        val_count += 1
                    except Exception as e:
                        print(f"Warning: Error in validation batch: {e}")
                        continue
        except Exception as e:
            print(f"Error during validation: {e}")
            # Fall back to training metrics if validation fails completely
            val_psnr = psnr
            val_ssim = ssim
            val_bit_acc = bit_acc
            val_count = 1
        
        # Calculate average validation metrics (avoid division by zero)
        if val_count > 0:
            val_psnr /= val_count
            val_ssim /= val_count
            val_bit_acc /= val_count
        
        # Log validation metrics
        writer.add_scalar('Phase2/Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Phase2/Validation/SSIM', val_ssim, epoch)
        writer.add_scalar('Phase2/Validation/BitAccuracy', val_bit_acc, epoch)
        
        print(f"Phase 2 - Validation Metrics:")
        print(f"  PSNR: {val_psnr:.2f} dB")
        print(f"  SSIM: {val_ssim:.4f}")
        print(f"  Bit Accuracy: {val_bit_acc:.4f}")
        
        # Save models periodically with error handling
        if (epoch + 1) % args.save_interval == 0:
            try:
                save_path = os.path.join(args.model_save_path, f"phase2_epoch_{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                
                torch.save(feature_analyzer.state_dict(), os.path.join(save_path, 'feature_analyzer.pth'))
                torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder.pth'))
                torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.pth'))
                torch.save(discriminator.state_dict(), os.path.join(save_path, 'discriminator.pth'))
                
                print(f"Phase 2 - Models saved to {save_path}")
            except Exception as e:
                print(f"Warning: Failed to save models: {e}")
    
    # Save final models
    try:
        final_path = os.path.join(args.model_save_path, "final_models")
        os.makedirs(final_path, exist_ok=True)
        
        torch.save(feature_analyzer.state_dict(), os.path.join(final_path, 'feature_analyzer.pth'))
        torch.save(encoder.state_dict(), os.path.join(final_path, 'encoder.pth'))
        torch.save(decoder.state_dict(), os.path.join(final_path, 'decoder.pth'))
        torch.save(discriminator.state_dict(), os.path.join(final_path, 'discriminator.pth'))
        
        print(f"Final models saved to {final_path}")
    except Exception as e:
        print(f"Warning: Failed to save final models: {e}")
        
    return feature_analyzer, encoder, decoder, discriminator

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model save directory
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        args.xray_dir, 
        args.label_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        image_size=(args.image_width, args.image_height),
        preprocess_fn=preprocess_patient_data
    )
    
    # Initialize models
    if args.use_enhanced_models:
        print("Using enhanced models with improved architecture")
        if args.use_simple_models:
            feature_analyzer = SimpleFeatureAnalyzer(in_channels=1).to(device)
        else:
            feature_analyzer = MedicalFeatureAnalyzer(in_channels=1, pretrained=args.use_pretrained).to(device)
            
        encoder = EnhancedSteganographyEncoder(image_channels=1, message_length=args.message_length).to(device)
        decoder = EnhancedSteganographyDecoder(image_channels=1, message_length=args.message_length).to(device)
        discriminator = EnhancedDiscriminator(image_channels=1, use_logits=True).to(device)  # Use logits for stability
    else:
        print("Using original model architecture")
        if args.use_simple_models:
            feature_analyzer = SimpleFeatureAnalyzer(in_channels=1).to(device)
        else:
            feature_analyzer = FeatureAnalysisDenseNet(
                in_channels=1, growth_rate=16, block_config=(3, 6, 12, 24), num_init_features=64
            ).to(device)
            
        encoder = SteganographyEncoder(image_channels=1, growth_rate=16, num_dense_layers=4).to(device)
        decoder = SteganographyDecoder(
            image_channels=1, growth_rate=16, num_layers=4, message_length=args.message_length
        ).to(device)
        discriminator = Discriminator(image_channels=1).to(device)
    
    noise_layer = EnhancedNoiseLayer().to(device)
    
    # Initialize optimizers
    optimizer_params = [
        {'params': feature_analyzer.parameters(), 'lr': args.lr}
    ]
    fa_optimizer = optim.Adam(optimizer_params, lr=args.lr)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
    
    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss()
    
    # Phase 1: Training without noise for image quality optimization
    print("\n=== Phase 1: Training for image quality without noise ===\n")
    
    for epoch in range(args.epochs_phase1):
        feature_analyzer.train()
        encoder.train()
        decoder.train()
        discriminator.train()
        
        total_disc_loss = 0
        total_encoder_loss = 0
        total_decoder_loss = 0
        total_combined_loss = 0
        
        print(f"Phase 1 - Epoch {epoch+1}/{args.epochs_phase1}")
        
        for batch_idx, data in enumerate(tqdm(train_loader)):
            images = data['image'].to(device)
            messages = data['patient_data'].to(device)
            
            # Ensure data is in valid range
            images = torch.clamp(images, 0, 1)
            
            # Ensure messages have the right dimensions
            if messages.dim() == 3:  # [B, 1, L]
                messages = messages.squeeze(1)  # [B, L]
            
            # Ensure message values are in [0,1]
            messages = torch.clamp(messages, 0, 1)
            
            # First, train discriminator
            disc_optimizer.zero_grad()
            
            # Generate feature weights
            feature_weights = feature_analyzer(images)
            feature_weights = torch.clamp(feature_weights, 0, 1)
            
            # Generate stego images
            stego_images = encoder(images, messages, feature_weights)
            stego_images = torch.clamp(stego_images, 0, 1)
            
            # Train discriminator to distinguish between real and stego images
            if hasattr(discriminator, 'use_logits') and discriminator.use_logits:
                # For discriminators that output logits
                real_logits = discriminator(images)
                fake_logits = discriminator(stego_images.detach())
                
                # Create labels with slight label smoothing
                real_labels = torch.ones_like(real_logits).to(device) * 0.9
                fake_labels = torch.zeros_like(fake_logits).to(device) * 0.1
                
                # Use BCEWithLogitsLoss for better stability
                bce_logits_loss = nn.BCEWithLogitsLoss()
                disc_loss_real = bce_logits_loss(real_logits, real_labels)
                disc_loss_fake = bce_logits_loss(fake_logits, fake_labels)
            else:
                real_preds = discriminator(images)
                fake_preds = discriminator(stego_images.detach())
                
                # Ensure values in valid range for BCE
                real_preds = torch.clamp(real_preds, 1e-7, 1 - 1e-7)
                fake_preds = torch.clamp(fake_preds, 1e-7, 1 - 1e-7)
                
                # Create labels with slight smoothing
                real_labels = torch.ones_like(real_preds).to(device) * 0.9
                fake_labels = torch.zeros_like(fake_preds).to(device) * 0.1
                
                disc_loss_real = bce_loss(real_preds, real_labels)
                disc_loss_fake = bce_loss(fake_preds, fake_labels)
            
            disc_loss = disc_loss_real + disc_loss_fake
            
            disc_loss.backward()
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            disc_optimizer.step()
            
            # Train encoder, decoder, and feature analyzer
            fa_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Generate feature weights again (for encoder training)
            feature_weights = feature_analyzer(images)
            feature_weights = torch.clamp(feature_weights, 0, 1)
            
            # Generate stego images again
            stego_images = encoder(images, messages, feature_weights)
            stego_images = torch.clamp(stego_images, 0, 1)
            
            # Decode messages from stego images (no noise in Phase 1)
            decoded_messages = decoder(stego_images)
            decoded_messages = torch.clamp(decoded_messages, 1e-7, 1 - 1e-7)
            
            # Get discriminator predictions for generator training
            if hasattr(discriminator, 'use_logits') and discriminator.use_logits:
                disc_logits = discriminator(stego_images)
                adv_loss = bce_logits_loss(disc_logits, real_labels)
            else:
                disc_preds = discriminator(stego_images)
                disc_preds = torch.clamp(disc_preds, 1e-7, 1 - 1e-7)
                adv_loss = bce_loss(disc_preds, real_labels)
            
            # Calculate losses
            # 1. Message loss - how well the decoder extracts the message
            message_loss = bce_loss(decoded_messages, messages)
            
            # 2. Image distortion loss - how similar the stego image is to the original
            # Combine MSE and SSIM for better perceptual quality
            img_mse_loss = mse_loss(stego_images, images)
            
            try:
                img_ssim_loss = ssim_loss(stego_images, images)
            except Exception as e:
                print(f"WARNING: SSIM loss error: {e}")
                img_ssim_loss = torch.tensor(0.0).to(device)
                
            image_loss = img_mse_loss + args.lambda_ssim * img_ssim_loss
            
            # Combined loss
            combined_loss = args.lambda_message * message_loss + \
                          args.lambda_image * image_loss + \
                          args.lambda_adv * adv_loss
            
            combined_loss.backward()
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(feature_analyzer.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            fa_optimizer.step()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            # Update running losses
            total_disc_loss += disc_loss.item()
            total_encoder_loss += image_loss.item()
            total_decoder_loss += message_loss.item()
            total_combined_loss += combined_loss.item()
            
            # Log every N batches
            if batch_idx % args.log_interval == 0:
                # Calculate metrics
                with torch.no_grad():
                    psnr = compute_psnr(images, stego_images)
                    ssim = compute_ssim(images, stego_images)
                    bit_acc = compute_bit_accuracy(messages, decoded_messages)
                
                # Log to tensorboard
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Phase1/Loss/Discriminator', disc_loss.item(), step)
                writer.add_scalar('Phase1/Loss/Encoder', image_loss.item(), step)
                writer.add_scalar('Phase1/Loss/Decoder', message_loss.item(), step)
                writer.add_scalar('Phase1/Loss/Combined', combined_loss.item(), step)
                writer.add_scalar('Phase1/Metrics/PSNR', psnr, step)
                writer.add_scalar('Phase1/Metrics/SSIM', ssim, step)
                writer.add_scalar('Phase1/Metrics/BitAccuracy', bit_acc, step)
                
                # Print stats periodically
                if batch_idx % (args.log_interval * 10) == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}, "
                          f"D_loss: {disc_loss.item():.4f}, "
                          f"G_loss: {image_loss.item():.4f}, "
                          f"M_loss: {message_loss.item():.4f}, "
                          f"PSNR: {psnr:.2f}")
                
                # Add images to tensorboard
                if batch_idx % (args.log_interval * 5) == 0:
                    writer.add_images('Phase1/Images/Original', images[:4], step)
                    writer.add_images('Phase1/Images/Stego', stego_images[:4], step)
                    
                    # Add feature weights visualization
                    writer.add_images('Phase1/Features/Weights', feature_weights[:4], step)
        
        # End of epoch - calculate average losses
        avg_disc_loss = total_disc_loss / len(train_loader)
        avg_encoder_loss = total_encoder_loss / len(train_loader)
        avg_decoder_loss = total_decoder_loss / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)
        
        print(f"Phase 1 - Epoch {epoch+1} Average Losses:")
        print(f"  Discriminator: {avg_disc_loss:.6f}")
        print(f"  Encoder: {avg_encoder_loss:.6f}")
        print(f"  Decoder: {avg_decoder_loss:.6f}")
        print(f"  Combined: {avg_combined_loss:.6f}")
        
        # Validation
        feature_analyzer.eval()
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        
        val_psnr = 0
        val_ssim = 0
        val_bit_acc = 0
        val_count = 0
        
        with torch.no_grad():
            for data in val_loader:
                try:
                    images = data['image'].to(device)
                    messages = data['patient_data'].to(device)
                    
                    # Ensure values in valid range
                    images = torch.clamp(images, 0, 1)
                    
                    # Ensure messages have the right dimensions
                    if messages.dim() == 3:  # [B, 1, L]
                        messages = messages.squeeze(1)  # [B, L]
                    
                    messages = torch.clamp(messages, 0, 1)
                    
                    # Generate feature weights
                    feature_weights = feature_analyzer(images)
                    feature_weights = torch.clamp(feature_weights, 0, 1)
                    
                    # Generate stego images
                    stego_images = encoder(images, messages, feature_weights)
                    stego_images = torch.clamp(stego_images, 0, 1)
                    
                    # Decode without noise
                    decoded_msgs = decoder(stego_images)
                    
                    # Calculate metrics
                    val_psnr += compute_psnr(images, stego_images)
                    val_ssim += compute_ssim(images, stego_images)
                    val_bit_acc += compute_bit_accuracy(messages, decoded_msgs)
                    val_count += 1
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if val_count > 0:
            # Calculate average validation metrics
            val_psnr /= val_count
            val_ssim /= val_count
            val_bit_acc /= val_count
            
            # Log validation metrics
            writer.add_scalar('Phase1/Validation/PSNR', val_psnr, epoch)
            writer.add_scalar('Phase1/Validation/SSIM', val_ssim, epoch)
            writer.add_scalar('Phase1/Validation/BitAccuracy', val_bit_acc, epoch)
            
            print(f"Phase 1 - Validation Metrics:")
            print(f"  PSNR: {val_psnr:.2f} dB")
            print(f"  SSIM: {val_ssim:.4f}")
            print(f"  Bit Accuracy: {val_bit_acc:.4f}")
        
        # Save models periodically
        if (epoch + 1) % args.save_interval == 0:
            try:
                save_path = os.path.join(args.model_save_path, f"phase1_epoch_{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                
                torch.save(feature_analyzer.state_dict(), os.path.join(save_path, 'feature_analyzer.pth'))
                torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder.pth'))
                torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.pth'))
                torch.save(discriminator.state_dict(), os.path.join(save_path, 'discriminator.pth'))
                
                print(f"Phase 1 - Models saved to {save_path}")
            except Exception as e:
                print(f"Error saving models: {e}")
    
    # Phase 2: Use the robust implementation
    feature_analyzer, encoder, decoder, discriminator = train_phase2_with_fixes(
        feature_analyzer=feature_analyzer,
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        noise_layer=noise_layer,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        writer=writer
    )
    
    writer.close()
    return feature_analyzer, encoder, decoder, discriminator



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train steganography models")
    
    # Data paths
    parser.add_argument('--xray_dir', type=str, required=True, help='Directory containing X-ray images')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing patient data text files')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs_phase1', type=int, default=30, help='Number of epochs for Phase 1 (image quality)')
    parser.add_argument('--epochs_phase2', type=int, default=70, help='Number of epochs for Phase 2 (robustness)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set ratio')
    
    # Loss weights
    parser.add_argument('--lambda_message', type=float, default=30.0, help='Weight for message loss')
    parser.add_argument('--lambda_image', type=float, default=5.0, help='Weight for image loss')
    parser.add_argument('--lambda_ssim', type=float, default=15.0, help='Weight for SSIM loss')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='Weight for adversarial loss')
    
    # Model configuration
    parser.add_argument('--message_length', type=int, default=256, help='Length of binary message')
    parser.add_argument('--image_width', type=int, default=512, help='Target image width')
    parser.add_argument('--image_height', type=int, default=512, help='Target image height')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained weights for feature analyzer')
    parser.add_argument('--use_simple_models', action='store_true', help='Use simplified models for faster training')
    parser.add_argument('--use_enhanced_models', action='store_true', help='Use enhanced models for training')


    
    # Noise types
    parser.add_argument('--noise_types', nargs='+', default=['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper'],
                    help='Types of noise to use during training')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for tensorboard logs')
    parser.add_argument('--model_save_path', type=str, default='./models/weights', help='Directory to save models')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging')
    parser.add_argument('--save_interval', type=int, default=5, help='How many epochs to wait before saving')
    
    args = parser.parse_args()
    train(args)