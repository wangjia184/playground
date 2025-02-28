from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from torchinfo import summary
from torch.amp import autocast, GradScaler

# Import your model components
from netwk import Encoder, Decoder
from quant import VectorQuantizer2
from vqvae import VQVAE


# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
batch_size = 64
num_training_updates = 50000
learning_rate = 1e-3
commitment_cost = 0.25  # This is beta in your VQVAE

# Image transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load datasets
train_dataset = ImageFolder(root='/data/train/', transform=transform)
test_dataset = ImageFolder(root='/data/test/', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=20)
val_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True)

def calculate_variance(loader):
    """Calculate variance using parallel batch processing"""
    print("Calculating data variance with parallel processing...")
    
    # Initialize accumulators
    total_sum = 0.0
    total_sq = 0.0
    total_pixels = 0
    
    # Process batches in parallel
    with mp.Pool(processes=8) as pool:  # Match num_workers
        results = []
        for batch, _ in loader:
            # Submit batch processing to pool
            future = pool.apply_async(process_batch, (batch,))
            results.append(future)
        
        # Collect results
        for future in results:
            batch_sum, batch_sq, batch_pixels = future.get()
            total_sum += batch_sum
            total_sq += batch_sq
            total_pixels += batch_pixels
    
    # Calculate final variance
    mean = total_sum / total_pixels
    variance = (total_sq / total_pixels) - (mean ** 2)
    return variance

def process_batch(batch):
    """Process a single batch"""
    denorm_batch = (batch * 0.5) + 0.5  # Denormalize on CPU
    batch_sum = denorm_batch.sum().item()
    batch_sq = (denorm_batch ** 2).sum().item()
    batch_pixels = batch.numel()  # batch_size * C * H * W
    return batch_sum, batch_sq, batch_pixels

# Calculate variance using parallel processing
data_variance = calculate_variance(train_loader)
print(f"Data variance: {data_variance:.4f}")



def log_reconstructions(model, val_loader, writer, step, device):
    """Log original and reconstructed images to TensorBoard."""
    model.eval()
    with torch.no_grad():
        # Get a batch of validation data
        data, _ = next(iter(val_loader))
        data = data.to(device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast( 'cuda', enabled=False):
            reconstructed, _, vq_loss = model(data)
        
        # Convert reconstructions to FP32 for visualization
        reconstructed = reconstructed.float()
        recon_error = F.mse_loss(reconstructed, data) / data_variance
        
        # Denormalize images (ensure FP32 for image logging)
        data_denorm = (data * 0.5) + 0.5  # Already FP32
        recon_denorm = (reconstructed * 0.5) + 0.5  # Now FP32
        
        # Create comparison grid
        combined = torch.cat([data_denorm, recon_denorm], dim=3)
        grid = torchvision.utils.make_grid(combined, nrow=8, normalize=False)
        
        # Log to TensorBoard
        writer.add_scalar('TEST/commitment', vq_loss.item(), global_step=step)
        writer.add_scalar('TEST/reconstruction', recon_error.item(), global_step=step)
        writer.add_image('TEST/comparison', grid, step)
    model.train()

# Initialize model with your architecture parameters
model = VQVAE(
    vocab_size=4096,
    z_channels=32,
    ch=128,
    dropout=0.0,
    beta=commitment_cost,
    quant_conv_ks=3,
    quant_resi=0.5,
    share_quant_resi=4,
    default_qresi_counts=0,
    ch_mult=(1, 1, 2, 2, 4),
    v_patch_nums=(1, 2, 3, 4),
    test_mode=False
).to(device)

summary(model.encoder, input_size=(1, 3, 64, 64))

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
writer = SummaryWriter(f'runs/vqvae_train_{datetime.now().strftime("%Y%m%d-%H%M%S")}')



# Training loop
model.train()
for step in range(num_training_updates):
 
    data, _ = next(iter(train_loader))
    data = data.to(device)
    
    optimizer.zero_grad()

    reconstructed, _, vq_loss = model(data)
    
    if torch.isnan(reconstructed).any():
        print("NaN detected in reconstructions, skipping batch")
        continue

    recon_error = F.mse_loss(reconstructed, data) / (data_variance + 1e-6)
    total_loss = recon_error + vq_loss

    if torch.isnan(total_loss).any():
        print("NaN detected in loss, skipping batch")
        continue
 
    
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Check gradients before step
    grad_norm = torch.norm(torch.stack([
        torch.norm(p.grad.detach(), 2) 
        for p in model.parameters() 
        if p.grad is not None
    ]), 2)
    
    if torch.isnan(grad_norm):
        print("NaN gradients detected, skipping update")
        optimizer.zero_grad()
        continue
    
    optimizer.step()
    
    # Log training metrics
    writer.add_scalar('GradNorm', grad_norm.item(), step)
    writer.add_scalar('LOSS/total', total_loss.item(), step)
    writer.add_scalar('LOSS/reconstruction', recon_error.item(), step)
    writer.add_scalar('LOSS/commitment', vq_loss.item(), step)
    
    # Log reconstructions periodically
    if (step + 1) % 100 == 0:
        log_reconstructions(model, val_loader, writer, step + 1, device)
        print(f"Step {step+1}/{num_training_updates} | Loss: {total_loss.item():.4f}")

# Save final model
torch.save(model.state_dict(), 'vqvae_final.pth')
writer.close()

