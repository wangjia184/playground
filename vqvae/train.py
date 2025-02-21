from __future__ import print_function
from six.moves import xrange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import VQVAE


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


training_data = datasets.CIFAR10(root="data", train=True, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))

training_data_denorm = (training_data.data / 255.0 - 0.5) / 0.5  # [0, 255] → [-1, 1]
data_variance = np.var(training_data_denorm)



batch_size = 128
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

num_embeddings = 512
embedding_dim = 64

commitment_cost = 0.25
decay = 0.99

learning_rate = 1e-3


training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)


validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)


    
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

writer = SummaryWriter(f'/app/runs/vqvae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

model.train()



def log_reconstructions(model, validation_loader, writer, step, device):
    """Log original and reconstructed images to TensorBoard."""
    model.eval()
    with torch.no_grad():
        # Get a batch of validation data
        data, _ = next(iter(validation_loader))
        data = data.to(device)
        
        # Forward pass to get reconstructions
        vq_loss, data_recon, _ = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        
        
        # Denormalize images (reverse Normalize(mean=0.5, std=1.0)
        data_denorm = (data * 0.5) + 0.5  # [-1, 1] → [0, 1]
        recon_denorm = (data_recon * 0.5) + 0.5
        
        # Concatenate original and reconstructed side-by-side
        combined = torch.cat([data_denorm, recon_denorm], dim=3)  # [B, C, H, W*2]
        
        # Create a grid of images (4 rows x 8 columns)
        grid = torchvision.utils.make_grid(combined, nrow=8, normalize=False)
        
        # Add to TensorBoard
        writer.add_scalar('TEST/commitment', vq_loss.item(), global_step=i+1)
        writer.add_scalar('TEST/reconstruction', recon_error.item(), global_step=i+1)
        writer.add_image('TEST/comparison', grid, step)
    model.train()



for i in xrange(num_training_updates):
    (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    writer.add_scalar('LOSS/commitment', vq_loss.item(), global_step=i+1)
    writer.add_scalar('LOSS/reconstruction', recon_error.item(), global_step=i+1)
    writer.add_scalar('perplexity', perplexity.item(), global_step=i+1)

    if (i + 1) % 1000 == 0:
        log_reconstructions(model, validation_loader, writer, i + 1, device)



