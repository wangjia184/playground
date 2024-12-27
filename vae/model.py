import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/

@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.
    
    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor


class Encoder(nn.Module):
    def __init__(self, width, height, latent_dims):
        super(Encoder, self).__init__()
        self.width = width
        self.height = height
        self.latent_dims = latent_dims
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Output: [64, height/2, width/2]
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: [128, height/4, width/4]
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Output: [128, height/8, width/8]
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Output: [128, height/16, width/16]
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Output: [128, height/32, width/32]
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Output: [128, height/64, width/64]
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        
        final_height = height // 64
        final_width = width // 64
        self.flatten = nn.Flatten()
        self.mean = nn.Linear(128 * final_height * final_width, latent_dims)
        self.log_var = nn.Linear(128 * final_height * final_width, latent_dims)

    def predict(self, x):
        if x.shape[1:] != (3, self.height, self.width):
            raise ValueError(f"Input shape must be (batch_size, 3, {self.height}, {self.width}),"
                             f" but got {x.shape[1:]}.")
        
        x = self.layers(x)
        x = self.flatten(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, width, height, latent_dims):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.width = width
        self.height = height
        
        final_height = height // 64
        final_width = width // 64
        self.fc = nn.Linear(latent_dims, 128 * final_height * final_width)
        self.reshape = lambda x: x.view(-1, 128, final_height, final_width)
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [128, height/32, width/32]
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [128, height/16, width/16]
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [128, height/8, width/8]
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [64, height/4, width/4]
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [64, height/2, width/2]
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [3, height, width]
            nn.Tanh(),
        )

    def predict(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.layers(x)
        return x


class VAE(nn.Module):
    def __init__(self, device, width, height, latent_dims):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Encoder(width, height, latent_dims).to(device)
        self.softplus = nn.Softplus()
        self.decoder = Decoder(width, height, latent_dims).to(device)

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        mean, log_var = self.encoder.predict(x)

        # Adds a small value eps to the softplus-activated log-variance 
        # to better handle numerical stability by enforcing a lower bound.
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale)
        
        # Using torch.distributions for an encoder output: torch.distributions.MultivariateNormal
        # This gives access to in-built functionality for re-parameterized sampling 
        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
        
    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        
        Args:
            z (torch.Tensor): Data in the latent space.
        
        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder.predict(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.
        
        Returns:
            VAEOutput: VAE output dataclass.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)
        
        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        
        # compute loss terms 
        loss_recon = F.mse_loss(recon_x, x)

        
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                
        loss = loss_recon + loss_kl
        
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )
