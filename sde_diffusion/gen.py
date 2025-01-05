import torch
import sys
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from datetime import datetime
from network import UNet
from sde import VariancePreservingSDE

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print("Python :", sys.version)
print("Pytorch :", torch.__version__)
print("CUDA :", torch.version.cuda)
print("Device :", device)

width = 32
height = 32


def export_image(tensor, filename):
    """
    Export a tensor as an image file.
    """
    tensor = (tensor.squeeze(0) + 1) / 2 * 255  # Scale to [0, 255]
    tensor = tensor.byte()  # Convert to uint8
    tensor = tensor.permute(1, 2, 0)  # Change shape to [H, W, C]
    image = Image.fromarray(tensor.cpu().numpy())
    image.save(filename, 'PNG')

def euler_sampling(model, sde, num_samples=1, num_steps=1000):
    """
    Generate samples using Euler method for the reverse ODE.
    
    Args:
        model (nn.Module): The trained UNet model.
        sde (VariancePreservingSDE): The SDE object.
        num_samples (int): Number of samples to generate.
        num_steps (int): Number of Euler steps.
    
    Returns:
        torch.Tensor: Generated samples.
    """
    # Initial noise (Gaussian prior)
    x = torch.randn((num_samples, 3, 32, 32)).to(device)  # Shape: [N, C, H, W]
    
    # Time steps (from T to 0)
    timesteps = torch.linspace(sde.T, sde.t_epsilon, num_steps + 1).to(device)  # Shape: [T+1]
    h = (sde.T - sde.t_epsilon) / num_steps  # Step size
    
    # Euler method
    for i in tqdm(range(num_steps), desc="Sampling"):
        t = timesteps[i]
        
        # Predict the score (score function)
        with torch.no_grad():
            score = model(x, t.expand(num_samples))  # Shape: [N, C, H, W]
        
        # Compute drift and diffusion terms
        drift = sde.f(t, x)  # Drift term
        diffusion = sde.g(t, x)  # Diffusion term
        
        # Update x using Euler step
        x = x + h * (drift - 0.5 * diffusion**2 * score)
    
    return x

def main():
    
    if not os.path.exists('/app/models/sde_final.pth'):
        print("`/app/models/sde_final.pth` does not exist")
        return
    
    sde = VariancePreservingSDE()
    model = UNet(
            input_channels=3,
            input_height=32,
            ch=128,
            ch_mult=(1, 2, 2, 2),
            num_res_blocks=2,
            attn_resolutions=(16,),
            resamp_with_conv=True,
            dropout=0,
            )
    model.load_state_dict(torch.load('/app/models/sde_final.pth', weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Generate samples
    num_samples = 1  # Number of images to generate
    num_steps = 1000  # Number of Euler steps
    generated_images = euler_sampling(model, sde, num_samples, num_steps)
    
    # Save the generated images
    if not os.path.exists('/app/generated/'):
        os.makedirs('/app/generated/')
    
    for i in range(num_samples):
        export_image(generated_images[i], f'/app/generated/sample_{i}.png')
        print(f"Saved generated image to /app/generated/sample_{i}.png")

    

if __name__ == "__main__":
    main()