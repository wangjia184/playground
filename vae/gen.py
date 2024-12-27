import torch
import sys
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from datetime import datetime
from model import VAE

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print("Python :", sys.version)
print("Pytorch :", torch.__version__)
print("CUDA :", torch.version.cuda)
print("Device :", device)

width = 256
height = 256
latent_dims = 64*64

def generate_single_image(model) :
    z = torch.randn(1, latent_dims).to(device)
    tensor = model.decode(z)  # [NCWH] = [1, 3, width, height]

    tensor = (tensor.squeeze(0) + 1) / 2 * 255  # Shape becomes [3, width, height]
    tensor = tensor.byte()  

    # Rearrange the tensor to [width, height, 3] for PIL
    tensor = tensor.permute(1, 2, 0)  # Shape becomes [width, height, 3]

    image = Image.fromarray(tensor.cpu().numpy())
    image.save('output.png', 'PNG')


def generate_animation(model, num_frames=30, output_filename='animation.gif'):
    frames = []
    
    # Sample start and end latent vectors
    z_start = torch.randn(1, latent_dims).to(device)
    z_end = torch.randn(1, latent_dims).to(device)
    
    for i in range(num_frames):
        # Interpolate between the start and end latent vectors
        alpha = i / (num_frames - 1)  # Normalize i to [0, 1]
        z = (1 - alpha) * z_start + alpha * z_end
        
        tensor = model.decode(z)  # [NCWH] = [1, 3, width, height]
        tensor = (tensor.squeeze(0) + 1) / 2 * 255  # Shape becomes [3, width, height]
        tensor = tensor.byte()  

        # Rearrange the tensor to [width, height, 3] for PIL
        tensor = tensor.permute(1, 2, 0)  # Shape becomes [width, height, 3]

        image = Image.fromarray(tensor.cpu().numpy())
        frames.append(image)

    # Save frames as an animated GIF
    frames[0].save(output_filename, save_all=True, append_images=frames[1:], loop=0, duration=100)

# Example usage:
# generate_animation(model, latent_dims, device)

def main():
    
    if not os.path.exists('/app/models/vae_final.pth'):
        print("`/app/models/vae_final.pth` does not exist")
        return

    model = VAE(device, width, height, latent_dims)
    model.load_state_dict(torch.load('/app/models/vae_final.pth', weights_only=True))
    model.eval()
 
    generate_animation(model)
    

if __name__ == "__main__":
    main()