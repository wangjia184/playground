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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Python :", sys.version)
print("Pytorch :", torch.__version__)
print("CUDA :", torch.version.cuda)
print("Device :", device)


def generate_single_image(model) :
    z = torch.randn(1, 4096).to(device)
    tensor = model.decode(z)  # [NCWH] = [1, 3, 512, 512]

    tensor = (tensor.squeeze(0) + 1) / 2 * 255  # Shape becomes [3, 512, 512]
    tensor = tensor.byte()  

    # Rearrange the tensor to [512, 512, 3] for PIL
    tensor = tensor.permute(1, 2, 0)  # Shape becomes [512, 512, 3]

    image = Image.fromarray(tensor.cpu().numpy())
    image.save('output.png', 'PNG')


def main():
    
    if not os.path.exists('/app/models/vae_final.pth'):
        print("`/app/models/vae_final.pth` does not exist")
        return

    model = VAE(device)
    model.load_state_dict(torch.load('/app/models/vae_final.pth', weights_only=True))
    model.eval()
 
    generate_single_image(model)
    

if __name__ == "__main__":
    main()