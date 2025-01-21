import torch
import sys
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from datetime import datetime
from sde import VE_SDE, VP_SDE, SubVP_SDE
from network import create_improved_unet

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


def main():
    
    checkpoint_path = '/app/ckpts/9.ckpt'
    if not os.path.exists(checkpoint_path):
        print("{checkpoint_path} does not exist")
        return
    
    model = create_improved_unet()

    sde = VP_SDE(eps=1e-3, rescale=True).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Generate samples
    samples = sde.predictor_corrector_sample(model, (32, 1, 28, 28), device, n_lang_steps=2)
    samples = torch.clamp(samples, 0, 1)


    
    # Save the generated images
    if not os.path.exists('/app/generated/'):
        os.makedirs('/app/generated/')
    
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image

    fig = plt.figure(figsize=(20, 6))
    for i in range(1, 15):
        fig.add_subplot(2, 7, i)
        plt.axis('off')
        plt.imshow(to_pil_image(samples[i]))
        plt.tight_layout()
    plt.savefig('/app/generated/samples.png')

    

if __name__ == "__main__":
    main()