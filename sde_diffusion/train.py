import torch
import sys
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from datetime import datetime

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from sde import VE_SDE, VP_SDE, SubVP_SDE
from network import create_improved_unet
import xformers

# Print the version of xformers


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print("Python :", sys.version)
print("Pytorch :", torch.__version__)
print("CUDA :", torch.version.cuda)
print("Device :", device)
print("xformers version :", xformers.__version__)

#######################################################################
batch_size=32
learning_rate = 2e-4
weight_decay = 1e-12
ema_decay=0.999
num_epochs = 10
width = 32
height = 32


def export_image(tensor, filename):
    tensor = (tensor.squeeze(0) + 1) / 2 * 255  # Shape becomes [3, width, height]
    tensor = tensor.byte()  

    tensor = tensor.permute(1, 2, 0)  # Shape becomes [width, height, 3]

    image = Image.fromarray(tensor.cpu().numpy())
    image.save(filename, 'PNG')


#######################################################################
# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height))

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)  # Shape: [3, width, height] (RGB)

    image_tensor = image_tensor * 2.0 - 1.0  # Scale to [-1, 1]
    
    return image_tensor

"""
# Custom dataset class
class Cifar10Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        return preprocess_image(img_path)

# Create datasets
train_datasets = []
for i in range(0, 1):  # Directories 1 to 9 for training
    train_datasets.append(Cifar10Dataset(image_dir=f'/cifar10-64/train/class{i}/'))

# Combine all training datasets into one
train_dataset = torch.utils.data.ConcatDataset(train_datasets)

test_datasets = []
for i in range(0, 10):  # Directories 1 to 9 for training
    test_datasets.append(Cifar10Dataset(image_dir=f'/cifar10-64/test/class{i}/'))

test_dataset = torch.utils.data.ConcatDataset(test_datasets)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=9)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

#######################################################################
eps=1e-5
"""



# Modified training function
def train(model, sde, dataloader, optimizer, ema, writer=None):
    """
    Trains the model on the given data.
    
    Args:
        model (nn.Module): The model to train.
        sde: The SDE object.
        dataloader (torch.utils.data.DataLoader): The data loader.
        optimizer: The optimizer.
        prev_updates (int): The number of previous updates.
        ema (EMA): The EMA object.
        writer: The TensorBoard writer.
    """
    model.train()  # Set the model to training mode

    n_upd = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        epoch_loss = 0.
        for data, _ in tqdm(dataloader):
            n_upd += 1

            optimizer.zero_grad() # Zero the gradients
            x0 = data.to(device) # Shape = [N, C, W, H]

            loss = sde.loss(model, x0)
            loss.backward()

            if n_upd % 100 == 0:
                # Calculate and log gradient norms
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
            
                print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} Grad: {total_norm:.4f}')

                if writer is not None:
                    global_step = n_upd
                    writer.add_scalar('Loss/Train', loss.item(), global_step)
                    writer.add_scalar('GradNorm/Train', total_norm, global_step)
                    
            optimizer.step()
            ema.update()
            epoch_loss += loss.item()
        
        with ema.average_parameters():
            torch.save(model.state_dict(), f'/app/ckpts/{epoch}.ckpt')
        





#######################################################################


def test(model, dataloader, cur_step, writer=None):
    """
    Tests the model on the given data.
    
    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            
            output = model.forward(data, compute_loss=True)  # Forward pass
            
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
            
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    
    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)
        
        # Log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, width, height), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, width, height), global_step=cur_step)
        
        # Log random samples from the latent space
        z = torch.randn(16, latent_dims).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, width, height), global_step=cur_step)

#######################################################################

# Modified main function
def main():
    if not os.path.exists('/app/ckpts/'):
        os.makedirs('/app/ckpts/')

    train_data = FashionMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Define the UNet2DModel with parameters matching your custom U-Net
    model = create_improved_unet().to(device)

    sde = VP_SDE(rescale=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    # Define TensorBoard writer
    writer = SummaryWriter(f'/app/runs/sde_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    train( model, sde, train_dataloader, optimizer, ema, writer)

if __name__ == "__main__":
    main()