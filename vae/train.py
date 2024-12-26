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
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Python :", sys.version)
print("Pytorch :", torch.__version__)
print("CUDA :", torch.version.cuda)
print("Device :", device)


#######################################################################
batch_size=32
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 1



#######################################################################
# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    image_tensor = transform(image)  # Shape: [3, H, W] (RGB)

    # Add padding
    height, width = image_tensor.shape[1:3]
    left = (512 - width) // 2
    right = 512 - width - left
    top = (512 - height) // 2
    bottom = 512 - height - top
    padding = (left, right, top, bottom)  # (left, right, top, bottom)
    image_tensor = F.pad(image_tensor, padding, mode='constant', value=1)  # Change value as needed
    
    # Adjust pixel value range to [-1, 1]
    image_tensor = image_tensor * 2.0 - 1.0  # Scale to [-1, 1]
    
    return image_tensor

# Custom dataset class
class CartoonSetDataset(Dataset):
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
for i in range(1, 10):  # Directories 1 to 9 for training
    train_datasets.append(CartoonSetDataset(image_dir=f'/cartoonset100k/{i}/'))

# Combine all training datasets into one
train_dataset = torch.utils.data.ConcatDataset(train_datasets)

test_dataset = CartoonSetDataset(image_dir='/cartoonset100k/0/')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=9)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

#######################################################################


def train(model, dataloader, optimizer, prev_updates, writer=None):
    """
    Trains the model on the given data.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    model.train()  # Set the model to training mode
    
    for batch_idx, data in enumerate(tqdm(dataloader)):

        n_upd = prev_updates + batch_idx
        
        data = data.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        
        output = model.forward(data, compute_loss = True)  # Forward pass
        loss = output.loss
        
        loss.backward()
        
        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        
            print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)
            
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters
        
    return prev_updates + len(dataloader)



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
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 512, 512), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, 512, 512), global_step=cur_step)
        
        # Log random samples from the latent space
        z = torch.randn(16, 4096).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, 512, 512), global_step=cur_step)

#######################################################################

def main():
    
    if not os.path.exists('/app/models/'):
        os.makedirs('/app/models/')

    model = VAE(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter(f'/app/runs/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    prev_updates = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer)
        test(model, test_loader, prev_updates, writer=writer)

        torch.save(model.state_dict(), f'/app/models/vae_epoch_{epoch}.pth')

    torch.save(model.state_dict(), '/app/models/vae_final.pth')

if __name__ == "__main__":
    main()