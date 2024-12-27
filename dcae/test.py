# build DC-AE models
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
from efficientvit.ae_model_zoo import dc_ae_f32c32
from efficientvit.models.efficientvit.dc_ae import DCAE
from safetensors.torch import load_file

class DCAESafeTensor(DCAE):
    def load_model(self):
        state_dict = load_file(self.cfg.pretrained_path)
        self.load_state_dict(state_dict)

config = dc_ae_f32c32("dc-ae-f32c32-in-1.0", f"/dc-ae-f32c32-in-1.0/model.safetensors")
dc_ae = DCAESafeTensor(config)

# encode
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.apps.utils.image import DMCrop

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
dc_ae = dc_ae.to(device).eval()

transform = transforms.Compose([
    DMCrop(512), # resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
image = Image.open("girl.png")
x = transform(image)[None].to(device)
latent = dc_ae.encode(x)
print(latent.shape)

# decode
y = dc_ae.decode(latent)
save_image(y * 0.5 + 0.5, "reconstructed.png")