# https://drscotthawley.github.io/blog/posts/FlowModels.html

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from torchvision.utils import save_image

from einops import rearrange 



from utils import default, exists, divisible_by

from functools import partial
from flow import RectifiedFlow
from network import create_unet

# dataset classes

from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as T

from PIL import Image

from torch.optim import AdamW
from accelerate import Accelerator  # https://pypi.org/project/accelerate/
from torch.utils.data import DataLoader
from ema_pytorch import EMA

class ImageDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.is_dir()

        self.folder = folder
        self.image_size = image_size

        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image

            return image.convert(img_type)

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer



def cycle(dl):
    while True:
        for batch in dl:
            yield batch

class Trainer(Module):
    def __init__(
        self,
        rectified_flow: dict | RectifiedFlow,
        *,
        dataset: dict | Dataset,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size = 32,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema = True
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        if isinstance(dataset, dict):
            dataset = ImageDataset(**dataset)

        if isinstance(rectified_flow, dict):
            rectified_flow = RectifiedFlow(**rectified_flow)

        self.model = rectified_flow

        # determine whether to keep track of EMA (if not using consistency FM)
        # which will determine which model to use for sampling

        use_ema &= not self.model.use_consistency

        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)


        # optimizer, dataloader, and all that

        self.optimizer = AdamW(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps

        # folders

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict() if self.ema_model is not None else None,
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return
        
        load_package = torch.load(path)
        
        self.model.load_state_dict(load_package["model"])
        self.ema_model.load_state_dict(load_package["ema_model"])
        self.optimizer.load_state_dict(load_package["optimizer"])

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def sample(self, fname):
        eval_model = default(self.ema_model, self.model)
        dl = cycle(self.dl)
        mock_data = next(dl)
        data_shape = mock_data.shape[1:]

        with torch.no_grad():
            sampled = eval_model.sample(batch_size=self.num_samples, data_shape=data_shape)
      
        sampled = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row = self.num_sample_rows)
        sampled.clamp_(0., 1.)

        save_image(sampled, fname)
        return sampled

    def forward(self):

        dl = cycle(self.dl)

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            data = next(dl)
            loss, loss_breakdown = self.model(data, return_loss_breakdown = True)

            self.log(loss_breakdown._asdict(), step = step)

            self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.model.use_consistency:
                self.model.ema_model.update()

            if self.is_main and self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:

                if divisible_by(step, self.save_results_every):

                    sampled = self.sample(fname=str(self.results_folder / f'results.{step}.png'))

                    self.log_images(sampled, step = step)

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')


def main():
    model = create_unet()

    rectified_flow = RectifiedFlow(
        model,
        loss_fn = 'pseudo_huber',
        use_consistency = False
    )

    img_dataset = ImageDataset(
        folder = './data',
        image_size = 64
    )

    trainer = Trainer(
        rectified_flow,
        dataset = img_dataset,
        num_train_steps = 50_000,
        results_folder = './.results',
        accelerate_kwargs = dict(mixed_precision = 'fp16'),
        adam_kwargs = dict(weight_decay = 1e-12),
        use_ema = True
    )

    trainer()

if __name__ == "__main__":
    main()