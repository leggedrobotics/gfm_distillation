import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torchvision.models import regnet_x_400mf
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops import Conv2dNormActivation
from dinov2.configs import load_and_merge_config
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils
from dinov2.data.datasets import WebDatasetVisionPNG
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from contextlib import nullcontext
import os

class DinoTeacher(nn.Module):
    def __init__(self, config_path, ckpt_path):
        super().__init__()
        conf = load_and_merge_config(config_path)
        self.backbone, _ = build_model_from_cfg(conf, only_teacher=True)
        dinov2_utils.load_pretrained_weights(self.backbone, ckpt_path, "teacher")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Run in autocast if CUDA is available
        autocast_context = torch.cuda.amp.autocast() if x.is_cuda else nullcontext()
        with autocast_context, torch.no_grad():
            feats = self.backbone.get_intermediate_layers(x, n=1, reshape=True)[0]  # (B, 1024, 16, 16)
            feats = F.avg_pool2d(feats, kernel_size=2, stride=2)  # (B, 1024, 8, 8)
        return feats

# class RegNetStudent(nn.Module):
#     def __init__(self, out_dim=1024):
#         super().__init__()
#         base_model = regnet_x_400mf(weights=None)
#         base_model.stem[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

#         self.enc = base_model.stem
#         self.enc_1 = base_model.trunk_output.block1
#         self.enc_2 = base_model.trunk_output.block2
#         self.enc_3 = base_model.trunk_output.block3

#         self.fpn = FeaturePyramidNetwork([64, 160, 400], out_channels=256)
#         self.proj = nn.Conv2d(256, out_dim, kernel_size=1)

#     def forward(self, x):
#         out = {}
#         x = self.enc(x)
#         out['feat1'] = self.enc_1(x)
#         out['feat2'] = self.enc_2(out['feat1'])
#         out['feat3'] = self.enc_3(out['feat2'])

#         x = self.fpn(out)  # all returned at 1/4 scale (16x16 for 64x64 input)
#         x = self.proj(x['feat1'])
#         return x  # shape: (B, 1024, 16, 16)

class RegNetStudent(nn.Module):
    def __init__(self, in_channel=1, out_channel=256, out_dim=1024):
        super().__init__()
        encoder = regnet_x_400mf(weights=None)
        # Remove classification head from the encoder
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        # Modify the first layer to accept the number of channels in the input image
        encoder[0][0] = nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc = encoder[0]
        self.enc_1 = encoder[1][:2]
        self.enc_2 = encoder[1][2]
        self.enc_3 = encoder[1][3]
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([64, 160, 400], out_channel)

        self.proj = nn.Conv2d(out_channel, out_dim, kernel_size=1)

    def forward(self, x):
        # check if depth has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        out = {}
        x = self.enc(x)
        out['feat1'] = self.enc_1(x)
        out['feat2'] = self.enc_2(out['feat1'])
        out['feat3'] = self.enc_3(out['feat2'])
        
        out = self.fpn(out)
        
        return self.proj(out['feat1'])

class DistillDatasetTransform(object):
    def __init__(self):
        
        self.transform = T.Compose([
            T.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            T.ToTensor(),
        ])
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
        self.downsample = T.Resize((64, 64), interpolation=T.InterpolationMode.BILINEAR)

    def __call__(self, image):
        depth_teacher = self.transform(image)
        depth_student = self.downsample(depth_teacher)
        depth_student = depth_student[:1,:,:]  # Keep only one channel for the student model

        return depth_teacher, depth_student


class DistillModule(pl.LightningModule):
    def __init__(self, teacher: DinoTeacher, student: RegNetStudent, lr=1e-4, max_steps=100000):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.max_steps = max_steps  # Set this to the number of training steps you want

    def training_step(self, batch, batch_idx):
        depths, _ = batch

        with torch.no_grad():
            target_feat = self.teacher(depths[0])

        pred_feat = self.student(depths[1])

        loss = self.loss_fn(pred_feat, target_feat)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.max_steps,  # match with max_steps
            pct_start=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # update every batch
            }
        }


if __name__ == "__main__":
    MAX_STEPS = 100000
    EXP_NAME = "distill_regnet_hm3d"
    OUTPUT_PATH = "/home/patelm/gfm/submodules/rl_nav/output"
    
    # Paths to your DINO model and image folder
    model_conf = "/home/patelm/vitl14_depth_aug_96GPU_imagenet22k/config"
    ckpt_path = "/home/patelm/vitl14_depth_aug_96GPU_imagenet22k/training_262499/teacher_checkpoint.pth"
    dataset_path = "/home/patelm/Downloads/gfm_datasets/omnidata"

    # add date and time to the output directory
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_PATH}/{EXP_NAME}/{date_time}/"

    os.makedirs(output_dir, exist_ok=True)

    wandb_logger = WandbLogger(project="rl_nav", log_model=False, name=EXP_NAME, save_dir=output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/ckpt",
        filename="step_{step}",              # You can customize this
        every_n_train_steps=5000,           # Save every 5000 steps
        save_top_k=5,                      # Save all checkpoints (change to 1 to save best only)
        monitor="train_loss",               # Monitor the training loss
        mode="min",                         # Save the best model with minimum loss
        save_last=True,                    # Save the last model
        save_on_train_epoch_end=False,      # Important: ensures it's step-based, not epoch-based
    )
    

    # depth_noise = DepthNoise(focal_length=80.0, baseline=0.12)
    teacher = DinoTeacher(model_conf, ckpt_path)
    student = RegNetStudent(out_dim=1024)
    dataset_transform = DistillDatasetTransform()
    dataset_distill = WebDatasetVisionPNG(dataset_path, transform=dataset_transform, target_transform=None)
    dataloader = DataLoader(dataset_distill.dataset, batch_size=128, num_workers=4)

    batch = next(iter(dataloader))

    model = DistillModule(teacher, student, max_steps=MAX_STEPS)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_steps=MAX_STEPS, callbacks=[checkpoint_callback],
                          logger=wandb_logger, precision="16-mixed", default_root_dir=output_dir)
    trainer.fit(model, dataloader)