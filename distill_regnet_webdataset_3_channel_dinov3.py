from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torchvision.models import regnet_x_400mf
from torchvision.ops import FeaturePyramidNetwork
from dinov2.configs import load_and_merge_config
from dinov2.models import build_model_from_cfg
from data.webdataset_vision_png import WebDatasetVisionPNGMinMax
from dinov2.layers import DINOHead
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from contextlib import nullcontext
import os
import argparse
from functools import partial
from utils.depth_noise import DepthNoise
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision.ops import Conv2dNormActivation

class DinoTeacher(nn.Module):
    def __init__(self, repo_dir, backbone, weights):
        super().__init__()
        
        teacher_model_dict = dict()
        backbone = torch.hub.load(
                repo_dir, 
                backbone, 
                source='local', 
                weights=weights,
            )

        teacher_model_dict["backbone"] = backbone

        self.teacher = nn.ModuleDict(teacher_model_dict)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Run in autocast if CUDA is available
        autocast_context = torch.cuda.amp.autocast() if x.is_cuda else nullcontext()
        with autocast_context, torch.no_grad():
            feats = self.teacher["backbone"].get_intermediate_layers(x, n=1, reshape=True)[0]  # (B, 1024, 16, 16)
            feats = F.avg_pool2d(feats, kernel_size=2, stride=2)  # (B, 1024, 8, 8)
        return feats

class RegNetStudent(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, out_dim=1024):
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
            T.ToTensor(),
            T.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            # T.RandomRotation(degrees=10),
            # T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            # T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.normalize = T.Normalize(mean=[0.3347, 0.5781, 0.4711], std=[0.2514, 0.3264, 0.3328])
        self.downsample = T.Resize((64, 64), interpolation=T.InterpolationMode.BILINEAR)

    def __call__(self, image):
        depth_teacher = self.transform(image)
        depth_student = self.downsample(depth_teacher)

        return depth_teacher, depth_student


class DistillModule(pl.LightningModule):
    def __init__(self, teacher: DinoTeacher, student: RegNetStudent, lr=1e-4, max_steps=100000,
     add_noise=True, teacher_focal_length=120.0, teacher_baseline=0.12, student_focal_length=25.0, student_baseline=0.12, min_depth=0.25, max_depth=10.0, loss_type='mse'):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.lr = lr
        self.max_steps = max_steps  # Set this to the number of training steps you want
        self.add_noise = add_noise
        self.max_depth = max_depth
        self.loss_type = loss_type
        # Set loss function
        if self.loss_type == 'mse':
            self.loss_fn = None
        elif self.loss_type == 'cosine':
            self.loss_fn = None  # will use F.cosine_similarity in step
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def training_step(self, batch, batch_idx):
        depth, _ = batch

        depth_teacher = depth[0]
        depth_student = depth[1]

        with torch.no_grad():
            target_feat = self.teacher(depth_teacher)

        pred_feat = self.student(depth_student)

        # Compute metrics once
        mse = F.mse_loss(pred_feat, target_feat)
        cosine_sim = F.cosine_similarity(pred_feat, target_feat, dim=1).mean()
        if self.loss_type == 'mse':
            loss = mse
        elif self.loss_type == 'cosine':
            loss = 1 - cosine_sim

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/cosine_sim", cosine_sim, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/mse", mse, on_step=True, on_epoch=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        # plot_depth_batch(depth_teacher, depth_student, n=4)
        
        return loss


    def validation_step(self, batch, batch_idx):
        depth, _ = batch

        depth_teacher = depth[0]
        depth_student = depth[1]

        with torch.no_grad():
            target_feat = self.teacher(depth_teacher)
            pred_feat = self.student(depth_student)

        # Compute metrics once
        mse = F.mse_loss(pred_feat, target_feat)
        cosine_sim = F.cosine_similarity(pred_feat, target_feat, dim=1).mean()
        if self.loss_type == 'mse':
            loss = mse
        elif self.loss_type == 'cosine':
            loss = 1 - cosine_sim

        self.log("val/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("val/cosine_sim", cosine_sim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mse", mse, on_step=True, on_epoch=True, sync_dist=True)

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

def plot_depth_batch(teacher_batch, student_batch, n=2):
    """
    Plot n images from teacher and student depth batches side by side.
    teacher_batch, student_batch: torch.Tensor, shape (B, 1, H, W) or (B, H, W)
    """
    teacher_batch = teacher_batch.detach().cpu()
    student_batch = student_batch.detach().cpu()
    if teacher_batch.dim() == 4:
        teacher_batch = teacher_batch[:, 2]
    if student_batch.dim() == 4:
        student_batch = student_batch[:, 2]
    n = min(n, teacher_batch.shape[0], student_batch.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    for i in range(n):
        axes[i, 0].imshow(teacher_batch[i], cmap='viridis')
        axes[i, 0].set_title(f'Teacher {i}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(student_batch[i], cmap='viridis')
        axes[i, 1].set_title(f'Student {i}')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/distill_regnet_webdataset_with_noise.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    MAX_STEPS = cfg.max_steps
    EXP_NAME = cfg.exp_name
    OUTPUT_PATH = cfg.output_path
    NUM_GPUS = cfg.num_gpus
    batch_size = cfg.batch_size
    NUM_WORKERS = cfg.num_workers
    WANDB_PROJECT = cfg.wandb.project
    WANDB_ENTITY = cfg.wandb.entity
    LEARNING_RATE = cfg.learning_rate

    dataset_path = cfg.dataset.path
    val_dataset_path = cfg.dataset.val_path
    dataset_list = cfg.dataset.get('dataset_list', None)

    repo_dir = cfg.teacher.repo_dir
    backbone = cfg.teacher.backbone
    weights = cfg.teacher.weights

    # Noise params
    add_noise = cfg.noise.add_noise
    teacher_focal_length = cfg.noise.teacher.focal_length
    teacher_baseline = cfg.noise.teacher.baseline
    min_depth = cfg.noise.min_depth
    max_depth = cfg.noise.max_depth
    student_focal_length = cfg.noise.student.focal_length
    student_baseline = cfg.noise.student.baseline

    # Loss type
    loss_type = getattr(cfg, 'loss_type', 'mse')

    # add date and time to the output directory
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_PATH}/{EXP_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    ckpt_dir = os.path.join(output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.isfile(last_ckpt_path) else None

    wandb_logger = WandbLogger(project=WANDB_PROJECT, entity=WANDB_ENTITY, log_model=False, name=EXP_NAME, save_dir=output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=3,
        monitor="train/loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True,
    )

    teacher = DinoTeacher(repo_dir=repo_dir, backbone=backbone, weights=weights)
    student = RegNetStudent(in_channel=3, out_channel=64, out_dim=1024)
    dataset_transform = DistillDatasetTransform()
    dataset_distill = WebDatasetVisionPNGMinMax(dataset_path, transform=dataset_transform, target_transform=None, noise_prob=0.5, dataset_list_file=dataset_list)
    dataloader = DataLoader(dataset_distill.dataset, batch_size=batch_size, num_workers=NUM_WORKERS)

    # Validation dataset and loader
    val_dataset_distill = WebDatasetVisionPNGMinMax(val_dataset_path, transform=dataset_transform, target_transform=None, noise_prob=0.5, dataset_list_file=dataset_list)
    val_dataloader = DataLoader(val_dataset_distill.dataset, batch_size=batch_size, num_workers=NUM_WORKERS)

    model = DistillModule(
        teacher,
        student,
        max_steps=MAX_STEPS,
        add_noise=add_noise,
        teacher_focal_length=teacher_focal_length,
        teacher_baseline=teacher_baseline,
        student_focal_length=student_focal_length,
        student_baseline=student_baseline,
        min_depth=min_depth,
        max_depth=max_depth,
        lr=LEARNING_RATE,
        loss_type=loss_type,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=NUM_GPUS,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_steps=MAX_STEPS,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        precision="16-mixed",
        default_root_dir=output_dir,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )
    trainer.fit(model, dataloader, val_dataloader, ckpt_path=resume_ckpt) 