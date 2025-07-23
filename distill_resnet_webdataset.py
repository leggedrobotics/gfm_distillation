from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, resnet50
from dinov2.configs import load_and_merge_config
from dinov2.models import build_model_from_cfg
from dinov2.data.datasets import WebDatasetVisionPNG
from dinov2.layers import DINOHead
from dinov2.loss import DINOLoss
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from contextlib import nullcontext
import os
import argparse
from functools import partial
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

class DinoTeacher(nn.Module):
    def __init__(self, config_path, ckpt_path):
        super().__init__()
        cfg = load_and_merge_config(config_path)
        teacher_model_dict = dict()
        backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)



        teacher_model_dict["backbone"] = backbone
        dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )

        teacher_model_dict["dino_head"] = dino_head()

        self.teacher = nn.ModuleDict(teacher_model_dict)

        # Now load the pretrained weights
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["teacher"]

        # Load each submodule
        self.teacher["backbone"].load_state_dict({
            k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")
        }, strict=True)

        self.teacher["dino_head"].load_state_dict({
            k.replace("dino_head.", ""): v for k, v in state_dict.items() if k.startswith("dino_head.")
        }, strict=True)
  
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Run in autocast if CUDA is available
        autocast_context = torch.cuda.amp.autocast() if x.is_cuda else nullcontext()
        with autocast_context, torch.no_grad():
            # Get CLS token from backbone
            cls_token = self.teacher["backbone"].forward_features(x)["x_norm_clstoken"]  # This should return the CLS token
            # Pass CLS token through DINO head
            dino_output = self.teacher["dino_head"](cls_token)
        return dino_output


class ResNetStudent(nn.Module):
    def __init__(self, teacher_cfg, in_channels=3, teacher_embed_dim=384, resnet_type='resnet18'):
        super().__init__()
        # Create ResNet backbone based on type
        if resnet_type == 'resnet18':
            self.backbone = resnet18(weights=None)
        elif resnet_type == 'resnet50':
            self.backbone = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}. Choose 'resnet18' or 'resnet50'")

        print(f"Using ResNet backbone: {resnet_type} with input channels: {in_channels} and teacher embed dim: {teacher_embed_dim}")

        cfg = load_and_merge_config(teacher_cfg)
        
        # Modify first conv layer if needed for different input channels
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get the feature dimension from ResNet (512 for resnet18, 2048 for resnet50)
        backbone_dim = self.backbone.fc.in_features
        
        # Remove the classification head
        self.backbone.fc = nn.Linear(backbone_dim, teacher_embed_dim)
        
        self.dino_head = DINOHead(
            in_dim=teacher_embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )

    def forward(self, x):
        # Get features from ResNet backbone
        features = self.backbone(x)  # Shape: (B, 512)
        # Pass through DINO head
        dino_output = self.dino_head(features)
        return dino_output


class DistillDatasetTransform(object):
    def __init__(self, crop_size=224):
        self.transform = T.Compose([
            T.RandomResizedCrop((crop_size, crop_size), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
        ])

    def __call__(self, image):
        # Both teacher and student get the same 224x224 augmented image
        transformed_image = self.transform(image)
        return transformed_image, transformed_image[:1,:,:]


class DistillModule(pl.LightningModule):
    def __init__(self, teacher: DinoTeacher, student: ResNetStudent, lr=1e-4, max_steps=100000, 
                 student_temp=0.1, teacher_temp=0.04, out_dim=65536, center_momentum=0.9):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.lr = lr
        self.max_steps = max_steps
        self.teacher_temp = teacher_temp
        
        # Initialize DINO loss with centering
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            student_temp=student_temp,
            center_momentum=center_momentum
        )

    def training_step(self, batch, batch_idx):
        depths, _ = batch
        teacher_input, student_input = depths

        # Get teacher output (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher(teacher_input)
            # Apply centering and temperature to teacher output
            teacher_softmax = self.dino_loss.softmax_center_teacher(teacher_output, self.teacher_temp)
            # Update center for next iteration
            self.dino_loss.update_center(teacher_output)
        
        # Get student output
        student_output = self.student(student_input)
        
        # Compute DINO loss
        loss = self.dino_loss([student_output], [teacher_softmax])
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            teacher_pred = torch.argmax(teacher_output, dim=-1)
            student_pred = torch.argmax(student_output, dim=-1)
            accuracy = (teacher_pred == student_pred).float().mean()
        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        depths, _ = batch
        teacher_input, student_input = depths
        
        # Get teacher output (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher(teacher_input)
            # Apply centering and temperature to teacher output
            teacher_softmax = self.dino_loss.softmax_center_teacher(teacher_output, self.teacher_temp)
            
            # Get student output
            student_output = self.student(student_input)
            
            # Compute DINO loss
            loss = self.dino_loss([student_output], [teacher_softmax])
            
            # Compute accuracy for monitoring
            teacher_pred = torch.argmax(teacher_output, dim=-1)
            student_pred = torch.argmax(student_output, dim=-1)
            accuracy = (teacher_pred == student_pred).float().mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("val/accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.max_steps,
            pct_start=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/distill_resnet_webdataset.yaml")
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
    OUT_DIM = getattr(cfg, 'out_dim', 384)
    CROP_SIZE = getattr(cfg, 'crop_size', 224)
    resnet_type = getattr(cfg, 'resnet_type', 'resnet18')  # Add this line

    model_conf = cfg.teacher.config_path
    ckpt_path = cfg.teacher.ckpt_path
    dataset_path = cfg.dataset.path
    val_dataset_path = cfg.dataset.val_path

    # Temperature parameters for distillation
    student_temp = getattr(cfg, 'student_temp', 0.1)
    teacher_temp = getattr(cfg, 'teacher_temp', 0.04)
    
    # DINO loss parameters
    dino_out_dim = getattr(cfg, 'dino_out_dim', 65536)
    # Student model configuration
    student_in_channels = getattr(cfg, 'student_in_channels', 1)
    
    # add date and time to the output directory
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

    teacher = DinoTeacher(model_conf, ckpt_path)
    student = ResNetStudent(model_conf, in_channels=student_in_channels, teacher_embed_dim=OUT_DIM)
    dataset_transform = DistillDatasetTransform(CROP_SIZE)
    dataset_distill = WebDatasetVisionPNG(dataset_path, transform=dataset_transform, target_transform=None)
    dataloader = DataLoader(dataset_distill.dataset, batch_size=batch_size, num_workers=NUM_WORKERS)

    # Validation dataset and loader
    val_dataset_distill = WebDatasetVisionPNG(val_dataset_path, transform=dataset_transform, target_transform=None)
    val_dataloader = DataLoader(val_dataset_distill.dataset, batch_size=batch_size, num_workers=NUM_WORKERS)

    model = DistillModule(
        teacher,
        student,
        max_steps=MAX_STEPS,
        lr=LEARNING_RATE,
        student_temp=student_temp,
        teacher_temp=teacher_temp,
        out_dim=dino_out_dim,
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
    # print the model
    print(model)
    trainer.fit(model, dataloader, val_dataloader, ckpt_path=resume_ckpt)