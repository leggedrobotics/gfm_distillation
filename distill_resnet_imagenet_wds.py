import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from data.webdataset import WebDatasetImagenet as ImageNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.resnet import ResNet18, ResNet34, ResNet50
import os
from datetime import datetime
from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy

# At the top, add a mapping for teacher models and their output dims
DINO_TEACHER_MODELS = {
    'vit_s': ('dinov2_vits14', 384),
    'vit_b': ('dinov2_vitb14', 768),
    'vit_l': ('dinov2_vitl14', 1024),
}

# 1. Load DINOv2 ViT-L/14 teacher from torch.hub
class DinoV2Teacher(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).eval()  # type: ignore[attr-defined]
        for p in self.model.parameters():
            p.requires_grad = False
    def forward(self, x):
        with torch.no_grad():
            out = self.model.forward_features(x)
            return out["x_norm_clstoken"]

# 2. Data transforms
transform_train = T.Compose([
    T.Resize(256),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=45),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. LightningModule for distillation   
class DistillModule(pl.LightningModule):
    def __init__(self, teacher, student, lr=1e-3, max_steps=None):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.max_steps = max_steps

    def training_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad():
            teacher_feat = self.teacher(x)  # (B, 1024)
        student_feat = self.student(x)      # (B, 1024)
        mse = F.mse_loss(student_feat, teacher_feat)
        smooth_l1 = F.smooth_l1_loss(student_feat, teacher_feat)
        cosine_sim = F.cosine_similarity(student_feat, teacher_feat, dim=1).mean()
        self.log('train/loss', mse, prog_bar=True, sync_dist=True)
        self.log('train/mse', mse, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train/smooth_l1', smooth_l1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train/cosine_sim', cosine_sim, on_epoch=True, on_step=False, sync_dist=True)
        return mse

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad():
            teacher_feat = self.teacher(x)
        student_feat = self.student(x)
        mse = F.mse_loss(student_feat, teacher_feat)
        smooth_l1 = F.smooth_l1_loss(student_feat, teacher_feat)
        cosine_sim = F.cosine_similarity(student_feat, teacher_feat, dim=1).mean()
        self.log('val/loss', mse, prog_bar=True, sync_dist=True)
        self.log('val/mse', mse, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val/smooth_l1', smooth_l1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val/cosine_sim', cosine_sim, on_epoch=True, on_step=False, sync_dist=True)
        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.max_steps if self.max_steps is not None else 1000,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Dataset paths
    train_path = cfg.dataset.train_path
    val_path = cfg.dataset.val_path

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    max_epochs = cfg.max_epochs
    
    lr = cfg.lr
    exp_name = cfg.exp_name
    output_path = cfg.output_path
    num_gpus = getattr(cfg, 'num_gpus', 1)
    log_every_n_steps = getattr(cfg, 'log_every_n_steps', 100)


    # wandb config
    wandb_project = cfg.wandb.project
    wandb_entity = cfg.wandb.entity

    # 4. Datasets and loaders
    train_ds = ImageNet(train_path, transform=transform_train, target_transform=None, resampled=True)
    val_ds = ImageNet(val_path, transform=transform_val, target_transform=None, resampled=False, shardshuffle=False)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds.dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_ds.dataset, batch_size=batch_size, num_workers=num_workers)

    max_steps = max_epochs * (train_ds.estimated_num_samples // batch_size) // cfg.num_gpus
    print(f"Max steps: {max_steps}, Total training samples: {train_ds.estimated_num_samples}, Batch size: {batch_size}, Num GPUs: {num_gpus}")
    # 5. Logging and checkpointing
    output_dir = os.path.join(output_path, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    wandb_logger = WandbLogger(project=wandb_project, entity=wandb_entity, name=exp_name, save_dir=output_dir)
    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    # Check for last.ckpt to resume
    resume_ckpt = os.path.join(output_dir, 'last.ckpt')
    if os.path.exists(resume_ckpt):
        print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt_path = resume_ckpt
    else:
        ckpt_path = None

    limit_train_batches = (train_ds.estimated_num_samples // batch_size)
    limit_val_batches = (val_ds.estimated_num_samples // batch_size) 

    limit_train_batches = 5000 // cfg.num_gpus
    limit_val_batches = 196  // cfg.num_gpus

    print(f"Training on {num_gpus} GPUs with batch size {batch_size}. Total train batches: {limit_train_batches}, Total val batches: {limit_val_batches}")
    # 6. Model and trainer
    teacher_model_key = getattr(cfg, 'teacher_model', 'vit_l').lower()
    teacher_model_name, teacher_dim = DINO_TEACHER_MODELS[teacher_model_key]

    teacher = DinoV2Teacher(teacher_model_name)
    student_backbone = getattr(cfg, 'student_backbone', 'resnet50').lower()
    if student_backbone == 'resnet18':
        StudentNet = ResNet18
        feature_dim = 512
    elif student_backbone == 'resnet34':
        StudentNet = ResNet34
        feature_dim = 512
    elif student_backbone == 'resnet50':
        StudentNet = ResNet50
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown student_backbone: {student_backbone}")

    projector = nn.Linear(feature_dim, teacher_dim)
    student = StudentNet(pretrained=False, projector=projector)
    model = DistillModule(teacher, student, lr=lr, max_steps=max_steps)
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=num_gpus,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[ckpt_cb],
        default_root_dir=output_dir,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=cfg.max_epochs,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path) 