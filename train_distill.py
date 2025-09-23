import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch

from models.multi_student_module import MultiStudentDistillationModule
from data.webdataset_vision_png import WebDatasetVisionPNG
from data.augmentations_depth import DataAugmentationDINODepthNorm
import argparse

class StepCheckpoint(pl.Callback):
    """Custom checkpointing every N steps, plus student export."""
    def __init__(self, cfg):
        self.checkpoint_every = cfg.train.checkpoint_every
        self.export_students_every = cfg.train.export_students_every
        self.output_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
        self.num_students = len(cfg.students)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step // self.num_students  # Hack to fix the global step counting with multiple students

        # Regular Lightning checkpoint
        if global_step > 0 and global_step % self.checkpoint_every == 0:
            ckpt_path = os.path.join(self.output_dir, f"last.ckpt")
            trainer.save_checkpoint(ckpt_path)

        # Save each student separately
        if global_step > 0 and global_step % self.export_students_every == 0:
            for name, student in pl_module.students.items():
                student_path = os.path.join(self.output_dir, "students", f"{name}_step-{global_step}.pth")
                torch.save(student.state_dict(), student_path)

def main():

    parser = argparse.ArgumentParser(description="Multi-Student Distillation Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multi_student_distillation.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    cfg.train.log_every = cfg.train.log_every * len(cfg.students)  # Hack to fix the log_every counting with multiple students

    num_students = len(cfg.students)

    output_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "students"), exist_ok=True)

    # Resume from last checkpoint if exists
    ckpts = [f for f in os.listdir(output_dir) if f.endswith("last.ckpt")]
    # ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1].split(".")[0])) if ckpts else []
    resume_ckpt = os.path.join(output_dir, ckpts[-1]) if ckpts else None

    # Wandb
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.experiment.name,
    )


    data_transform = DataAugmentationDINODepthNorm(
        global_crop_scale=(0.32, 1.0),
        teacher_global_crop_size=224,
        student_global_crop_size=256,
    )
    webdataset = WebDatasetVisionPNG(root=cfg.data.root, transform=data_transform, noise_prob=0.1)

    dataloader = DataLoader(webdataset.dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    # Module
    model = MultiStudentDistillationModule(cfg)

    # Trainer config
    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=cfg.trainer.precision,
        max_steps=cfg.train.max_steps * num_students,  # Hack to Fix the num_steps counting with multiple students
        log_every_n_steps=cfg.train.log_every,
        logger=wandb_logger,
        callbacks=[StepCheckpoint(cfg)],
    )

    trainer.fit(model, dataloader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
