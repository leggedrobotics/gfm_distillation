import torch
import torch.nn as nn
import pytorch_lightning as pl
from dinov2.loss import DINOLoss, iBOTPatchLoss
from dinov2.configs import load_and_merge_config
from models.model_wrappers import DinoTeacher, StudentWrapper
from utils.depth_noise import DepthNoise
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

class MultiStudentDistillationModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # Teacher (frozen)
        teacher_cfg = cfg["teacher"]
        teacher_cfg_dino = load_and_merge_config(teacher_cfg["cfg_path"])

        self.do_ibot = cfg.loss.do_ibot
    
        self.teacher = DinoTeacher(
            config_path=teacher_cfg["cfg_path"],
            ckpt_path=teacher_cfg["ckpt_path"],
            do_ibot=self.do_ibot,
            do_dino=True,
        )
        self.teacher_temp = teacher_cfg["temperature"]

        # Build students
        self.students = nn.ModuleDict()
        for s in cfg["students"]:
            self.students[s["name"]] = StudentWrapper(
                backbone_name=s["name"],
                head_hidden_dim=s["head_hidden_dim"],
                head_bottleneck_dim=s["head_bottleneck_dim"],
                head_nlayers=s["head_nlayers"],
                teacher_cfg_path=teacher_cfg["cfg_path"],
                out_channels=s["out_channels"],
                do_ibot=self.do_ibot,
                do_dino=True,
            )

        # Losses (shared centers)
        dino_out_dim = teacher_cfg_dino.dino.head_n_prototypes
        ibot_out_dim = teacher_cfg_dino.ibot.head_n_prototypes

        loss_cfg = cfg["loss"]
        self.dino_loss = DINOLoss(
            out_dim=dino_out_dim,
            student_temp=loss_cfg["dino_student_temp"],
        )

        if self.do_ibot:
            self.ibot_patch_loss = iBOTPatchLoss(
                patch_out_dim=ibot_out_dim,
                student_temp=loss_cfg["ibot_student_temp"],
            )
            print(f"Using iBOT patch loss")
        else:
            print(f"Not using iBOT patch loss")

        train_cfg = cfg["train"]
        self.lr = train_cfg["lr"]
        self.weight_decay = train_cfg["weight_decay"]

        self.visualize = False
        self.automatic_optimization = False

        self.grad_clip_val = cfg.train.gradient_clip_val
    
    # def training_step(self, batch, batch_idx):
    #     images, _ = batch

    #     # Teacher forward once per batch
    #     teacher_out = self.teacher(images["teacher"])
    #     teacher_cls_centered = self.dino_loss.softmax_center_teacher(
    #         teacher_out["cls_token"], self.teacher_temp
    #     )
    #     teacher_patches_centered = self.ibot_patch_loss.softmax_center_teacher(
    #         teacher_out["spatial_tokens"], self.teacher_temp
    #     )

    #     # Update centers once per batch
    #     self.dino_loss.update_center(teacher_out["cls_token"])
    #     self.ibot_patch_loss.update_center(teacher_out["spatial_tokens"])

    #     # Optimizers
    #     optimizers = self.optimizers()
    #     schedulers = self.lr_schedulers()

    #     total_loss = 0.0
    #     # One independent step per student
    #     for i, (name, student) in enumerate(self.students.items()):

    #         opt = optimizers[i]
    #         sched = schedulers[i]

    #         opt.zero_grad()
    #         student_out = student(images["student"])

    #         loss_cls = self.dino_loss([student_out["cls_token"]], [teacher_cls_centered])
    #         loss_patches = self.ibot_patch_loss(student_out["spatial_tokens"], teacher_patches_centered)
    #         loss = loss_cls + loss_patches

    #         # Backward + step only for this student
    #         self.manual_backward(loss)
    #         opt.step()
    #         sched.step()

    #         # Logging
    #         self.log(f"train/{name}/cls", loss_cls, prog_bar=False, sync_dist=True)
    #         self.log(f"train/{name}/patches", loss_patches, prog_bar=False, sync_dist=True)
    #         self.log(f"train/{name}/total", loss, prog_bar=True, sync_dist=True)

    #         total_loss += loss

    #     # Optional: return average for monitoring

    #     return total_loss / len(self.students)
    
    def training_step(self, batch, batch_idx):
        images, _ = batch

        # Teacher forward once per batch
        teacher_out = self.teacher(images["teacher"])
        teacher_cls_centered = self.dino_loss.softmax_center_teacher(
            teacher_out["cls_token"], self.teacher_temp
        )
        self.dino_loss.update_center(teacher_out["cls_token"])

        if self.do_ibot:
            teacher_patches_centered = self.ibot_patch_loss.softmax_center_teacher(
                teacher_out["spatial_tokens"], self.teacher_temp
            )
            self.ibot_patch_loss.update_center(teacher_out["spatial_tokens"])

        # Optimizers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()

        total_loss = 0.0
        # One independent step per student
        for i, (name, student) in enumerate(self.students.items()):

            opt = optimizers[i]
            sched = schedulers[i]

            opt.zero_grad()
            student_out = student(images["student"])

            loss_cls = self.dino_loss([student_out["cls_token"]], [teacher_cls_centered])

            if self.do_ibot:
                loss_patches = self.ibot_patch_loss(student_out["spatial_tokens"], teacher_patches_centered)
                loss = loss_cls + loss_patches
                self.log(f"train/{name}/patches", loss_patches, prog_bar=False, sync_dist=True)
            else:
                loss = loss_cls
            # log
            self.log(f"train/{name}/loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"train/{name}/cls", loss_cls, prog_bar=False, sync_dist=True)
            

            with torch.no_grad():
                teacher_pred = torch.argmax(teacher_out["cls_token"], dim=-1)
                student_pred = torch.argmax(student_out["cls_token"], dim=-1)
                accuracy = (teacher_pred == student_pred).float().mean()
        
            self.log(f"train/{name}/accuracy", accuracy, prog_bar=False, sync_dist=True)

            # backward + step
            self.manual_backward(loss)
            clip_grad_norm_(student.parameters(), max_norm=self.grad_clip_val)  # Gradient clipping
            opt.step()
            sched.step()

            # log LR
            current_lr = opt.param_groups[0]["lr"]
            self.log(f"lr/{name}", current_lr, prog_bar=False, sync_dist=True)

            total_loss += loss.detach()

        # log avg
        total_loss = total_loss / len(self.students)
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizers, schedulers = [], []
        max_steps = self.cfg.train.max_steps
        lr = self.cfg.train.lr
        wd = self.cfg.train.weight_decay

        for name, student in self.students.items():
            opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
            sched = {
                "scheduler": OneCycleLR(
                    opt,
                    max_lr=lr,
                    total_steps=max_steps,
                    pct_start=0.1,   # 10% warmup
                    anneal_strategy="cos",
                    cycle_momentum=False,
                ),
                # "interval": "step",  # step once per training_step
                # "frequency": 1,
            }
            optimizers.append(opt)
            schedulers.append(sched)

        return optimizers, schedulers
    
    # def forward(self, x):
    #     teacher_out = self.teacher(x["teacher"])
    #     student_outs = {name: model(x["student"]) for name, model in self.students.items()}
    #     return teacher_out, student_outs
    
    # def training_step(self, batch, batch_idx):
    #     images, labels = batch  # expects preprocessed images

    #     if self.visualize:
    #         teacher_crops = images["teacher"][:8]
    #         student_crops = images["student"][:8]
    #         import matplotlib.pyplot as plt
    #         import numpy as np
    #         fig, axs = plt.subplots(2, 8, figsize=(24, 6))

    #         for i in range(8):
    #             axs[0, i].imshow(teacher_crops[i][2].cpu().numpy())
    #             axs[0, i].axis('off')
    #             axs[0, i].set_title('Teacher Crop')
    #             axs[1, i].imshow(student_crops[i][2].cpu().numpy())
    #             axs[1, i].axis('off')
    #             axs[1, i].set_title('Student Crop')
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close(fig)

    #     teacher_out, student_outs = self(images)

    #     # print(f"Teacher cls shape: {teacher_out['cls_token'].shape}")
    #     # print(f"Teacher patches shape: {teacher_out['spatial_tokens'].shape}")
    #     # for name, out in student_outs.items():
    #     #     print(f"Student {name} cls shape: {out['cls_token'].shape}")
    #     #     print(f"Student {name} patches shape: {out['spatial_tokens'].shape}")

    #     # Teacher centering
    #     teacher_cls_centered = self.dino_loss.softmax_center_teacher(
    #         teacher_out["cls_token"], self.teacher_temp
    #     )
    #     self.dino_loss.update_center(teacher_out["cls_token"])

    #     teacher_patches_centered = self.ibot_patch_loss.softmax_center_teacher(
    #         teacher_out["spatial_tokens"], self.teacher_temp
    #     )
    #     self.ibot_patch_loss.update_center(teacher_out["spatial_tokens"])

    #     # Loss per student
    #     total_loss = 0.0
    #     for name, out in student_outs.items():
    #         loss_cls = self.dino_loss([out["cls_token"]], [teacher_cls_centered])
    #         loss_patches = self.ibot_patch_loss(out["spatial_tokens"], teacher_patches_centered)
    #         loss = loss_cls + loss_patches

    #         self.log(f"train/{name}_cls", loss_cls, prog_bar=False)
    #         self.log(f"train/{name}_patches", loss_patches, prog_bar=False)
    #         self.log(f"train/{name}_total", loss, prog_bar=True)

    #         total_loss += loss

    #     total_loss /= len(student_outs)
    #     self.log("train/loss", total_loss, prog_bar=True)
    #     return total_loss

    # def configure_optimizers(self):
    #     params = []
    #     for student in self.students.values():
    #         params += list(student.parameters())
    #     optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
    #     return optimizer

    # def configure_optimizers(self):
    #     optimizers, schedulers = [], []

    #     total_steps = self.cfg.train.max_steps  # fixed number of steps

    #     for name, student in self.students.items():
    #         opt = torch.optim.AdamW(
    #             student.parameters(),
    #             lr=self.lr,
    #             weight_decay=self.weight_decay,
    #         )
    #         sched = OneCycleLR(
    #             opt,
    #             max_lr=self.lr,
    #             total_steps=total_steps,
    #             pct_start=0.1,   # 10% warmup
    #             anneal_strategy="cos",
    #             cycle_momentum=False,
    #         )
    #         optimizers.append(opt)
    #         schedulers.append(sched)

    #     return optimizers, schedulers