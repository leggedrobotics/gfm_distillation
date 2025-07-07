import os
import torch
import argparse
from omegaconf import OmegaConf
from distill_regnet_webdataset_with_noise import RegNetStudent, DinoTeacher, DistillModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Determine checkpoint and output paths
    exp_name = cfg.exp_name
    output_dir = os.path.join(cfg.output_path, exp_name)
    ckpt_path = os.path.join(output_dir, 'ckpt', 'last.ckpt')
    output_path = os.path.join(output_dir, f'student_{exp_name}.pth')

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load model components from config
    teacher = DinoTeacher(cfg.teacher.config_path, cfg.teacher.ckpt_path)
    student = RegNetStudent(out_dim=1024)
    model = DistillModule(
        teacher,
        student,
        max_steps=cfg.max_steps,
        add_noise=cfg.noise.add_noise,
        teacher_focal_length=cfg.noise.teacher.focal_length,
        teacher_baseline=cfg.noise.teacher.baseline,
        student_focal_length=cfg.noise.student.focal_length,
        student_baseline=cfg.noise.student.baseline,
        min_depth=cfg.noise.min_depth,
        max_depth=cfg.noise.max_depth,
        lr=getattr(cfg, 'learning_rate', 1e-4),
        loss_type=getattr(cfg, 'loss_type', 'mse'),
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Save only the student model
    torch.save(model.student.state_dict(), output_path)
    print(f"Student model weights saved to {output_path}") 