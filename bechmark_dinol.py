import torch
import time
from dinov2.configs import load_and_merge_config
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils

# Setup paths
model_conf = "/home/patelm/vitl14_depth_aug_96GPU_imagenet22k/config"
ckpt_path = "/home/patelm/vitl14_depth_aug_96GPU_imagenet22k/training_262499/teacher_checkpoint.pth"

# Load DINOv2 model
conf = load_and_merge_config(model_conf)
backbone, _ = build_model_from_cfg(conf, only_teacher=True)
dinov2_utils.load_pretrained_weights(backbone, ckpt_path, "teacher")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device).eval()
use_fp16 = True

if use_fp16:
    backbone = backbone.half()

# Benchmark helpers
def find_max_batch_size(model, device, input_shape, dtype=torch.float16):
    batch_size = 1
    max_batch = 1
    while True:
        try:
            dummy = torch.randn((batch_size, *input_shape), dtype=dtype, device=device)
            with torch.no_grad():
                _ = model.get_intermediate_layers(dummy, n=1, reshape=False)
            max_batch = batch_size
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                break
            else:
                raise e
    return max_batch

def benchmark(model, device, batch_size, input_shape, runs=50, dtype=torch.float16):
    dummy = torch.randn((batch_size, *input_shape), dtype=dtype, device=device)
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.get_intermediate_layers(dummy, n=1, reshape=False)
    torch.cuda.synchronize()
    
    # Timed run
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model.get_intermediate_layers(dummy, n=1, reshape=False)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / runs
    fps = batch_size / avg_time
    return avg_time, fps

# Main
if __name__ == "__main__":
    input_shape = (3, 224, 224)
    dtype = torch.float16 if use_fp16 else torch.float32

    print("🔍 Finding max batch size...")
    max_batch = find_max_batch_size(backbone, device, input_shape, dtype)
    print(f"✅ Max batch size: {max_batch}")

    print("🚀 Running benchmark...")
    for i in [64,128,256,512]:
        avg_time, fps = benchmark(backbone, device, i, input_shape, dtype=dtype)
        print(f"⏱️ Avg inference time: {avg_time*1000:.2f} ms for batch size {i}")
        print(f"🚀 FPS: {fps:.2f}")

    avg_time, fps = benchmark(backbone, device, max_batch, input_shape, dtype=dtype)
    print(f"⏱️ Avg inference time: {avg_time*1000:.2f} ms")
    print(f"🚀 FPS: {fps:.2f}")
