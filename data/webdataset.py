import os
import io
import json
import glob
import random
from pathlib import Path
from PIL import Image
from typing import Optional, Callable

import webdataset as wds
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import numpy as np
import webdataset as wds

class WebDatasetImagenet(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        images_per_shard: int = 3000,
        shard_pattern: str = "*.tar",
        shuffle_buffer: int = 1000,
        resampled: bool = True,
        shardshuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.images_per_shard = images_per_shard

        random.seed(seed)

        # 🔍 Gather shard files
        contains_subfolders = any(os.path.isdir(os.path.join(root, entry)) for entry in os.listdir(root))
        if contains_subfolders:
            self.shard_files = glob.glob(os.path.join(root, "**", shard_pattern), recursive=True)
        else:
            self.shard_files = glob.glob(os.path.join(root, shard_pattern))

        self.num_shards = len(self.shard_files)
        self.estimated_num_samples = self.num_shards * images_per_shard

        print(f"📦 Loaded {self.num_shards} shards ({self.estimated_num_samples} estimated samples) from '{root}'.")

        # 🔀 Shuffle shard files if desired
        if shardshuffle:
            random.shuffle(self.shard_files)

        # ✅ WebDataset pipeline for JPEG + JSON
        self.dataset = (
            wds.WebDataset(self.shard_files, resampled=resampled, nodesplitter=wds.split_by_node)
            .shuffle(shuffle_buffer)
            .to_tuple("jpg", "json")
            .map(self.process_sample)
        )

    def process_sample(self, sample):
        """Process a single sample: decode JPEG and extract class label."""
        jpg_data, json_data = sample

        image = self.decode_jpg(jpg_data)
        metadata = self.safe_json_decode(json_data)

        # Extract class from metadata
        target = metadata.get("class", "unknown")

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def decode_jpg(jpg_bytes):
        """Decode JPEG bytes into PIL Image."""
        try:
            with io.BytesIO(jpg_bytes) as f:
                img = Image.open(f)
                img = img.convert("RGB")
                return img
        except Exception as e:
            print(f"⚠️ JPEG decode failed: {e} — Returning blank image.")
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    @staticmethod
    def safe_json_decode(json_bytes):
        """Decode JSON metadata bytes safely."""
        try:
            json_str = json_bytes.decode("utf-8", errors="ignore")
            return json.loads(json_str)
        except Exception as e:
            print(f"⚠️ JSON decode failed: {e}")
            return {}

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.estimated_num_samples
