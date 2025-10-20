from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize as tv_resize, InterpolationMode

class YoloHandKptsDataset(Dataset):
    def __init__(self, root, size=192, transform=None, n_kpts=21):
        root = Path(root)
        self.img_dir = root / "images"
        self.lab_dir = root / "labels"
        self.size = int(size)
        self.transform = transform
        self.n_kpts = int(n_kpts)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.items = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in exts)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        img_path = self.items[i]
        lab_path = self.lab_dir / (img_path.stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        img = tv_resize(img, [self.size, self.size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        img_t = to_tensor(img)
        if lab_path.exists():
            line = lab_path.read_text(encoding="utf-8").splitlines()[0].strip()
            vals = [float(x) for x in line.split()]
            k = np.array(vals[5:], dtype=np.float32).reshape(-1, 3)
        else:
            k = np.zeros((self.n_kpts, 3), dtype=np.float32)
        if self.transform is not None:
            img_t, k = self.transform(img_t, k)
        return img_t, torch.from_numpy(k)