from pathlib import Path
from PIL import Image
import torch
from numpy import resize
from torchvision.transforms.functional import to_tensor, resize
from torch.utils.data import Dataset, DataLoader
import numpy as np

class YoloHandKptsDataset(Dataset):
    def __init__(self, root, size=192):
        self.img_dir = Path(root) / "images"
        self.lab_dir = Path(root) / "labels"
        self.size = size
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.items = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in exts])
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        img_path = self.items[i]
        lab_path = self.lab_dir / (img_path.stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        img = resize(img, [self.size, self.size])
        img_t = to_tensor(img)
        if lab_path.exists():
            line = lab_path.read_text(encoding="utf-8").strip().splitlines()[0]
            vals = [float(x) for x in line.split()]
            k = np.array(vals[5:], dtype=np.float32).reshape(-1, 3)
        else:
            k = np.zeros((21, 3), dtype=np.float32)
        k_t = torch.from_numpy(k)
        return img_t, k_t

def make_loaders(root="raw/hand-keypoints", bs=32, size=192, workers=4):
    train_ds = YoloHandKptsDataset(Path(root)/"train", size=size)
    val_ds   = YoloHandKptsDataset(Path(root)/"val",   size=size)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_dl, val_dl

#train_loader, val_loader = make_loaders()


# Dimension: [Batch][Img = 0][Channel R/G/B][Row][Column] [Batch][Data = 1][Keypoints][x,y,visibility]
#yolo = YoloHandKptsDataset(root="raw/hand-keypoints/train", size=192)
#print(yolo[3][0].shape)
