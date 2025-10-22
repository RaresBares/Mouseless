import numpy as np
import matplotlib.pyplot as plt
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# MediaPipe-21-Topologie (0=Wrist; Daumen 1–4; Zeige 5–8; Mittel 9–12; Ring 13–16; Kleiner 17–20)
SKELETON = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20)  # Little
]

SKELETON_TIPS = [
    (0,4),      # Thumb
    (0,8),      # Index
    (0,12),     # Middle
    (0,16),     # Ring
    (0,20)      # Little
]

def denorm_imgnet(x):
    x = x.clone()
    for c,(m,s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        x[c] = x[c]*s + m
    return x.clamp(0,1)

def show_image_kpts(img, kpts, normalized=True, show_skeleton=True, title=None, do_show=True):
    """
    img:  [3,H,W] Tensor in [0,1] oder ImageNet-normalized; auch [H,W,3] NumPy erlaubt
    kpts: [21,3] (x,y,v) ; x,y relativ (0..1) wenn normalized=True, sonst Pixel
    """


    # Bild nach HWC [H,W,3] in [0,1]
    if isinstance(img, torch.Tensor):
        if img.ndim == 3 and img.shape[0] == 3:
            # versuche zu erkennen, ob ImageNet-normalized (Heuristik)
            x = img
            if x.mean().abs().item() > 0.6:  # grobe Heuristik: stark „weg“ von 0..1
                x = denorm_imgnet(x)
            img_np = x.permute(1,2,0).cpu().numpy()
        else:
            img_np = img.cpu().numpy()
    else:
        img_np = np.asarray(img, dtype=np.float32)
        if img_np.max() > 1.0:
            img_np = img_np/255.0


    H, W = img_np.shape[:2]
    kpts[:, 0] *= W
    kpts[:, 1] *= H
    k = kpts.detach().cpu().numpy() if isinstance(kpts, torch.Tensor) else np.asarray(kpts, np.float32)

    xy = k[:, :2].copy()
    if normalized:  # relative → Pixel
        xy[:, 0] *= W
        xy[:, 1] *= H
    v = k[:, 2] if k.shape[1] > 2 else np.full(21, 2, dtype=np.float32)

    plt.figure(figsize=(5,5))
    plt.imshow(img_np)
    # Skelett
    if show_skeleton:
        for a,b in SKELETON:
            if v[a] > 0 and v[b] > 0:
                xa,ya = xy[a]; xb,yb = xy[b]
                plt.plot([xa,xb],[ya,yb], linewidth=2)
    vis = v > 0
    plt.scatter(xy[vis,0], xy[vis,1], s=20)
    if title: plt.title(title)
    plt.axis('off')
    if do_show:
        plt.show()

## Output: Fingertips only
def show_image_tips(img, kpts, normalized=True, show_skeleton=True, connected_tips=False, show_wrist=True, title=None):
    """
    img:  [3,H,W] Tensor in [0,1] oder ImageNet-normalized; auch [H,W,3] NumPy erlaubt
    kpts: [21,3] (x,y,v) ; x,y relativ (0..1) wenn normalized=True, sonst Pixel
    """
    WRIST_IDX = 0
    TIPS = [4, 8, 12, 16, 20]

    # Bild nach HWC [H,W,3] in [0,1]
    if isinstance(img, torch.Tensor):
        if img.ndim == 3 and img.shape[0] == 3:
            x = img
            if x.mean().abs().item() > 0.6:  # grobe Heuristik: stark „weg“ von 0..1
                x = denorm_imgnet(x)
            img_np = x.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img.cpu().numpy()
    else:
        img_np = np.asarray(img, dtype=np.float32)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

    H, W = img_np.shape[:2]
    kpts[:, 0] *= W
    kpts[:, 1] *= H
    # Keypoints nach NumPy und ggf. nach Pixel skalieren
    k = kpts.detach().cpu().numpy() if isinstance(kpts, torch.Tensor) else np.asarray(kpts, np.float32)
    xy = k[:, :2].copy()
    if normalized:  # relative → Pixel
        xy[:, 0] *= W
        xy[:, 1] *= H
    v = k[:, 2] if k.shape[1] > 2 else np.full(21, 2, dtype=np.float32)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)

    # Optional: vollständiges Skelett einzeichnen
    if show_skeleton:
        for a, b in SKELETON_TIPS:
            if v[a] > 0 and v[b] > 0:
                xa, ya = xy[a]
                xb, yb = xy[b]
                plt.plot([xa, xb], [ya, yb], linewidth=2)

    # Verbinde Fingerspitzen in Reihenfolge: Daumen->Zeige->Mittel->Ring->Kleiner
    if connected_tips:
        # Grundkette zwischen den Tips
        for a, b in zip(TIPS[:-1], TIPS[1:]):
            if v[a] > 0 and v[b] > 0:
                xa, ya = xy[a]
                xb, yb = xy[b]
                plt.plot([xa, xb], [ya, yb], linewidth=2)
        # Abschluss: je nach show_wrist an das Handgelenk oder zurück zum Daumen
        last_tip = TIPS[-1]
        first_tip = TIPS[0]
        if show_wrist and v[WRIST_IDX] > 0 and v[last_tip] > 0:
            xa, ya = xy[last_tip]
            xb, yb = xy[WRIST_IDX]
            plt.plot([xa, xb], [ya, yb], linewidth=2)
            # Zusätzlich Handgelenk mit Daumen verbinden (wenn sichtbar)
            if v[first_tip] > 0:
                xa, ya = xy[WRIST_IDX]
                xb, yb = xy[first_tip]
                plt.plot([xa, xb], [ya, yb], linewidth=2)
        else:
            # Wenn kein Wrist gezeichnet werden soll, schließe den Kreis (Kleiner -> Daumen)
            if v[last_tip] > 0 and v[first_tip] > 0:
                xa, ya = xy[last_tip]
                xb, yb = xy[first_tip]
                plt.plot([xa, xb], [ya, yb], linewidth=2)

    # Nur Fingerspitzen
    tips_vis = v[TIPS] > 0
    tips_xy = xy[TIPS][tips_vis]
    if tips_xy.size:
        plt.scatter(tips_xy[:, 0], tips_xy[:, 1], s=40)

    # Optional: Handgelenk (Wrist)
    if show_wrist and v[WRIST_IDX] > 0:
        wx, wy = xy[WRIST_IDX]
        plt.scatter([wx], [wy], s=40, marker='x')

    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()



## Wrappers of this class
def show_Circumference(img, kpts, wrist=False):
    show_image_tips(img, kpts, normalized=False, show_skeleton=False, connected_tips=True, show_wrist=wrist)

def show_Skeleton(img, kpts):
    show_image_kpts(img,kpts,normalized=False,show_skeleton=True)

def show_Tips(img, kpts, wrist=False):
    show_image_tips(img,kpts,normalized=False,show_skeleton=False,show_wrist=wrist)

def show_middle_point_wrapper(middlex, middley, skeleton=True):
    def _fn(img, kpts):
        return show_middle_point(img, kpts, middlex, middley, skeleton=skeleton)
    return _fn

def show_middle_point(img, kpts, middlex, middley, skeleton = True):
    show_image_kpts(img, kpts, normalized=False, show_skeleton=skeleton, do_show=False)

    plt.scatter(middlex*img.shape[2], middley*img.shape[1], s=240)
    plt.show()
    return plt
"""
MVP to show a picture

yolo = YoloHandKptsDataset(root="../data/raw/hand-keypoints/train", size=192)
img, kp = yolo[14]
show_image_tips(img, kp, normalized=False, show_skeleton=False, connected_tips=True, show_wrist=True)


"""
