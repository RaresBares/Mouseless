import os, shutil
from pathlib import Path
import kagglehub

def fetch_data(dst_root="raw/hand-keypoints", exts_img=(".jpg",".jpeg",".png"), exts_lab=(".txt",".json")):
    src = Path(kagglehub.dataset_download("riondsilva21/hand-keypoint-dataset-26k"))
    print("fetched data: ", src)
    def find_split(root, kind):
        cands = [p for p in root.rglob(kind) if p.is_dir() and (p/"train").exists() and (p/"val").exists()]
        if not cands:
            raise FileNotFoundError(f"no {kind}/train and {kind}/val dirs found under {root}")
        return cands[0]/"train", cands[0]/"val"
    img_train_src, img_val_src = find_split(src, "images")
    try:
        lab_train_src, lab_val_src = find_split(src, "labels")
    except FileNotFoundError:
        lab_train_src = lab_val_src = None

    dst_root = Path(dst_root)
    img_train_dst = dst_root/"train"/"images"
    img_val_dst   = dst_root/"val"/"images"
    lab_train_dst = dst_root/"train"/"labels"
    lab_val_dst   = dst_root/"val"/"labels"
    for d in [img_train_dst, img_val_dst, lab_train_dst, lab_val_dst]:
        d.mkdir(parents=True, exist_ok=True)

    def copy_imgs(src_dir, dst_dir):
        for p in src_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts_img:
                out = dst_dir/p.name
                shutil.copy2(p, out)
    def index_stems(dir_, exts):
        out = set()
        for p in dir_.glob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.add(p.stem)
        return out
    def copy_labels(src_dir, dst_dir, allow_stems):
        if src_dir is None:
            return
        for p in src_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts_lab and p.stem in allow_stems:
                out = dst_dir/f"{p.stem}{p.suffix}"
                shutil.copy2(p, out)

    copy_imgs(img_train_src, img_train_dst)
    copy_imgs(img_val_src, img_val_dst)

    train_stems = index_stems(img_train_dst, exts_img)
    val_stems   = index_stems(img_val_dst, exts_img)
    copy_labels(lab_train_src, lab_train_dst, train_stems)
    copy_labels(lab_val_src,   lab_val_dst,   val_stems)

    return {
        "train_images": str(img_train_dst.resolve()),
        "train_labels": str(lab_train_dst.resolve()),
        "val_images": str(img_val_dst.resolve()),
        "val_labels": str(lab_val_dst.resolve())
    }
