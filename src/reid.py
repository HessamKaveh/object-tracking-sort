"""Appearance embeddings for the tracker: a generic CNN, a foundation model, and a MOT17 ReID network.

Three embedders, all producing L2-normalised vectors for person crops:

* ``imagenet``  ResNet-18 ImageNet features (global average pool), zero-shot.
* ``dinov2``    DINOv2 ViT-S/14 class token, zero-shot (a self-supervised foundation model).
* ``reid``      ResNet-18 trained for person re-identification on MOT17 with a **cross-fold split by
                sequence**: the first and second half of a MOT17 sequence show the *same* people, so a ReID
                model trained on train halves would have seen every val-half identity. Instead
                fold A = {02, 04, 05, 09} and fold B = {10, 11, 13}; the model trained on fold B embeds
                fold A sequences and vice versa, so no sequence is embedded by a network that saw it.

    python src/reid.py crops                        # cache GT crops of both folds
    python src/reid.py train --fold A               # model that will embed fold A (trained on fold B)
    python src/reid.py train --fold B
    python src/reid.py embed --dets yolo11s_mot17 public_frcnn oracle_gt
"""
import argparse
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FOLDS = {"A": ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09"], "B": ["MOT17-10", "MOT17-11", "MOT17-13"]}
H, W = 192, 96
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EMBEDDERS = ["imagenet", "dinov2", "reid"]


def fold_of(seq):
    return "A" if seq in FOLDS["A"] else "B"


def other(fold):
    return "B" if fold == "A" else "A"


def crop(img, box, h=H, w=W):
    """Crop a box (xyxy, clipped to the image) and resize to h x w RGB uint8."""
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = int(np.clip(x1, 0, iw - 1)), int(np.clip(y1, 0, ih - 1))
    x2, y2 = int(np.clip(np.ceil(x2), x1 + 1, iw)), int(np.clip(np.ceil(y2), y1 + 1, ih))
    return cv2.cvtColor(cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def to_tensor(crops, size=None):
    x = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float().div_(255)
    if size is not None:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    m, s = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    return (x - m) / s


# ------------------------------------------------------------------------------------------------ crops
def build_crops(every=3, min_vis=0.3):
    """GT pedestrian crops (visibility >= min_vis, every `every`-th frame) of each fold -> cache/reid_crops_{fold}.npz."""
    os.makedirs(CACHE, exist_ok=True)
    for fold, seqs in FOLDS.items():
        ims, pids, cams = [], [], []
        for si, seq in enumerate(seqs):
            g = mot.load_gt(seq)
            g = g[(g[:, 7] == mot.PEDESTRIAN) & (g[:, 6] != 0) & (g[:, 8] >= min_vis)]
            for f in range(1, mot.seq_info(seq)["length"] + 1, every):
                r = g[g[:, 0] == f]
                if not len(r):
                    continue
                img = cv2.imread(mot.frame_path(seq, f))
                for row in r:
                    b = row[2:6].copy(); b[2:] += b[:2]
                    ims.append(crop(img, b)); pids.append(f"{seq}:{int(row[1])}"); cams.append(f)
            print(fold, seq, len(ims), flush=True)
        uniq = {p: i for i, p in enumerate(sorted(set(pids)))}
        np.savez(os.path.join(CACHE, f"reid_crops_{fold}.npz"), x=np.stack(ims),
                 y=np.array([uniq[p] for p in pids]), frame=np.array(cams))
        print(fold, "crops", len(ims), "identities", len(uniq))


# ------------------------------------------------------------------------------------------------ model
class ReIDNet(nn.Module):
    """ResNet-18 backbone, last stride 1, BNNeck (Luo et al., 'Bag of Tricks for ReID', 2019)."""

    def __init__(self, n_ids=0):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        r.layer4[0].conv1.stride = (1, 1); r.layer4[0].downsample[0].stride = (1, 1)
        self.body = nn.Sequential(*list(r.children())[:-2])
        self.bn = nn.BatchNorm1d(512); self.bn.bias.requires_grad_(False)
        self.cls = nn.Linear(512, n_ids, bias=False) if n_ids else None

    def forward(self, x):
        f = self.body(x).mean((2, 3))
        b = self.bn(f)
        return (f, self.cls(b)) if self.training and self.cls is not None else F.normalize(b, dim=1)


def batch_hard_triplet(f, y, margin=0.3):
    d = torch.cdist(f, f)
    same = y[:, None] == y[None, :]
    pos = (d * same.float()).max(1).values
    neg = (d + 1e5 * same.float()).min(1).values
    return F.relu(pos - neg + margin).mean()


def augment(x):
    """x: float batch (N,3,H,W) normalised. Flip, pad-and-crop, random erasing."""
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = F.pad(x, (8, 8, 8, 8), mode="reflect")
    i, j = np.random.randint(0, 17, 2)
    x = x[:, :, i:i + H, j:j + W].contiguous()
    for k in range(n):
        if np.random.rand() < 0.5:
            eh, ew = np.random.randint(H // 8, H // 2), np.random.randint(W // 8, W // 2)
            y0, x0 = np.random.randint(0, H - eh), np.random.randint(0, W - ew)
            x[k, :, y0:y0 + eh, x0:x0 + ew] = torch.randn(3, eh, ew, device=x.device) * 0.5
    return x


def train(fold, steps=4000, P=16, K=4, lr=3.5e-4, seed=0):
    """Train the model that embeds `fold`, on the crops of the *other* fold."""
    torch.manual_seed(seed); np.random.seed(seed)
    d = np.load(os.path.join(CACHE, f"reid_crops_{other(fold)}.npz"))
    x, y = d["x"], d["y"]
    by_id = [np.where(y == i)[0] for i in range(y.max() + 1)]
    ids = [i for i, v in enumerate(by_id) if len(v) >= 2]
    dev = "cuda"
    net = ReIDNet(y.max() + 1).to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    net.train(); t0 = time.time()
    for step in range(steps):
        pick = np.random.choice(ids, P, replace=False)
        idx = np.concatenate([np.random.choice(by_id[i], K, replace=len(by_id[i]) < K) for i in pick])
        xb = augment(to_tensor(x[idx]).to(dev))
        yb = torch.from_numpy(y[idx]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            f, logits = net(xb)
            loss = F.cross_entropy(logits.float(), yb, label_smoothing=0.1) + batch_hard_triplet(f.float(), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 250 == 0 or step == steps - 1:
            print(f"fold {fold} step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
    torch.save({k: v for k, v in net.state_dict().items() if not k.startswith("cls.")},
               os.path.join(ROOT, "weights", f"reid_r18_embeds_fold{fold}.pt"))


# ------------------------------------------------------------------------------------------------ embed
class Embedder:
    def __init__(self, kind, fold=None, dev="cuda", path=None):
        self.kind, self.dev = kind, dev
        if kind == "imagenet":
            import torchvision
            r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            self.net, self.size = nn.Sequential(*list(r.children())[:-1], nn.Flatten()), (H, W)
        elif kind == "dinov2":
            self.net, self.size = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False), (224, 112)
        else:
            self.net, self.size = ReIDNet(), (H, W)
            path = path or os.path.join(ROOT, "weights", f"reid_r18_embeds_fold{fold}.pt")
            sd = torch.load(path, map_location="cpu")
            self.net.load_state_dict(sd, strict=False)
        self.net.eval().to(dev)

    @torch.no_grad()
    def __call__(self, crops, bs=256):
        out = []
        for i in range(0, len(crops), bs):
            with torch.autocast("cuda", dtype=torch.float16):
                f = self.net(to_tensor(crops[i:i + bs], self.size).to(self.dev))
            out.append(F.normalize(f.float(), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def embed(det_names, half="val", min_score=0.5):
    """Embed every detection with score >= min_score -> cache/emb_{dets}_{kind}.pkl {seq: {frame: (N, D)}}
    (rows of lower-score boxes are zero; the tracker only uses appearance for confident detections)."""
    from framerate_study import oracle_dets
    from run_benchmark import load_dets
    all_dets = {d: (oracle_dets(half) if d == "oracle_gt" else load_dets(d)) for d in det_names}
    for kind in EMBEDDERS:
        out = {d: {} for d in det_names}
        shared = None if kind == "reid" else Embedder(kind)
        for seq in mot.SEQS:
            emb = Embedder(kind, fold=fold_of(seq)) if kind == "reid" else shared
            for d in det_names:
                out[d][seq] = {}
            for f in mot.half_frames(seq, half):
                img = None
                for d in det_names:
                    b, s = all_dets[d][seq].get(f, (np.zeros((0, 4)), np.zeros(0)))
                    keep = np.where(s >= min_score)[0]
                    res = np.zeros((len(s), 1), np.float32)
                    if len(keep):
                        img = cv2.imread(mot.frame_path(seq, f)) if img is None else img
                        e = emb(np.stack([crop(img, b[i]) for i in keep]))
                        res = np.zeros((len(s), e.shape[1]), np.float32); res[keep] = e
                    out[d][seq][f] = res
            print(kind, seq, "done", flush=True)
        for d in det_names:
            with open(os.path.join(CACHE, f"emb_{d}_{kind}.pkl"), "wb") as fh:
                pickle.dump(out[d], fh)


def load_emb(det_name, kind):
    with open(os.path.join(CACHE, f"emb_{det_name}_{kind}.pkl"), "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["crops", "train", "embed"])
    ap.add_argument("--fold", choices=["A", "B"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dets", nargs="+", default=["yolo11s_mot17", "public_frcnn", "oracle_gt"])
    a = ap.parse_args()
    if a.cmd == "crops":
        build_crops()
    elif a.cmd == "train":
        train(a.fold, steps=a.steps)
    else:
        embed(a.dets)


if __name__ == "__main__":
    main()
