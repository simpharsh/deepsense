# ============================================================
#  DEEPFAKE DETECTION — Training Script v2
#  EfficientNet-B4 · AMP · Video-Level Split · No Leakage
#  Run with:  python deepfake_train.py
# ============================================================

import multiprocessing
multiprocessing.freeze_support()   # required on Windows

import os, io, cv2, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, classification_report


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE = "Datasets"

FF_REAL       = os.path.join(BASE, "faceforensic", "original")
FF_FAKES      = [os.path.join(BASE, "faceforensic", d)
                 for d in ["Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures"]]
CELEB_REALS   = [os.path.join(BASE, "celeb", "Celeb-real"),
                 os.path.join(BASE, "celeb", "YouTube-real")]
CELEB_FAKE    = os.path.join(BASE, "celeb", "Celeb-synthesis")
CELEB_TLIST   = os.path.join(BASE, "celeb", "List_of_testing_videos.txt")
DFD_REAL      = os.path.join(BASE, "DFD", "DFD_original sequences")
DFD_FAKE      = os.path.join(BASE, "DFD", "DFD_manipulated sequences")
WILD_REAL     = os.path.join(BASE, "wild-deepfake", "real")
WILD_FAKE     = os.path.join(BASE, "wild-deepfake", "fake")

EXTRACTED_PATH  = "processed_frames"
IMG_SIZE        = 224
FRAME_SKIP      = 5
MAX_FRAMES      = 20
USE_FACE_DETECT = False

BATCH_SIZE      = 32           # smaller → more gradient noise → better generalization
EPOCHS          = 40
PATIENCE        = 7            # more patience since real val F1 will be lower
VAL_SPLIT       = 0.15
NUM_WORKERS     = 4
LABEL_SMOOTHING = 0.1          # stronger smoothing → less overconfidence
BEST_MODEL_PATH = "best_model.pth"

INFER_THRESHOLD = 0.6          # raised → reduce false-fake predictions
INFER_FRAMES    = 15

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

cudnn.benchmark     = True
cudnn.deterministic = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
#  FACE DETECTOR  (optional)
# ═══════════════════════════════════════════════════════════════

face_detector = None
if USE_FACE_DETECT:
    try:
        from facenet_pytorch import MTCNN
        face_detector = MTCNN(
            image_size=IMG_SIZE, margin=30,
            keep_all=False, device=device, post_process=False
        )
        print("✅ MTCNN face detector loaded")
    except ImportError:
        USE_FACE_DETECT = False
        print("⚠️  facenet-pytorch not found → full-frame mode")


# ═══════════════════════════════════════════════════════════════
#  FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_frames(src_folder, dst_folder,
                   frame_skip=FRAME_SKIP, max_frames=MAX_FRAMES):
    os.makedirs(dst_folder, exist_ok=True)
    VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")

    video_files = [
        os.path.join(r, f)
        for r, _, files in os.walk(src_folder)
        for f in files if f.lower().endswith(VIDEO_EXT)
    ]
    if not video_files:
        print(f"  ⚠️  No videos in: {src_folder}")
        return

    for vpath in tqdm(video_files, desc=os.path.basename(src_folder), ncols=90):
        stem = os.path.splitext(os.path.basename(vpath))[0]
        if sum(1 for f in os.listdir(dst_folder)
               if f.startswith(stem + "_")) >= max_frames:
            continue

        cap = cv2.VideoCapture(vpath)
        frame_idx, saved = 0, 0

        while cap.isOpened() and saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                if USE_FACE_DETECT and face_detector is not None:
                    try:
                        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        ft  = face_detector(pil)
                        if ft is not None:
                            face_np = ft.permute(1,2,0).numpy()
                            frame   = cv2.cvtColor(
                                np.clip(face_np,0,255).astype(np.uint8),
                                cv2.COLOR_RGB2BGR)
                    except Exception:
                        pass
                out = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(
                    os.path.join(dst_folder, f"{stem}_{saved:04d}.jpg"),
                    out, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                saved += 1
            frame_idx += 1
        cap.release()


# ═══════════════════════════════════════════════════════════════
#  AUGMENTATIONS  — stronger for better generalization
# ═══════════════════════════════════════════════════════════════

class JPEGNoise:
    def __init__(self, quality_range=(50, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).copy()
        return img


transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    JPEGNoise(quality_range=(50, 95), p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ═══════════════════════════════════════════════════════════════
#  VIDEO-LEVEL SPLIT — prevents data leakage
#  Frames from the same video NEVER appear in both train & val
# ═══════════════════════════════════════════════════════════════

def get_video_stem(filename):
    """
    Extract original video name from frame filename.
    "video001_0003.jpg"  ->  "video001"
    "abc_def_0012.jpg"   ->  "abc_def"
    """
    name  = os.path.splitext(filename)[0]
    parts = name.split("_")
    if parts[-1].isdigit():
        return "_".join(parts[:-1])
    return name


def video_level_split(dataset_path, val_split=VAL_SPLIT, seed=42):
    """
    Split by VIDEO not by frame.
    Returns train_indices, val_indices, full ImageFolder dataset.
    """
    real_dir = os.path.join(dataset_path, "real")
    fake_dir = os.path.join(dataset_path, "fake")

    real_videos = sorted(set(
        get_video_stem(f) for f in os.listdir(real_dir)
        if f.lower().endswith(".jpg")
    ))
    fake_videos = sorted(set(
        get_video_stem(f) for f in os.listdir(fake_dir)
        if f.lower().endswith(".jpg")
    ))

    random.seed(seed)
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    real_val_n   = max(1, int(len(real_videos) * val_split))
    fake_val_n   = max(1, int(len(fake_videos) * val_split))
    val_real_set = set(real_videos[:real_val_n])
    val_fake_set = set(fake_videos[:fake_val_n])

    print(f"  Real videos → train: {len(real_videos)-real_val_n}  val: {real_val_n}")
    print(f"  Fake videos → train: {len(fake_videos)-fake_val_n}  val: {fake_val_n}")

    full_dataset = ImageFolder(dataset_path, transform=transform_val)
    fake_idx     = full_dataset.class_to_idx.get("fake", 0)
    real_idx     = full_dataset.class_to_idx.get("real", 1)

    train_indices, val_indices = [], []
    for idx, (path, label) in enumerate(full_dataset.samples):
        fname  = os.path.basename(path)
        stem   = get_video_stem(os.path.splitext(fname)[0])
        is_val = (stem in val_real_set) if label == real_idx else (stem in val_fake_set)
        (val_indices if is_val else train_indices).append(idx)

    return train_indices, val_indices, full_dataset


# ═══════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════

def build_model(device):
    m = models.efficientnet_b4(weights="IMAGENET1K_V1")

    params        = list(m.parameters())
    freeze_until  = int(len(params) * 0.30)
    for i, p in enumerate(params):
        p.requires_grad = (i >= freeze_until)

    in_feat = m.classifier[1].in_features   # 1792 for B4
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_feat, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.4),
        nn.Linear(512, 1),
    )
    return m.to(device)


# ═══════════════════════════════════════════════════════════════
#  LOSS
# ═══════════════════════════════════════════════════════════════

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smooth)


# ═══════════════════════════════════════════════════════════════
#  INFERENCE UTILITIES
# ═══════════════════════════════════════════════════════════════

_tta_transforms = [
    transform_val,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]


def load_best_model(path=BEST_MODEL_PATH):
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    m    = build_model(device)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        m.load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Loaded  epoch={ckpt.get('epoch','?')}  "
              f"Val F1={ckpt.get('best_val_f1',0):.4f}")
    else:
        m.load_state_dict(ckpt)
        print("✅ Loaded legacy weights")
    m.eval()
    return m


def predict_video(video_path, model=None, n_frames=INFER_FRAMES,
                  frame_skip=10, threshold=INFER_THRESHOLD, tta=True):
    if model is None:
        model = load_best_model()
    model.eval()

    tfms  = _tta_transforms if tta else [transform_val]
    cap   = cv2.VideoCapture(video_path)
    probs = []
    count = 0

    while cap.isOpened() and len(probs) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            pil = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ).resize((IMG_SIZE, IMG_SIZE))
            fp = []
            for tfm in tfms:
                t = tfm(pil).unsqueeze(0).to(device)
                with torch.no_grad(), autocast('cuda'):
                    fp.append(torch.sigmoid(model(t)).item())
            probs.append(float(np.mean(fp)))
        count += 1

    cap.release()
    if not probs:
        return 0.5, "UNKNOWN", 50.0

    p          = float(np.mean(probs))
    label      = "FAKE" if p > threshold else "REAL"
    confidence = p * 100 if label == "FAKE" else (1 - p) * 100
    return p, label, confidence


def predict_image(image_path, model=None, threshold=INFER_THRESHOLD, tta=True):
    if model is None:
        model = load_best_model()
    model.eval()

    pil  = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    tfms = _tta_transforms if tta else [transform_val]
    fp   = []
    for tfm in tfms:
        t = tfm(pil).unsqueeze(0).to(device)
        with torch.no_grad(), autocast('cuda'):
            fp.append(torch.sigmoid(model(t)).item())

    p          = float(np.mean(fp))
    label      = "FAKE" if p > threshold else "REAL"
    confidence = p * 100 if label == "FAKE" else (1 - p) * 100
    return p, label, confidence


def test_celeb_df(model=None):
    if model is None:
        model = load_best_model()

    all_preds, all_labels = [], []
    test_set = set()
    if os.path.exists(CELEB_TLIST):
        with open(CELEB_TLIST) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_set.add(os.path.basename(parts[1]))
        print(f"Using {len(test_set)} official CelebDF-V2 test videos")

    def want(fname):
        return (not test_set) or (os.path.basename(fname) in test_set)

    VIDEO_EXT = (".mp4", ".avi", ".mov")

    for folder in CELEB_REALS:
        if not os.path.exists(folder):
            continue
        for f in tqdm(os.listdir(folder),
                      desc=f"Celeb-real {os.path.basename(folder)}"):
            if not f.lower().endswith(VIDEO_EXT):
                continue
            vpath = os.path.join(folder, f)
            if not want(vpath):
                continue
            prob, _, _ = predict_video(vpath, model=model)
            all_preds.append(1 if prob > INFER_THRESHOLD else 0)
            all_labels.append(0)

    if os.path.exists(CELEB_FAKE):
        for f in tqdm(os.listdir(CELEB_FAKE), desc="Celeb-synthesis"):
            if not f.lower().endswith(VIDEO_EXT):
                continue
            vpath = os.path.join(CELEB_FAKE, f)
            if not want(vpath):
                continue
            prob, _, _ = predict_video(vpath, model=model)
            all_preds.append(1 if prob > INFER_THRESHOLD else 0)
            all_labels.append(1)

    if not all_preds:
        print("⚠️  No test videos found!")
        return

    print("\n" + "="*55)
    print("  CELEB-DF V2 TEST RESULTS")
    print("="*55)
    print(confusion_matrix(all_labels, all_preds))
    print()
    print(classification_report(all_labels, all_preds,
                                 target_names=["Real","Fake"]))


def test_wild_deepfake(model=None):
    if model is None:
        model = load_best_model()

    all_preds, all_labels = [], []
    IMG_EXT   = (".jpg", ".jpeg", ".png")
    VIDEO_EXT = (".mp4", ".avi", ".mov")

    for label_val, folder, tag in [
        (0, WILD_REAL, "real"), (1, WILD_FAKE, "fake")
    ]:
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue
        for f in tqdm(os.listdir(folder), desc=f"Wild-Deepfake {tag}"):
            fpath = os.path.join(folder, f)
            fl    = f.lower()
            if fl.endswith(VIDEO_EXT):
                prob, _, _ = predict_video(fpath, model=model)
            elif fl.endswith(IMG_EXT):
                prob, _, _ = predict_image(fpath, model=model)
            else:
                continue
            all_preds.append(1 if prob > INFER_THRESHOLD else 0)
            all_labels.append(label_val)

    if not all_preds:
        print("⚠️  No files found for Wild-Deepfake testing!")
        return

    print("\n" + "="*55)
    print("  WILD-DEEPFAKE TEST RESULTS")
    print("="*55)
    print(confusion_matrix(all_labels, all_preds))
    print()
    print(classification_report(all_labels, all_preds,
                                 target_names=["Real","Fake"]))


# ═══════════════════════════════════════════════════════════════
#  MAIN — everything inside this guard for Windows multiprocessing
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── FRAME EXTRACTION (skipped if already done) ────────────────────────
    TRAIN_REAL = os.path.join(EXTRACTED_PATH, "train", "real")
    TRAIN_FAKE = os.path.join(EXTRACTED_PATH, "train", "fake")

    if not os.path.exists(TRAIN_REAL) or len(os.listdir(TRAIN_REAL)) == 0:
        print("\n[1/4] FaceForensics++")
        extract_frames(FF_REAL, TRAIN_REAL)
        for p in FF_FAKES:
            extract_frames(p, TRAIN_FAKE)
        print("\n[2/4] CelebDF-V2")
        for p in CELEB_REALS:
            extract_frames(p, TRAIN_REAL)
        extract_frames(CELEB_FAKE, TRAIN_FAKE)
        print("\n[3/4] DFD")
        extract_frames(DFD_REAL, TRAIN_REAL)
        extract_frames(DFD_FAKE, TRAIN_FAKE)
        print("\n[4/4] Wild-Deepfake")
        extract_frames(WILD_REAL, TRAIN_REAL)
        extract_frames(WILD_FAKE, TRAIN_FAKE)
    else:
        real_cnt = len(os.listdir(TRAIN_REAL))
        fake_cnt = len(os.listdir(TRAIN_FAKE))
        print(f"✅ Frames already extracted  real: {real_cnt:,}  fake: {fake_cnt:,}")

    # ── VIDEO-LEVEL SPLIT ─────────────────────────────────────────────────
    print("\n📂 Building video-level train/val split (no leakage)...")
    train_indices, val_indices, base_dataset = video_level_split(
        os.path.join(EXTRACTED_PATH, "train"), VAL_SPLIT
    )

    train_data_src = ImageFolder(
        os.path.join(EXTRACTED_PATH, "train"), transform=transform_train)
    val_data_src   = ImageFolder(
        os.path.join(EXTRACTED_PATH, "train"), transform=transform_val)

    train_dataset = Subset(train_data_src, train_indices)
    val_dataset   = Subset(val_data_src,   val_indices)

    train_targets  = [base_dataset.targets[i] for i in train_indices]
    class_counts   = np.bincount(train_targets)
    class_weights  = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True
    )

    fake_idx = base_dataset.class_to_idx.get("fake", 0)
    real_idx = base_dataset.class_to_idx.get("real", 1)
    print(f"\nTotal   : {len(base_dataset):,} samples")
    print(f"Train   : {len(train_indices):,}  |  Val: {len(val_indices):,}")
    print(f"Class   : real={class_counts[real_idx]:,}  fake={class_counts[fake_idx]:,}")
    print(f"Batches : train={len(train_loader)}  val={len(val_loader)}")

    # ── MODEL ─────────────────────────────────────────────────────────────
    model     = build_model(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"\nModel   : EfficientNet-B4")
    print(f"Params  : {trainable:,} trainable / {total_p:,} total "
          f"({100*trainable/total_p:.1f} %)")

    # ── LOSS / OPTIMIZER / SCHEDULER ─────────────────────────────────────
    pos_weight = torch.tensor([class_counts[fake_idx] / class_counts[real_idx]]).to(device)
    criterion  = LabelSmoothingBCE(smoothing=LABEL_SMOOTHING, pos_weight=pos_weight)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-3, betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-7
    )
    scaler = GradScaler('cuda')

    # ── LOAD CHECKPOINT ───────────────────────────────────────────────────
    start_epoch   = 0
    best_val_f1   = 0.0
    best_val_loss = float("inf")
    early_counter = 0
    temporary_models: list = []

    if os.path.exists(BEST_MODEL_PATH):
        print(f"\n🔄 Found {BEST_MODEL_PATH} — attempting to load...")
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
        try:
            state_dict    = (ckpt["model_state_dict"]
                             if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                             else ckpt)
            model_keys    = set(model.state_dict().keys())
            ckpt_keys     = set(state_dict.keys())
            has_orig      = any(k.startswith("_orig_mod.") for k in model_keys)
            ckpt_has_orig = any(k.startswith("_orig_mod.") for k in ckpt_keys)
            if has_orig and not ckpt_has_orig:
                state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}
            elif not has_orig and ckpt_has_orig:
                state_dict = {k.replace("_orig_mod.", "", 1): v
                              for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                # ── Load weights only — always retrain FULL epochs from scratch
                # ── Optimizer/scheduler/scaler intentionally NOT restored
                # ── so LR starts fresh and model gets a full new training run
                start_epoch   = 0
                best_val_f1   = ckpt.get("best_val_f1", 0.0)
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                print(f"   ✅ Loaded best weights  (previous Val F1: {best_val_f1:.4f})")
                print(f"   🔁 Retraining full {EPOCHS} epochs on top of these weights")
                print(f"   ⚡ Optimizer & scheduler reset — fresh LR schedule")
                # Reset tracked best so this run competes fairly
                best_val_f1   = 0.0
                best_val_loss = float("inf")
            else:
                print("   ✅ Loaded weights. Optimizer starts fresh.")
        except RuntimeError as e:
            print(f"   ⚠️  Incompatible: {str(e)[:120]}")
            backup = BEST_MODEL_PATH.replace(".pth", "_old.pth")
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(BEST_MODEL_PATH, backup)
            print(f"   💾 Backed up to: {backup}  — starting fresh.")
    else:
        print("\n⚡ No checkpoint found — training from scratch.")

    # ── TRAINING LOOP ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DEEPFAKE DETECTION TRAINING  |  EfficientNet-B4  |  {device}")
    print(f"{'='*70}")
    print(f"  Epochs {start_epoch}→{EPOCHS}  |  Batch {BATCH_SIZE}  |  "
          f"Workers {NUM_WORKERS}  |  AMP ON  |  Patience {PATIENCE}")
    print(f"  Split: VIDEO-LEVEL (no leakage)  |  Threshold: {INFER_THRESHOLD}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, EPOCHS):

        # Train
        model.train()
        t_loss, t_preds, t_labels = 0.0, [], []
        bar = tqdm(train_loader,
                   desc=f"Ep {epoch+1:02d}/{EPOCHS} train",
                   ncols=90, leave=False)

        for imgs, lbls in bar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.float().unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                out  = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            preds   = (torch.sigmoid(out.detach()) > INFER_THRESHOLD).float()
            t_preds.extend(preds.cpu().numpy())
            t_labels.extend(lbls.cpu().numpy())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = t_loss / len(train_loader)
        train_f1   = f1_score(t_labels, t_preds, zero_division=0)

        # Validate
        model.eval()
        v_loss, v_preds, v_labels = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader,
                                   desc=f"Ep {epoch+1:02d}/{EPOCHS} val  ",
                                   ncols=90, leave=False):
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.float().unsqueeze(1).to(device, non_blocking=True)
                with autocast('cuda'):
                    out  = model(imgs)
                    loss = criterion(out, lbls)
                v_loss += loss.item()
                preds   = (torch.sigmoid(out) > INFER_THRESHOLD).float()
                v_preds.extend(preds.cpu().numpy())
                v_labels.extend(lbls.cpu().numpy())

        val_loss = v_loss / len(val_loader)
        val_f1   = f1_score(v_labels, v_preds, zero_division=0)
        scheduler.step(epoch + 1)
        lr   = optimizer.param_groups[0]["lr"]
        flag = " 🔥" if val_f1 > best_val_f1 else ""

        print(
            f"Ep {epoch+1:02d}/{EPOCHS} | "
            f"Train loss={train_loss:.4f} f1={train_f1:.4f} | "
            f"Val  loss={val_loss:.4f} f1={val_f1:.4f}{flag} | "
            f"lr={lr:.2e}"
        )

        # Save temp checkpoint
        tmp_path = f"temp_ep{epoch+1:02d}_valf1_{val_f1:.4f}.pth"
        torch.save(model.state_dict(), tmp_path)
        temporary_models.append((tmp_path, val_f1))

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            best_val_loss = val_loss
            early_counter = 0
            torch.save({
                "epoch"               : epoch + 1,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict"   : scaler.state_dict(),
                "best_val_f1"         : best_val_f1,
                "best_val_loss"       : best_val_loss,
                "train_f1"            : train_f1,
                "train_loss"          : train_loss,
            }, BEST_MODEL_PATH)
            print(f"   💾 Saved → {BEST_MODEL_PATH}  "
                  f"(Val F1: {best_val_f1:.4f})")
        else:
            early_counter += 1
            print(f"   ⚠️  No improvement  [{early_counter}/{PATIENCE}]")
            if early_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}.")
                break

    print(f"\n{'='*70}")
    print(f"  Training done!  Best Val F1: {best_val_f1:.4f}")
    print(f"{'='*70}")

    # ── POST-TRAINING: re-evaluate all temp checkpoints ───────────────────
    print("\n🔍 Re-evaluating all temp checkpoints on validation set...")
    print("-" * 60)

    final_best_f1, final_best_file = 0.0, None

    for tmp_path, _ in temporary_models:
        if not os.path.exists(tmp_path):
            continue
        model.load_state_dict(torch.load(tmp_path, map_location=device))
        model.eval()
        preds_list, labels_list = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.float().unsqueeze(1).to(device, non_blocking=True)
                with autocast('cuda'):
                    out = model(imgs)
                preds = (torch.sigmoid(out) > INFER_THRESHOLD).float()
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(lbls.cpu().numpy())
        f1  = f1_score(labels_list, preds_list, zero_division=0)
        tag = " ← BEST" if f1 > final_best_f1 else ""
        print(f"  {tmp_path:<50} Val F1: {f1:.4f}{tag}")
        if f1 > final_best_f1:
            final_best_f1, final_best_file = f1, tmp_path

    if final_best_file and final_best_f1 > best_val_f1:
        model.load_state_dict(torch.load(final_best_file, map_location=device))
        torch.save({
            "epoch"               : EPOCHS,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict"   : scaler.state_dict(),
            "best_val_f1"         : final_best_f1,
            "best_val_loss"       : best_val_loss,
        }, BEST_MODEL_PATH)
        print(f"\n🏆 Better: {final_best_file} (Val F1: {final_best_f1:.4f})")
        print(f"   💾 Updated {BEST_MODEL_PATH}")
    else:
        print(f"\n✅ {BEST_MODEL_PATH} is best (Val F1: {best_val_f1:.4f})")

    # ── CLEANUP ───────────────────────────────────────────────────────────
    removed = 0
    for tmp_path, _ in temporary_models:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            removed += 1
    for f in os.listdir("."):
        if f.endswith(".pth") and f != BEST_MODEL_PATH:
            os.remove(f)
            removed += 1
    temporary_models.clear()
    print(f"🧹 Cleaned up {removed} temp file(s).  Kept: {BEST_MODEL_PATH}")

    # ── FINAL TESTING ─────────────────────────────────────────────────────
    test_celeb_df()
    test_wild_deepfake()