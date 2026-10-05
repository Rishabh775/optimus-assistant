"""Train the bare "optimus" wake-word verifier.

Same flow as train_minimal.py (the one that produced hey_optimus), improved:
- negatives include GENERATED adversarial clips (hey jarvis, optimum,
  octopus, hey alexa, ...) in addition to random validation-set frames,
  so near-homophones are explicitly trained against
- reports score distributions -> recommends a threshold
- evaluates on the held-out positive_test / negative_test dirs

Output: optimus.onnx + optimus.json (compatible with jarvis wake.py loader)
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).parent
# Parameterized via CLI: --data-dir <dir> --model-name <name>
POS_DIR = BASE / "out_optimus/optimus/positive_train"
POS_TEST_DIR = BASE / "out_optimus/optimus/positive_test"
NEG_DIR = BASE / "out_optimus/optimus/negative_train"
NEG_TEST_DIR = BASE / "out_optimus/optimus/negative_test"
NEG_POOL = BASE / "validation_set_features.npy"
OUT_DIR = BASE / "my_custom_model"
MODEL_NAME = "optimus"


def _configure(data_dir: str, model_name: str):
    global POS_DIR, POS_TEST_DIR, NEG_DIR, NEG_TEST_DIR, MODEL_NAME
    d = BASE / data_dir
    POS_DIR, POS_TEST_DIR = d / "positive_train", d / "positive_test"
    NEG_DIR, NEG_TEST_DIR = d / "negative_train", d / "negative_test"
    MODEL_NAME = model_name

sys.path.insert(0, str(BASE))  # local openwakeword clone


def load_wav_int16(path):
    import wave
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sr != 16000:
        raise ValueError(f"need 16kHz, got {sr} for {path}")
    return audio


def extract_embeddings(audio_paths, label=""):
    from openwakeword.utils import AudioFeatures
    af = AudioFeatures()
    all_feats = []
    for i, p in enumerate(audio_paths):
        try:
            emb = af._get_embeddings(load_wav_int16(p))
            if len(emb):
                all_feats.append(emb)
        except Exception as e:
            if i < 3:
                print(f"  skip {Path(p).name}: {e}")
    if not all_feats:
        raise RuntimeError(f"no {label} features extracted")
    feats = np.concatenate(all_feats, axis=0).astype(np.float32)
    print(f"  {label}: {len(audio_paths)} clips -> {feats.shape[0]} frames")
    return feats


class FrameDataset(Dataset):
    def __init__(self, pos, neg):
        n = min(len(pos), len(neg))
        self.pos = pos[:n]
        self.neg = neg[:n]

    def __len__(self):
        return len(self.pos) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return torch.from_numpy(self.pos[idx // 2]), torch.tensor(1.0)
        return torch.from_numpy(self.neg[idx // 2]), torch.tensor(0.0)


class WakeModel(nn.Module):
    def __init__(self, in_dim=96, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train(epochs=25, batch_size=512, lr=1e-3):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    pos_paths = sorted(POS_DIR.glob("*.wav"))
    print(f"positive clips: {len(pos_paths)}")
    if len(pos_paths) < 100:
        raise RuntimeError(f"too few positives ({len(pos_paths)})")

    print("extracting embeddings...")
    pos_feats = extract_embeddings(pos_paths, "pos")
    neg_gen = extract_embeddings(sorted(NEG_DIR.glob("*.wav")), "neg-gen")

    print("pooling random negative frames...")
    neg_all = np.load(NEG_POOL, mmap_mode='r')
    n_neg = min(len(neg_all), len(pos_feats) * 4)
    idx = np.random.choice(len(neg_all), n_neg, replace=False)
    neg_rand = np.array(neg_all[idx]).astype(np.float32)
    # balance: adversarial clips matter most; keep random frames too
    neg_feats = np.concatenate([neg_gen, neg_rand], axis=0)
    print(f"  negatives total: {neg_feats.shape[0]} frames "
          f"({neg_gen.shape[0]} adversarial + {neg_rand.shape[0]} random)")

    mean = float(pos_feats.mean())
    std = float(pos_feats.std())
    print(f"  norm: mean={mean:.4f} std={std:.4f}")
    pos_n = (pos_feats - mean) / (std + 1e-6)
    neg_n = (neg_feats - mean) / (std + 1e-6)

    n_val = min(500, len(pos_feats) // 5)
    perm = np.random.permutation(len(pos_n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    neg_perm = np.random.permutation(len(neg_n))
    neg_val, neg_train = neg_perm[:n_val], neg_perm[n_val:]

    train_ds = FrameDataset(pos_n[train_idx], neg_n[neg_train])
    val_ds = FrameDataset(pos_n[val_idx], neg_n[neg_val])
    print(f"  train frames: {len(train_ds)}  val frames: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = WakeModel(in_dim=pos_n.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for x, y in train_dl:
            opt.zero_grad()
            pred = model(x)
            loss = bce(pred, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(x)
        train_loss = loss_sum / len(train_ds)

        model.eval()
        correct = vloss_sum = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x)
                vloss_sum += bce(pred, y).item() * len(x)
                correct += ((pred > 0.5).float() == y).sum().item()
        acc = correct / len(val_ds)
        marker = ""
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        print(f"epoch {epoch+1:2d}/{epochs}  loss={train_loss:.4f}  "
              f"vloss={vloss_sum/len(val_ds):.4f}  vacc={acc:.4f}{marker}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"best val acc: {best_acc:.4f}")

    # ---- threshold calibration on held-out clips ----
    print("\nthreshold calibration (held-out clips):")
    from openwakeword.utils import AudioFeatures
    af = AudioFeatures()

    def score_dir(d):
        out = []
        for p in sorted(d.glob("*.wav")):
            try:
                e = af._get_embeddings(load_wav_int16(p))
                if len(e):
                    rows = ((e - mean) / (std + 1e-6)).astype(np.float32)
                    out.append(float(model(torch.from_numpy(rows)).max()))
            except Exception:
                pass
        return out

    with torch.no_grad():
        pos_s = score_dir(POS_TEST_DIR)
        neg_s = score_dir(NEG_TEST_DIR)
    if pos_s and neg_s:
        pos_arr, neg_arr = np.array(pos_s), np.array(neg_s)
        print(f"  positive scores: min={pos_arr.min():.3f} p5={np.percentile(pos_arr,5):.3f} "
              f"median={np.median(pos_arr):.3f}")
        print(f"  negative scores: max={neg_arr.max():.3f} p95={np.percentile(neg_arr,95):.3f} "
              f"median={np.median(neg_arr):.3f}")
        rec = (pos_arr > 0.6).mean()
        fpr = (neg_arr > 0.6).mean()
        print(f"  @0.60: recall={rec:.2%} false-wake={fpr:.2%}")
        print(f"  suggested threshold: {(np.percentile(neg_arr,99) + np.percentile(pos_arr,5))/2:.2f}")

    # ---- export ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / f"{MODEL_NAME}.pt")
    sample = torch.zeros(1, pos_n.shape[1], dtype=torch.float32)
    onnx_path = OUT_DIR / f"{MODEL_NAME}.onnx"
    tmp_path = str(onnx_path) + ".tmp"
    torch.onnx.export(
        model, sample, tmp_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    import onnx
    from onnx import TensorProto
    m = onnx.load(tmp_path, load_external_data=True)
    for t in m.graph.initializer:
        if t.data_location == TensorProto.EXTERNAL:
            t.data_location = TensorProto.DEFAULT
            for entry in list(t.external_data):
                t.external_data.remove(entry)
    onnx.save(m, str(onnx_path))
    os.unlink(tmp_path)
    Path(str(onnx_path) + ".data").unlink(missing_ok=True)
    print(f"\nexported: {onnx_path}")

    meta = {
        "model_name": MODEL_NAME,
        "in_dim": int(pos_n.shape[1]),
        "mean": mean,
        "std": std,
        "best_val_acc": float(best_acc),
        "n_positive_clips": len(pos_paths),
        "framework": "pytorch",
    }
    meta_path = OUT_DIR / f"{MODEL_NAME}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"metadata: {meta_path}")
    return onnx_path, best_acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="out_optimus/optimus")
    ap.add_argument("--model-name", default="optimus")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    _configure(args.data_dir, args.model_name)
    train(epochs=args.epochs)
