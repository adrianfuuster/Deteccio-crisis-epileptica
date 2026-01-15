#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
system2_analysis.py - System2 (Backbone + LSTM) con splits como system1_analysis.py

- population: Leave-One-Patient-Out (LOPO) sobre TODOS los pacientes
- personalized: por paciente, Leave-One-Group-Out (LOGO) por:
    global_interval / filename / filename_interval

Notas clave:
- Construye secuencias K dentro de cada 'filename' (no cruza recordings).
- Para splits personalizados por grupo, DESCARTA secuencias que mezclen grupos dentro de K
  (para evitar leakage por secuencias).

W&B:
- 1 run por fold (igual estilo que system1_analysis)
- group recomendado:
    system2_population_LOPO
    system2_personalized_global_interval
    system2_personalized_filename
    system2_personalized_filename_interval
"""

import os
import glob
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import LeaveOneGroupOut

# sklearn metrics opcionales
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

# wandb
try:
    import wandb
    _WANDB_OK = True
except Exception:
    _WANDB_OK = False


# ----------------------------
# Reproducibilidad
# ----------------------------
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinístico (más lento). Si quieres más velocidad: pon benchmark=True y deterministic=False.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Descubrimiento y lectura
# ----------------------------
def find_patient_files(data_root: str):
    npz_paths = sorted(glob.glob(os.path.join(data_root, "*_seizure_EEGwindow_*.npz")))
    pq_paths  = sorted(glob.glob(os.path.join(data_root, "*_seizure_metadata_*.parquet")))

    def pat_from_path(p: str) -> str:
        base = os.path.basename(p)
        return base.split("_")[0].lower()

    npz_by_pat = {pat_from_path(p): p for p in npz_paths}
    pq_by_pat  = {pat_from_path(p): p for p in pq_paths}

    pats = sorted(set(npz_by_pat.keys()).intersection(set(pq_by_pat.keys())))
    if not pats:
        raise FileNotFoundError(
            f"No matching patient files found in {data_root}\n"
            f"Need both *_seizure_EEGwindow_*.npz and *_seizure_metadata_*.parquet"
        )
    return pats, npz_by_pat, pq_by_pat


def safe_read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            return pd.read_parquet(path, engine="fastparquet")

def load_npz_eegwin(npz_path: str) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)  # <-- CAMBIO CLAVE
    if "EEG_win" in z.files:
        arr = z["EEG_win"]
    else:
        arr = z[z.files[0]]

    # Si viene como array de objetos (lista de ventanas), lo apilamos
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)

    return arr.astype(np.float32)



class PatientCache:
    """Cache LRU para no recargar npz continuamente."""
    def __init__(self, npz_by_pat: Dict[str, str], max_in_ram: int = 2):
        self.npz_by_pat = npz_by_pat
        self.max_in_ram = max_in_ram
        self._cache: Dict[str, np.ndarray] = {}
        self._order: List[str] = []

    def get(self, pat: str) -> np.ndarray:
        if pat in self._cache:
            # refresh order
            if pat in self._order:
                self._order.remove(pat)
            self._order.append(pat)
            return self._cache[pat]

        arr = load_npz_eegwin(self.npz_by_pat[pat])
        self._cache[pat] = arr
        self._order.append(pat)

        while len(self._order) > self.max_in_ram:
            old = self._order.pop(0)
            if old in self._cache:
                del self._cache[old]
        return arr


# ----------------------------
# Indexado de secuencias
# ----------------------------
@dataclass
class SeqIndex:
    patient: str
    idxs: np.ndarray         # indices en EEG_win
    label: int               # class del último window
    group_val: Optional[str] # grupo para LOGO (puede ser str o int, lo normalizamos a str)


def build_sequences_for_patient(
    meta: pd.DataFrame,
    patient: str,
    K: int,
    group_key: Optional[str] = None,
    require_same_group: bool = False,
) -> List[SeqIndex]:
    """
    Construye secuencias [j-K+1 ... j] dentro de cada filename.
    Si require_same_group=True y group_key!=None:
      - SOLO añade secuencias donde TODAS las K ventanas tienen el mismo meta[group_key]
      - group_val = ese valor (normalizado a str)
    Si no:
      - group_val = meta[group_key] de la última ventana (si group_key existe)
    """
    required = {"class", "filename"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata for {patient} missing columns: {missing}")

    if group_key is not None and group_key not in meta.columns:
        raise ValueError(f"group_key={group_key} not in metadata columns for {patient}")

    meta = meta.reset_index(drop=True)
    seqs: List[SeqIndex] = []

    # secuencias SOLO dentro de cada recording
    for fname, g in meta.groupby("filename", sort=False):
        idxs = g.index.to_numpy()
        labels = g["class"].astype(int).to_numpy()

        for j in range(K - 1, len(idxs)):
            window_idxs = idxs[j - (K - 1): j + 1]
            y = int(labels[j])

            gv = None
            if group_key is not None:
                if require_same_group:
                    vals = meta.loc[window_idxs, group_key].to_numpy()
                    # si mezcla grupos dentro de K, descartamos para evitar leakage
                    if not np.all(vals == vals[0]):
                        continue
                    gv = str(vals[0])
                else:
                    gv = str(meta.loc[int(window_idxs[-1]), group_key])

            seqs.append(SeqIndex(patient=patient, idxs=window_idxs, label=y, group_val=gv))

    return seqs


class EEGSequenceDataset(Dataset):
    """
    Devuelve:
      X_seq: [K,21,128]
      y: 0/1 (última ventana)
    """
    def __init__(self, seq_index: List[SeqIndex], cache: PatientCache, normalize_per_window: bool = True):
        self.seq_index = seq_index
        self.cache = cache
        self.normalize = normalize_per_window

    @staticmethod
    def _znorm_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        return (x - mu) / (sd + eps)

    def __len__(self):
        return len(self.seq_index)

    def __getitem__(self, i: int):
        item = self.seq_index[i]
        eeg = self.cache.get(item.patient)   # [N,21,128]
        x_seq = eeg[item.idxs]               # [K,21,128]
        if self.normalize:
            x_seq = np.stack([self._znorm_per_channel(w) for w in x_seq], axis=0)

        return torch.from_numpy(x_seq).float(), torch.tensor(item.label, dtype=torch.long)


def build_all_metas_and_cache(data_root: str, max_cache_patients: int):
    pats, npz_by_pat, pq_by_pat = find_patient_files(data_root)
    metas: Dict[str, pd.DataFrame] = {}
    for p in pats:
        metas[p] = safe_read_parquet(pq_by_pat[p])
    cache = PatientCache(npz_by_pat=npz_by_pat, max_in_ram=max_cache_patients)
    return pats, metas, cache


# ----------------------------
# Modelo: Backbone + LSTM (tu propuesta)
# ----------------------------
class WindowBackbone(nn.Module):
    """
    Input:  [B,21,128]
    Output: [B,D]
    """
    def __init__(self, in_ch: int = 21, D: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),    # 128 -> 64
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),        # -> [B,128,1]
        )
        self.proj = nn.Linear(128, D)

    def forward(self, x):
        z = self.net(x).squeeze(-1)         # [B,128]
        return self.proj(z)                 # [B,D]


class SeqLSTMClassifier(nn.Module):
    """
    Input:  [B,K,21,128]
    - backbone per ventana -> [B,K,D]
    - LSTM sobre K -> salida última -> head -> [B,2]
    """
    def __init__(self, K: int, D: int = 128, hidden: int = 128, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.K = K
        self.backbone = WindowBackbone(in_ch=21, D=D)
        self.lstm = nn.LSTM(
            input_size=D,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x_seq):
        B, K, C, T = x_seq.shape
        x = x_seq.reshape(B * K, C, T)   # [B*K,21,128]
        f = self.backbone(x)             # [B*K,D]
        f = f.reshape(B, K, -1)          # [B,K,D]
        out, _ = self.lstm(f)            # [B,K,H]
        h_last = out[:, -1, :]           # [B,H]
        return self.head(h_last)         # [B,2]


# ----------------------------
# Métricas
# ----------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def compute_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, prob_pos: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    recall_pos = _safe_div(tp, tp + fn)
    recall_neg = _safe_div(tn, tn + fp)
    precision_pos = _safe_div(tp, tp + fp)

    f1_pos = _safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0.0
    bal_acc = 0.5 * (recall_pos + recall_neg)

    auroc = float("nan")
    auprc = float("nan")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if prob_pos is not None and _SKLEARN_OK and n_pos > 0 and n_neg > 0:
        try:
            auroc = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            pass
        try:
            auprc = float(average_precision_score(y_true, prob_pos))
        except Exception:
            pass

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "recall_pos": recall_pos,
        "recall_neg": recall_neg,
        "precision_pos": precision_pos,
        "f1_pos": f1_pos,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_pos": n_pos, "n_neg": n_neg,
        "auroc": auroc,
        "auprc": auprc,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)   # [N,2]
    y = torch.cat(all_y, dim=0)             # [N]
    prob_pos = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_true = y.numpy().astype(int)
    y_pred = torch.argmax(logits, dim=1).numpy().astype(int)
    m = compute_metrics_binary(y_true, y_pred, prob_pos=prob_pos)
    return m, y_true, y_pred, prob_pos


def compute_class_weights(train_seqs: List[SeqIndex]) -> torch.Tensor:
    ys = np.array([s.label for s in train_seqs], dtype=np.int64)
    n0 = int((ys == 0).sum())
    n1 = int((ys == 1).sum())
    # inversa de frecuencia (normalizado)
    w0 = 1.0 / max(1, n0)
    w1 = 1.0 / max(1, n1)
    w = np.array([w0, w1], dtype=np.float32)
    w = w / w.sum() * 2.0
    return torch.tensor(w, dtype=torch.float32)


# ----------------------------
# Training loop por fold
# ----------------------------
def train_fold(
    args,
    train_seqs: List[SeqIndex],
    val_seqs: List[SeqIndex],
    cache: PatientCache,
    run_name: str,
    wandb_group: str,
):
    device = torch.device(args.device)

    train_ds = EEGSequenceDataset(train_seqs, cache, normalize_per_window=not args.no_norm)
    val_ds   = EEGSequenceDataset(val_seqs, cache, normalize_per_window=not args.no_norm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=False)

    model = SeqLSTMClassifier(
        K=args.K, D=args.D, hidden=args.hidden, layers=args.lstm_layers, dropout=args.dropout
    ).to(device)

    class_w = compute_class_weights(train_seqs).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- W&B init ----
    use_wandb = bool(args.wandb) and (args.wandb_mode != "disabled")
    if use_wandb:
        if not _WANDB_OK:
            raise RuntimeError("wandb no está instalado. pip install wandb")
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=run_name,
            group=wandb_group,
            tags=(tags if tags else None),
            mode=args.wandb_mode,
            config=vars(args),
            reinit=True,
        )
        wandb.summary["n_train_seqs"] = len(train_seqs)
        wandb.summary["n_val_seqs"] = len(val_seqs)
        wandb.summary["class_weights"] = class_w.detach().cpu().tolist()
        wandb.watch(model, log="gradients", log_freq=200)

    best = {"f1_pos": -1.0, "epoch": -1, "metrics": None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            running_loss += float(loss.item()) * x.size(0)

        running_loss /= max(1, len(train_ds))

        val_metrics, y_true, y_pred, prob_pos = evaluate(model, val_loader, device)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[{run_name}] Ep {epoch:03d} "
                f"loss={running_loss:.4f} "
                f"val_acc={val_metrics['acc']:.3f} val_bal_acc={val_metrics['bal_acc']:.3f} "
                f"val_rec+={val_metrics['recall_pos']:.3f} val_rec-={val_metrics['recall_neg']:.3f} "
                f"val_f1+={val_metrics['f1_pos']:.3f}"
            )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": running_loss,
                "lr": float(optim.param_groups[0]["lr"]),
                "val/acc": val_metrics["acc"],
                "val/bal_acc": val_metrics["bal_acc"],
                "val/recall_pos": val_metrics["recall_pos"],
                "val/recall_neg": val_metrics["recall_neg"],
                "val/precision_pos": val_metrics["precision_pos"],
                "val/f1_pos": val_metrics["f1_pos"],
                "val/auroc": val_metrics["auroc"],
                "val/auprc": val_metrics["auprc"],
                "val/tp": val_metrics["tp"],
                "val/tn": val_metrics["tn"],
                "val/fp": val_metrics["fp"],
                "val/fn": val_metrics["fn"],
                "val/n_pos": val_metrics["n_pos"],
                "val/n_neg": val_metrics["n_neg"],
            })

            # confusion matrix (opcional)
            try:
                wandb.log({
                    "val/confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=y_true,
                        preds=y_pred,
                        class_names=["normal(0)", "ictal(1)"],
                    )
                })
            except Exception:
                pass

        # igual que system1_analysis: escoger mejor por f1_pos en "val"
        if val_metrics["f1_pos"] > best["f1_pos"]:
            best = {"f1_pos": val_metrics["f1_pos"], "epoch": epoch, "metrics": val_metrics}

            # guardado opcional
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "best_epoch": epoch,
                    "best_metrics": val_metrics,
                    "args": vars(args),
                }, ckpt_path)

                if use_wandb:
                    try:
                        art = wandb.Artifact(
                            name=f"best-model-{wandb.run.id}",
                            type="model",
                            metadata={
                                "best_epoch": epoch,
                                "best_f1_pos": float(val_metrics["f1_pos"]),
                                "wandb_group": wandb_group,
                                "run_name": run_name,
                            },
                        )
                        art.add_file(ckpt_path)
                        wandb.log_artifact(art)
                    except Exception:
                        pass

    if use_wandb:
        wandb.summary["best_epoch"] = best["epoch"]
        if best["metrics"] is not None:
            for k, v in best["metrics"].items():
                if isinstance(v, (int, float, np.floating)):
                    wandb.summary[f"best/{k}"] = float(v)
        wandb.finish()

    return best


# ----------------------------
# Analysis: population LOPO
# ----------------------------
def run_population_lopo(args, pats, metas, cache):
    print("\n=== ANALYSIS: population LOPO ===")
    results = []

    for test_pat in pats:
        # train seqs = todos menos test_pat
        train_seqs: List[SeqIndex] = []
        val_seqs: List[SeqIndex] = []

        for p in pats:
            # Para población NO necesitamos group_key
            seqs_p = build_sequences_for_patient(metas[p], p, args.K, group_key=None, require_same_group=False)
            if p == test_pat:
                val_seqs.extend(seqs_p)
            else:
                train_seqs.extend(seqs_p)

        if len(train_seqs) == 0 or len(val_seqs) == 0:
            print(f"[SKIP] test_pat={test_pat} empty split train={len(train_seqs)} val={len(val_seqs)}")
            continue

        run_name = f"system2_population_fold_test_{test_pat}"
        wandb_group = "system2_population_LOPO"

        best = train_fold(args, train_seqs, val_seqs, cache, run_name, wandb_group)
        row = {
            "analysis": "population_LOPO",
            "test_patient": test_pat,
            "best_epoch": best["epoch"],
            "best_f1_pos": float(best["f1_pos"]),
            **{f"best_{k}": float(v) for k, v in (best["metrics"] or {}).items() if isinstance(v, (int, float, np.floating))},
            "n_train_seqs": len(train_seqs),
            "n_val_seqs": len(val_seqs),
        }
        results.append(row)

    return results


# ----------------------------
# Analysis: personalized LOGO por paciente
# ----------------------------
def run_personalized_logo(args, pats, metas, cache):
    print("\n=== ANALYSIS: personalized (LOGO dentro de paciente) ===")
    results = []
    split_key = args.personal_group

    for p in pats:
        meta = metas[p]
        if split_key not in meta.columns:
            print(f"[SKIP] {p} no tiene columna {split_key}")
            continue

        # Construye secuencias con group_key y require_same_group=True para evitar leakage por K
        seqs = build_sequences_for_patient(meta, p, args.K, group_key=split_key, require_same_group=True)
        if len(seqs) == 0:
            print(f"[SKIP] {p} no tiene secuencias válidas (K={args.K}) para split_key={split_key}")
            continue

        groups = np.array([s.group_val for s in seqs], dtype=object)
        uniq = np.unique(groups)
        if len(uniq) < 2:
            print(f"[SKIP] {p} split_key={split_key} solo 1 grupo: {uniq[0]}")
            continue

        logo = LeaveOneGroupOut()

        # Creamos arrays dummy X,y para logo.split; solo interesa el índice
        y = np.array([s.label for s in seqs], dtype=np.int64)
        dummyX = np.zeros((len(seqs), 1), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(logo.split(dummyX, y, groups=groups), start=1):
            train_seqs = [seqs[i] for i in tr_idx]
            val_seqs   = [seqs[i] for i in va_idx]
            left_out = str(np.unique(groups[va_idx])[0])

            run_name = f"system2_personal_{p}_fold{fold:02d}_test_{split_key}_{left_out}"
            wandb_group = f"system2_personalized_{split_key}"

            best = train_fold(args, train_seqs, val_seqs, cache, run_name, wandb_group)
            row = {
                "analysis": f"personalized_{split_key}",
                "patient": p,
                "fold": fold,
                "left_out": left_out,
                "best_epoch": best["epoch"],
                "best_f1_pos": float(best["f1_pos"]),
                **{f"best_{k}": float(v) for k, v in (best["metrics"] or {}).items() if isinstance(v, (int, float, np.floating))},
                "n_train_seqs": len(train_seqs),
                "n_val_seqs": len(val_seqs),
            }
            results.append(row)

    return results


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str,
                    default=r"C:\Users\mcasesf\Desktop\Proj_4_PSIV_AA\input")

    ap.add_argument("--analysis", type=str, choices=["population", "personalized"], default="population")
    ap.add_argument("--personal_group", type=str, default="global_interval",
                    choices=["global_interval", "filename", "filename_interval"])

    # modelo / secuencia
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)

    # training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_cache_patients", type=int, default=2)
    ap.add_argument("--no_norm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_every", type=int, default=1)

    # guardado opcional
    ap.add_argument("--save_dir", type=str, default=".\checkpoints_system2")

    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="challenge4-eeg-system2")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_tags", type=str, default="system2,lstm")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    pats, metas, cache = build_all_metas_and_cache(args.data_root, args.max_cache_patients)

    print("=== DATA ROOT ===")
    print(args.data_root)
    print("Patients:", pats)

    if args.analysis == "population":
        results = run_population_lopo(args, pats, metas, cache)
    else:
        results = run_personalized_logo(args, pats, metas, cache)

    # resumen en csv
    if results:
        out_csv = f"system2_{args.analysis}_summary.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\nSaved summary: {out_csv}")
    else:
        print("\nNo results generated (check data / K / split).")


if __name__ == "__main__":
    main()
