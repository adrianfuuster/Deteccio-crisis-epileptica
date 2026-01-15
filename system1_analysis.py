"""
System 1 EEG (baseline) — análisis poblacional y personalizado

- Poblacional (cross-subject): Leave-One-Patient-Out (LOPO) usando patient_id.
- Personalizado (patient-specific): para cada paciente, Leave-One-Event-Out (o Leave-One-Recording-Out)
  agrupando por `global_interval` (recomendado) o `filename`.

Dataset esperado (por paciente chbXX):
  - chbXX_seizure_EEGwindow_1.npz  (clave: "EEG_win", shape [N, C, T])
  - chbXX_seizure_metadata_1.parquet (columna: "class" (0/1), y columnas de split)

Columnas de metadata que usa este script:
  - class (obligatoria)
  - filename (obligatoria)  -> también sirve para split por recording
  - global_interval (opcional, recomendado para personalizado)
  - filename_interval (opcional, alternativa)

Ejemplos:
  # Poblacional (LOPO)
  python system1_analysis.py --analysis population

  # Personalizado (Leave-one-event-out dentro de cada paciente, usando global_interval)
  python system1_analysis.py --analysis personalized --personal_group global_interval

  # Personalizado (Leave-one-recording-out dentro de cada paciente)
  python system1_analysis.py --analysis personalized --personal_group filename
"""

import os
import random
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import LeaveOneGroupOut

# -----------------------------
# Config por defecto
# -----------------------------
DEFAULT_DATA_ROOT = r"C:\Users\mcasesf\Desktop\Proj_4_PSIV_AA\input"

DEFAULT_FIRST_ID = 1
DEFAULT_LAST_ID = 24

DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_SEED = 0

DEFAULT_OUT_DIR = "outputs_system1"

# W&B (opcional)
DEFAULT_USE_WANDB = True
DEFAULT_WANDB_PROJECT = "eeg-psiv-system1_analisys_filename"
DEFAULT_WANDB_ENTITY = None  # si no quieres, deja None
DEFAULT_WANDB_TAGS = ["system1"]


# -----------------------------
# Métricas binarias
# -----------------------------
def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def compute_binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    acc = _safe_div(tp + tn, total)

    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)  # sensibilidad (seizure recall)
    recall_neg = _safe_div(tn, tn + fp)  # especificidad (non-seizure recall)

    f1_pos = _safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    bal_acc = 0.5 * (recall_pos + recall_neg)

    return {
        "acc": acc,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "recall_neg": recall_neg,
        "f1_pos": f1_pos,
        "balanced_acc": bal_acc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


# -----------------------------
# Dataset
# -----------------------------
class EEGWindowsDataset(Dataset):
    """
    Carga ventanas EEG:
      - .npz: clave "EEG_win" [N, C, T]
      - .parquet: "class"(0/1), "filename" y opcional "global_interval"/"filename_interval"
    """

    def __init__(self, root_dir: str, first_id: int = DEFAULT_FIRST_ID, last_id: int = DEFAULT_LAST_ID):
        super().__init__()
        self.root_dir = root_dir

        X, y, meta = self._load_all(root_dir, first_id, last_id)

        self.X = X.astype(np.float32)                # [N, C, T]
        self.y = y.astype(np.int64)                  # [N]
        self.meta = meta                             # DataFrame alineado a X/y

        self.n_windows = self.X.shape[0]
        self.n_channels = self.X.shape[1]
        self.win_len = self.X.shape[2]

        # Patient id (para split poblacional)
        self.patient_ids = self.meta["patient_id"].astype(str).to_numpy()

        # Groups extra (para personalizado)
        self.global_interval = self.meta.get("global_interval", pd.Series([-1] * self.n_windows)).to_numpy()
        self.filename = self.meta["filename"].astype(str).to_numpy()
        self.filename_interval = self.meta.get("filename_interval", pd.Series([-1] * self.n_windows)).to_numpy()

        print(
            f"Dataset cargado: {self.n_windows} ventanas, "
            f"{self.n_channels} canales, {self.win_len} muestras/ventana"
        )
        if self.y.min() < 0 or self.y.max() > 1:
            raise ValueError("Este script asume etiquetas binarias {0,1} en la columna 'class'.")
        print("Recuento por clase (0/1):", np.bincount(self.y, minlength=2))
        print("Pacientes distintos:", np.unique(self.patient_ids))

    def _load_all(self, root_dir: str, first_id: int, last_id: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_meta: List[pd.DataFrame] = []

        for pid_num in range(first_id, last_id + 1):
            pid = f"chb{pid_num:02d}"

            npz_path = os.path.join(root_dir, f"{pid}_seizure_EEGwindow_1.npz")
            meta_path = os.path.join(root_dir, f"{pid}_seizure_metadata_1.parquet")

            if not os.path.exists(npz_path) or not os.path.exists(meta_path):
                # no forzamos error: el dataset puede no tener todos los IDs
                print(f"[AVISO] Falta {pid}: {os.path.basename(npz_path)} o {os.path.basename(meta_path)}. Lo salto.")
                continue

            npz = np.load(npz_path, allow_pickle=True)
            if "EEG_win" not in npz:
                raise KeyError(f"En {npz_path} no existe la clave 'EEG_win'")
            X = npz["EEG_win"]

            # En algunos pipelines EEG_win puede ser object con arrays internos
            if isinstance(X, np.ndarray) and X.dtype == object:
                X = np.stack(list(X), axis=0)

            meta = pd.read_parquet(meta_path)

            if "class" not in meta.columns:
                raise KeyError(f"En {meta_path} no existe la columna 'class'")
            if "filename" not in meta.columns:
                raise KeyError(f"En {meta_path} no existe la columna 'filename'")

            y = meta["class"].astype(int).to_numpy()
            meta = meta.copy()
            meta["filename"] = meta["filename"].astype(str)

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"{pid}: ventanas en npz = {X.shape[0]} pero labels en parquet = {y.shape[0]}")

            # patient_id: preferimos parsear desde filename como en el enunciado,
            # pero si falla, usamos pid (chbXX) para todas las filas de ese archivo.
            def _parse_pid(fname: str) -> str:
                if "_" in fname:
                    return fname.split("_")[0]
                return pid

            meta["patient_id"] = meta["filename"].map(_parse_pid).astype(str)

            all_X.append(X)
            all_y.append(y)
            all_meta.append(meta.reset_index(drop=True))

        if not all_X:
            raise RuntimeError(
                f"No se ha cargado ningún paciente desde {root_dir}. "
                "Comprueba que existen los archivos chbXX_seizure_EEGwindow_1.npz "
                "y chbXX_seizure_metadata_1.parquet."
            )

        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        meta_all = pd.concat(all_meta, axis=0, ignore_index=True)

        return X_all, y_all, meta_all

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        x = self.X[idx]  # [C, T]
        y = self.y[idx]  # 0/1
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# -----------------------------
# Modelo: System 1 (concat + conv1d + FC)
# -----------------------------
class System1EEGNet(nn.Module):
    def __init__(self, n_channels: int, win_len: int, n_classes: int = 2):
        super().__init__()
        self.n_channels = n_channels
        self.win_len = win_len
        self.n_classes = n_classes

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, win_len)  # [1, C, T]
            feat = self._forward_features(dummy)
            flat_dim = feat.shape[1] * feat.shape[2]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]  -> concat: [B, 1, C*T]
        B, C, T = x.shape
        x = x.reshape(B, 1, C * T)
        x = self.conv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.train()

    running_loss = 0.0
    tp = tn = fp = fn = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)

        preds = outputs.argmax(dim=1)
        tp += ((preds == 1) & (targets == 1)).sum().item()
        tn += ((preds == 0) & (targets == 0)).sum().item()
        fp += ((preds == 1) & (targets == 0)).sum().item()
        fn += ((preds == 0) & (targets == 1)).sum().item()

    epoch_loss = running_loss / max(total, 1)
    metrics = compute_binary_metrics_from_counts(tp, tn, fp, fn)
    return epoch_loss, metrics


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()

    running_loss = 0.0
    tp = tn = fp = fn = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)

        preds = outputs.argmax(dim=1)
        tp += ((preds == 1) & (targets == 1)).sum().item()
        tn += ((preds == 0) & (targets == 0)).sum().item()
        fp += ((preds == 1) & (targets == 0)).sum().item()
        fn += ((preds == 0) & (targets == 1)).sum().item()

    epoch_loss = running_loss / max(total, 1)
    metrics = compute_binary_metrics_from_counts(tp, tn, fp, fn)
    return epoch_loss, metrics


# -----------------------------
# W&B (opcional)
# -----------------------------
def maybe_init_wandb(run_cfg: dict, enabled: bool):
    if not enabled:
        return None
    try:
        import wandb
        return wandb.init(**run_cfg)
    except Exception as e:
        print(f"[AVISO] No se pudo iniciar W&B: {e}")
        return None


def wandb_log(data: dict, step: Optional[int] = None):
    try:
        import wandb
        wandb.log(data, step=step)
    except Exception:
        pass


def wandb_finish():
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


# -----------------------------
# Experiment runners
# -----------------------------
@dataclass
class TrainConfig:
    data_root: str
    first_id: int
    last_id: int
    batch_size: int
    num_epochs: int
    lr: float
    seed: int
    out_dir: Path
    device: torch.device
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_tags: List[str]


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_training_loop(
    cfg: TrainConfig,
    dataset: EEGWindowsDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    run_name: str,
    wandb_group: str,
    extra_summary: Optional[dict] = None,
) -> Dict[str, float]:
    # Dataloaders
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = System1EEGNet(dataset.n_channels, dataset.win_len, n_classes=2).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # W&B
    run = maybe_init_wandb(
        {
            "project": cfg.wandb_project,
            "entity": cfg.wandb_entity,
            "name": run_name,
            "group": wandb_group,
            "tags": cfg.wandb_tags,
            "config": {
                "analysis": wandb_group,
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "lr": cfg.lr,
                "seed": cfg.seed,
                "n_channels": dataset.n_channels,
                "win_len": dataset.win_len,
                **(extra_summary or {}),
            },
            "reinit": True,
        },
        enabled=cfg.use_wandb,
    )

    # Train
    best_metric = -1.0
    best_state = None

    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        va_loss, va_m = eval_one_epoch(model, val_loader, criterion, cfg.device)

        # métrica para "mejor": F1 pos (seizure)
        current = va_m["f1_pos"]
        if current > best_metric:
            best_metric = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        log_dict = {
            "epoch": epoch,
            "train/loss": tr_loss,
            "val/loss": va_loss,
            **{f"train/{k}": v for k, v in tr_m.items() if k not in ["tp", "tn", "fp", "fn"]},
            **{f"val/{k}": v for k, v in va_m.items() if k not in ["tp", "tn", "fp", "fn"]},
            "val/tp": va_m["tp"],
            "val/tn": va_m["tn"],
            "val/fp": va_m["fp"],
            "val/fn": va_m["fn"],
        }
        wandb_log(log_dict, step=epoch)

        print(
            f"[{run_name}] epoch {epoch:02d} | "
            f"val_f1+={va_m['f1_pos']:.4f} val_rec+={va_m['recall_pos']:.4f} val_rec-={va_m['recall_neg']:.4f} "
            f"val_acc={va_m['acc']:.4f}"
        )

    # Guardar best model del run
    cfg.out_dir.mkdir(exist_ok=True, parents=True)
    best_path = cfg.out_dir / f"{run_name}_best_f1pos.pth"
    last_path = cfg.out_dir / f"{run_name}_last.pth"

    if best_state is not None:
        torch.save(best_state, best_path)
        try:
            import wandb
            wandb.save(str(best_path))
        except Exception:
            pass

    torch.save(model.state_dict(), last_path)
    try:
        import wandb
        wandb.save(str(last_path))
    except Exception:
        pass

    # Eval final con best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    va_loss, va_m = eval_one_epoch(model, val_loader, criterion, cfg.device)
    summary = {"final/val_loss": va_loss, **{f"final/val_{k}": v for k, v in va_m.items()}}
    wandb_log(summary)

    if run is not None:
        wandb_finish()

    return {"val_loss": va_loss, **va_m}


def run_population_lopo(cfg: TrainConfig, dataset: EEGWindowsDataset) -> pd.DataFrame:
    groups = dataset.patient_ids
    logo = LeaveOneGroupOut()

    rows = []
    for fold, (train_idx, val_idx) in enumerate(logo.split(dataset.X, dataset.y, groups=groups), start=1):
        val_patient = np.unique(groups[val_idx])[0]
        run_name = f"system1_population_fold{fold:02d}_test_{val_patient}"

        res = run_training_loop(
            cfg=cfg,
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            run_name=run_name,
            wandb_group="population_LOPO",
            extra_summary={"val_patient": val_patient, "fold": fold},
        )

        rows.append({"fold": fold, "val_patient": val_patient, **res})

    df = pd.DataFrame(rows)
    return df


def _choose_personal_groups(dataset: EEGWindowsDataset, idxs: np.ndarray, personal_group: str) -> np.ndarray:
    if personal_group == "global_interval":
        g = dataset.global_interval[idxs]
    elif personal_group == "filename_interval":
        g = dataset.filename_interval[idxs]
    elif personal_group == "filename":
        g = dataset.filename[idxs]
    else:
        raise ValueError("personal_group debe ser: global_interval | filename_interval | filename")
    return np.asarray(g)


def run_personalized(cfg: TrainConfig, dataset: EEGWindowsDataset, personal_group: str) -> pd.DataFrame:
    """
    Para cada paciente:
      - Subset de indices del paciente
      - LeaveOneGroupOut sobre el grupo elegido dentro del paciente
    """
    all_rows = []
    unique_pats = np.unique(dataset.patient_ids)

    for p in unique_pats:
        p_idxs = np.where(dataset.patient_ids == p)[0]
        if p_idxs.size < 2:
            continue

        groups = _choose_personal_groups(dataset, p_idxs, personal_group=personal_group)
        unique_groups = np.unique(groups)

        # Si no hay suficientes grupos, intentamos fallback automáticamente
        if unique_groups.size < 2:
            # fallback a filename si no lo era ya
            if personal_group != "filename":
                groups = _choose_personal_groups(dataset, p_idxs, personal_group="filename")
                unique_groups = np.unique(groups)

        if unique_groups.size < 2:
            print(f"[AVISO] Paciente {p}: no hay suficientes grupos para CV (personal_group={personal_group}). Lo salto.")
            continue

        logo = LeaveOneGroupOut()

        for fold, (tr_rel, va_rel) in enumerate(logo.split(np.zeros_like(p_idxs), dataset.y[p_idxs], groups=groups), start=1):
            train_idx = p_idxs[tr_rel]
            val_idx = p_idxs[va_rel]
            left_out = np.unique(groups[va_rel])[0]

            run_name = f"system1_personal_{p}_fold{fold:02d}_test_{left_out}"
            res = run_training_loop(
                cfg=cfg,
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                run_name=run_name,
                wandb_group=f"personalized_{personal_group}",
                extra_summary={"patient": p, "fold": fold, "left_out_group": str(left_out), "personal_group": personal_group},
            )

            all_rows.append({"patient": p, "fold": fold, "left_out_group": str(left_out), **res})

    return pd.DataFrame(all_rows)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--first_id", type=int, default=DEFAULT_FIRST_ID)
    ap.add_argument("--last_id", type=int, default=DEFAULT_LAST_ID)

    ap.add_argument("--analysis", type=str, choices=["population", "personalized"], default="population")
    ap.add_argument("--personal_group", type=str, choices=["global_interval", "filename_interval", "filename"], default="global_interval")

    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    ap.add_argument("--wandb_entity", type=str, default=DEFAULT_WANDB_ENTITY)

    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    set_seeds(args.seed)

    cfg = TrainConfig(
        data_root=args.data_root,
        first_id=args.first_id,
        last_id=args.last_id,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        seed=args.seed,
        out_dir=out_dir,
        device=device,
        use_wandb=(DEFAULT_USE_WANDB and not args.no_wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=DEFAULT_WANDB_TAGS + [args.analysis],
    )

    dataset = EEGWindowsDataset(args.data_root, args.first_id, args.last_id)

    if args.analysis == "population":
        df = run_population_lopo(cfg, dataset)
        out_csv = out_dir / "system1_population_lopo_results.csv"
        df.to_csv(out_csv, index=False)
        print("\n=== Resumen poblacional (media por fold) ===")
        if len(df) > 0:
            print(df[["val_patient", "acc", "recall_pos", "recall_neg", "f1_pos", "balanced_acc"]].mean(numeric_only=True))
        print(f"Resultados guardados en: {out_csv}")

    else:
        df = run_personalized(cfg, dataset, personal_group=args.personal_group)
        out_csv = out_dir / f"system1_personalized_{args.personal_group}_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n=== Resumen personalizado ({args.personal_group}) ===")
        if len(df) > 0:
            print(df[["acc", "recall_pos", "recall_neg", "f1_pos", "balanced_acc"]].mean(numeric_only=True))
        print(f"Resultados guardados en: {out_csv}")


if __name__ == "__main__":
    main()
