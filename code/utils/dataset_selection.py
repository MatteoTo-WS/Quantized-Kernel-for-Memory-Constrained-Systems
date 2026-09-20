"""
Dataset disponibili
-------------------
  "casp"          Physicochemical Properties of Protein Tertiary Structure
                  (CASP) — 45730 campioni, 9 feature, target: RMSD
                  Fonte: OpenML ID 42903

  "superconduct"  Superconductivity Data — 21263 campioni, 81 feature,
                  target: temperatura critica (K)
                  Fonte: OpenML name='superconduct', version=1
"""


import os
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml

_DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


_DATASET_INFO = {
    "casp": {
        "description": (
            "Physicochemical Properties of Protein Tertiary Structure (CASP)\n"
            "  Campioni : 45730\n"
            "  Feature  : 9  (F1-F9: attributi fisico-chimici dalla PDB)\n"
            "  Target   : RMSD (Root Mean Square Deviation, in Angstrom)\n"
            "  Fonte    : OpenML ID 42903"
        ),
        # target_column esplicito: evita il bug di sklearn con parser='auto'
        # quando il dataset non ha un target marcato nel file ARFF
        "openml_kwargs": {
            "data_id": 42903,
            "target_column": "RMSD",
            "as_frame": True,
            "parser": "auto",
        },
    },
    "superconduct": {
        "description": (
            "Superconductivity Data\n"
            "  Campioni : 21263\n"
            "  Feature  : 81  (proprietà dei materiali superconduttori)\n"
            "  Target   : critical_temp (temperatura critica in Kelvin)\n"
            "  Fonte    : OpenML name='superconduct', version=1"
        ),
        "openml_kwargs": {
            "name": "superconduct",
            "version": 1,
            "as_frame": True,
            "parser": "auto",
        },
    },
}


def _download_openml(openml_kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Scarica da OpenML e restituisce (X, y) come ndarray float64.

    Usa as_frame=True per gestire correttamente i dataset con target_column
    esplicito (workaround per il bug di sklearn con parser='auto' e ARFF
    senza target marcato).
    """
    dataset = fetch_openml(**openml_kwargs)

    # Con as_frame=True, dataset.data e dataset.target sono DataFrame/Series
    X = dataset.data
    y = dataset.target

    if hasattr(X, "to_numpy"):
        X = X.to_numpy(dtype=np.float64)
    else:
        X = np.array(X, dtype=np.float64)

    if hasattr(y, "to_numpy"):
        y = y.to_numpy(dtype=np.float64).ravel()
    else:
        y = np.array(y, dtype=np.float64).ravel()

    return X, y


# cache
def _cache_path(name: str) -> Path:
    return _DATASETS_DIR / f"{name}.npz"


def _save_local(name: str, X: np.ndarray, y: np.ndarray) -> None:
    np.savez_compressed(_cache_path(name), X=X, y=y)
    print(f"  [cache] Dataset salvato in: {_cache_path(name)}")


def _load_local(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(_cache_path(name))
    return data["X"], data["y"]


def _is_cached(name: str) -> bool:
    return _cache_path(name).exists()


# median heuristic
def _median_heuristic(
    X: np.ndarray,
    n_sub: int = 2000,
    random_state: int | None = None,
) -> float:
    """
        sigma = median( ||x_i - x_j|| )   per i < j
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.choice(n, size=min(n_sub, n), replace=False)
    X_sub = X[idx]

    # Distanze a coppie vettorizzate
    diff = X_sub[:, None, :] - X_sub[None, :, :]   # (n_sub, n_sub, d)
    dists = np.sqrt((diff ** 2).sum(axis=-1))        # (n_sub, n_sub)

    # Solo triangolo superiore (coppie distinte i < j)
    upper = dists[np.triu_indices(len(X_sub), k=1)]
    return float(np.median(upper))


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
def load_dataset(
    name: str,
    normalize: bool = False,
    median_heuristic: bool = False,
    n_samples: int | None = None,
    random_state: int | None = None,
):
    name = name.lower().strip()
    if name not in _DATASET_INFO:
        raise ValueError(
            f"Dataset '{name}' non riconosciuto. "
            f"Scegli tra: {list(_DATASET_INFO.keys())}"
        )

    info = _DATASET_INFO[name]

    if _is_cached(name):
        print(f"[load_dataset] '{name}' trovato in cache — caricamento da disco.")
        X, y = _load_local(name)
    else:
        print(f"[load_dataset] '{name}' non in cache — download da OpenML...")
        X, y = _download_openml(info["openml_kwargs"])
        _save_local(name, X, y)

   
   # remove NaN/ Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not mask.all():
        n_dropped = int((~mask).sum())
        print(f"  [pulizia] Rimosse {n_dropped} righe con NaN/Inf.")
        X, y = X[mask], y[mask]

    n_total = X.shape[0]

    # sample
    if n_samples is not None and n_samples < n_total:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_total, size=n_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f"  [campionamento] Estratti {n_samples} / {n_total} campioni.")

    # normalize
    if normalize:
        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0)
        X_std[X_std == 0] = 1.0      # feature costanti: evita divisione per 0
        X = (X - X_mean) / X_std

        y_mean, y_std = y.mean(), y.std()
        if y_std == 0:
            y_std = 1.0
        y = (y - y_mean) / y_std
        print("  [normalizzazione] X e y standardizzati (media=0, std=1).")

    # funny print
    print(f"\n--- {name.upper()} ---")
    print(info["description"])
    print(f"  Shape X  : {X.shape}")
    print(f"  Shape y  : {y.shape}  |  range y = [{y.min():.3f}, {y.max():.3f}]")

    # as numpy float64
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    # median heuristic
    if median_heuristic:
        sigma = _median_heuristic(X, random_state=random_state)
        print(f"  Sigma (mediana euristica distanze) : {sigma:.6f}\n")
        return X, y, sigma

    print()
    return X, y
