

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "data")


FOLDER_PATTERNS = {
    "ueda": "Ueda",
    "lorenz": "Lorenz",
    "rosler": "Rosler",
    "halvorsen": "Halvorsen",
    "rucklid": "Rucklidge",
}


def discover_data_dirs(data_root: str = None) -> Dict[str, str]:

    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    data_root = os.path.abspath(data_root)

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Katalog z danymi nie istnieje: {data_root}\n"
            f"Utwórz folder 'data/' obok skryptu i umieść w nim pobrane "
            f"foldery z danymi (C0ModelUeda..., C0ModelLorenz..., itp.)."
        )


    found = {}
    for entry in os.listdir(data_root):
        entry_path = os.path.join(data_root, entry)
        if not os.path.isdir(entry_path):
            continue

        entry_lower = entry.lower()
        for pattern, class_name in FOLDER_PATTERNS.items():
            if pattern in entry_lower:
                found[class_name] = entry_path
                break

    if not found:
        available = [e for e in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, e))]
        raise ValueError(
            f"Nie znaleziono folderów z danymi w: {data_root}\n"
            f"Oczekiwane nazwy zawierające: {list(FOLDER_PATTERNS.keys())}\n"
            f"Znalezione foldery: {available if available else '(brak)'}\n"
            f"Pobierz dane i umieść foldery C0Model... w katalogu 'data/'."
        )

    return found



CLASS_TO_IDX = {
    "Ueda": 0,
    "Lorenz": 1,
    "Rosler": 2,
    "Halvorsen": 3,
    "Rucklidge": 4,
}


MODEL_DIMS = {
    "Ueda": 2,
    "Lorenz": 3,
    "Rosler": 3,
    "Halvorsen": 3,
    "Rucklidge": 3,
}

DEFAULT_SEQ_LEN = 200

DEFAULT_TRAIN_RATIO = 0.8

RANDOM_SEED = 42




def load_single_file(filepath: str) -> Optional[np.ndarray]:

    try:
        data = np.loadtxt(filepath, delimiter=",")

        if data.ndim == 1:
            data = data.reshape(-1, 1)


        if np.isnan(data).any():
            nan_rows = np.where(np.isnan(data).any(axis=1))[0]
            first_nan = nan_rows[0]
            data = data[:first_nan]


        if len(data) > 0 and np.isinf(data).any():
            inf_rows = np.where(np.isinf(data).any(axis=1))[0]
            first_inf = inf_rows[0]
            data = data[:first_inf]

        if len(data) < 10:
            return None
        return data
    except Exception as e:
        print(f"[UWAGA] Nie udało się wczytać pliku: {filepath}\n  Błąd: {e}")
        return None


def load_all_data(
        data_root: str = None,
        data_dirs: Dict[str, str] = None,
        max_files_per_class: Optional[int] = None,
        verbose: bool = True
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, int]]:

    if data_dirs is None:
        data_dirs = discover_data_dirs(data_root)

    all_data = {}
    stats = {}

    for class_name, dir_path in data_dirs.items():
        if not os.path.isdir(dir_path):
            print(f"[BŁĄD] Folder nie istnieje: {dir_path}")
            continue

        txt_files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))

        if max_files_per_class is not None:
            txt_files = txt_files[:max_files_per_class]

        samples = []
        skipped = 0
        for fpath in txt_files:
            arr = load_single_file(fpath)
            if arr is not None:
                samples.append(arr)
            else:
                skipped += 1

        all_data[class_name] = samples
        stats[class_name] = len(samples)

        if verbose:
            dims = samples[0].shape[1] if samples else "?"
            print(f"  [{class_name:>10}] wczytano {len(samples):>5} plików "
                  f"(pominięto: {skipped:>3}), wymiar: {dims}, "
                  f"przykładowy kształt: {samples[0].shape if samples else 'brak'}")

    return all_data, stats



def pad_to_max_dim(data: np.ndarray, max_dim: int = 3) -> np.ndarray:

    current_dim = data.shape[1]
    if current_dim >= max_dim:
        return data[:, :max_dim]
    padding = np.zeros((data.shape[0], max_dim - current_dim))
    return np.hstack([data, padding])


def create_segments(
        time_series: np.ndarray,
        seq_len: int = DEFAULT_SEQ_LEN,
        stride: int = None
) -> List[np.ndarray]:

    if stride is None:
        stride = seq_len // 2

    total_len = time_series.shape[0]
    segments = []

    for start in range(0, total_len - seq_len + 1, stride):
        segment = time_series[start: start + seq_len]
        segments.append(segment)

    return segments


def preprocess_all_data(
        all_data: Dict[str, List[np.ndarray]],
        seq_len: int = DEFAULT_SEQ_LEN,
        stride: int = None,
        max_dim: int = 3,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:

    all_segments = []
    all_labels = []

    for class_name, samples in all_data.items():
        label = CLASS_TO_IDX[class_name]
        class_segments = 0

        for sample in samples:

            padded = pad_to_max_dim(sample, max_dim)


            segments = create_segments(padded, seq_len, stride)

            for seg in segments:
                all_segments.append(seg)
                all_labels.append(label)
                class_segments += 1

        if verbose:
            print(f"  [{class_name:>10}] -> {class_segments:>6} segmentów "
                  f"(seq_len={seq_len}, stride={stride or seq_len // 2})")

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    if verbose:
        print(f"\n  Łącznie: {X.shape[0]} segmentów")
        print(f"  Kształt X: {X.shape}")
        print(f"  Kształt y: {y.shape}")
        print(f"  Rozkład klas: ", end="")
        for name, idx in CLASS_TO_IDX.items():
            count = (y == idx).sum()
            print(f"{name}={count}  ", end="")
        print()

    return X, y



def normalize_data(
        X_train: np.ndarray,
        X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:

    n_train, seq_len, n_feat = X_train.shape
    n_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, n_feat)
    X_test_2d = X_test.reshape(-1, n_feat)

    mean = X_train_2d.mean(axis=0)
    std = X_train_2d.std(axis=0)

    std_safe = np.where(std == 0, 1.0, std)

    X_train_2d = (X_train_2d - mean) / std_safe
    X_test_2d = (X_test_2d - mean) / std_safe

    X_train_norm = X_train_2d.reshape(n_train, seq_len, n_feat).astype(np.float32)
    X_test_norm = X_test_2d.reshape(n_test, seq_len, n_feat).astype(np.float32)

    scaler_params = {"mean": mean, "std": std, "std_safe": std_safe}

    return X_train_norm, X_test_norm, scaler_params



# --- POPRAWKA DODANA PRZEZ TWOJEGO ZNAJOMEGO (AUTORA MODELU LSTM) ---
# Poprzednie podejście powodowało zjawisko Data Leakage, gdzie segmenty tego samego pliku
# wpadały zarówno do zbioru treningowego jak i testowego (stąd 100% skuteczności).
# Zamiast dzielić same wygenerowane segmenty, najpierw dzielimy całe pliki:
def split_data_files(
        all_data: Dict[str, List[np.ndarray]],
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        seed: int = RANDOM_SEED
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    np.random.seed(seed)
    train_data = {}
    test_data = {}
    for class_name, samples in all_data.items():
        n_samples = len(samples)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        n_train = int(n_samples * train_ratio)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_data[class_name] = [samples[i] for i in train_indices]
        test_data[class_name] = [samples[i] for i in test_indices]
        
    return train_data, test_data
# -----------------------------------------------------------------------


def split_data(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        seed: int = RANDOM_SEED,
        stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    np.random.seed(seed)

    if stratify:

        train_indices = []
        test_indices = []

        unique_labels = np.unique(y)
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            np.random.shuffle(label_indices)

            n_train = int(len(label_indices) * train_ratio)
            train_indices.extend(label_indices[:n_train])
            test_indices.extend(label_indices[n_train:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)


        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    else:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test



class DynamicalSystemsDataset(Dataset):


    def __init__(self, X: np.ndarray, y: np.ndarray):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]




def prepare_data(
        data_root: str = None,
        seq_len: int = DEFAULT_SEQ_LEN,
        stride: int = None,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        batch_size: int = 64,
        max_files_per_class: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True
) -> Tuple[DataLoader, DataLoader, dict]:

    if verbose:
        print("=" * 60)
        print("PIPELINE PRZYGOTOWANIA DANYCH")
        print("=" * 60)

    if verbose:
        print("\n[1/5] Ładowanie danych z plików...")
    all_data, stats = load_all_data(
        data_root=data_root,
        max_files_per_class=max_files_per_class,
        verbose=verbose
    )

    # --- POPRAWKA DODANA PRZEZ TWOJEGO ZNAJOMEGO ---
    if verbose:
        print(f"\n[2/5] Podział na zbiory (train={train_ratio:.0%}, test={1 - train_ratio:.0%}) na poziomie plików...")
    train_data, test_data = split_data_files(all_data, train_ratio=train_ratio)

    if verbose:
        print(f"\n[3/5] Przetwarzanie danych (seq_len={seq_len})...")
    
    if verbose: print("  Przetwarzanie zbioru treningowego:")
    X_train, y_train = preprocess_all_data(train_data, seq_len=seq_len, stride=stride, verbose=verbose)
    
    if verbose: print("  Przetwarzanie zbioru testowego:")
    X_test, y_test = preprocess_all_data(test_data, seq_len=seq_len, stride=stride, verbose=verbose)

    if verbose:
        print(f"  Treningowy: X={X_train.shape}, y={y_train.shape}")
        print(f"  Testowy:    X={X_test.shape}, y={y_test.shape}")
    # -----------------------------------------------

    scaler = None
    if normalize:
        if verbose:
            print(f"\n[4/5] Normalizacja danych (StandardScaler)...")
        X_train, X_test, scaler = normalize_data(X_train, X_test)
        if verbose:
            print(f"  Średnia train: {X_train.mean(axis=(0, 1))}")
            print(f"  Std train:     {X_train.std(axis=(0, 1))}")
    else:
        if verbose:
            print(f"\n[4/5] Normalizacja pominięta.")

    if verbose:
        print(f"\n[5/5] Tworzenie DataLoaderów (batch_size={batch_size})...")

    train_dataset = DynamicalSystemsDataset(X_train, y_train)
    test_dataset = DynamicalSystemsDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    info = {
        "num_classes": len(CLASS_TO_IDX),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {v: k for k, v in CLASS_TO_IDX.items()},
        "seq_len": seq_len,
        "num_features": X_train.shape[2],
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "scaler": scaler,
        "files_per_class": stats,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"GOTOWE!")
        print(f"  Klasy:          {list(CLASS_TO_IDX.keys())}")
        print(f"  Wymiar wejścia: ({seq_len}, {info['num_features']})")
        print(f"  Zbiór train:    {info['train_size']} próbek")
        print(f"  Zbiór test:     {info['test_size']} próbek")
        print(f"  Batchy train:   {len(train_loader)}")
        print(f"  Batchy test:    {len(test_loader)}")
        print(f"{'=' * 60}")

    return train_loader, test_loader, info



if __name__ == "__main__":

    train_loader, test_loader, info = prepare_data(
        seq_len=200,
        stride=100,
        train_ratio=0.8,
        batch_size=64,
        max_files_per_class=None,
        normalize=True,
        verbose=True
    )


    print("\n--- Przykładowy batch ---")
    for batch_X, batch_y in train_loader:
        print(f"  X shape: {batch_X.shape}")
        print(f"  y shape: {batch_y.shape}")  
        print(f"  y values: {batch_y[:10]}")
        print(f"  X dtype: {batch_X.dtype}")
        print(f"  y dtype: {batch_y.dtype}")
        print(f"  X[0, :5, :] (pierwsze 5 kroków, wszystkie cechy):")
        print(f"  {batch_X[0, :5, :]}")
        break
