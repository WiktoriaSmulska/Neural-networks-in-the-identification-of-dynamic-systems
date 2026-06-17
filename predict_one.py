import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import numpy as np
import torch
import torch.nn as nn

from data_pipeline import (
    load_single_file,
    pad_to_max_dim,
    create_segments,
    CLASS_TO_IDX
)


IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ============================================================
# PARAMETRY MUSZĄ BYĆ IDENTYCZNE JAK W TRENINGU
# ============================================================

FILTERS = 32
KERNEL_SIZE = 5
DROPOUT = 0.2

SEQ_LEN = 200
STRIDE = 100

MODEL_PATH = "models/best_cnn_model_fixed.pth"
SCALER_PATH = "models/cnn_scaler_fixed.npz"


# ============================================================
# ARCHITEKTURA CNN — taka sama jak w cnn_model.py
# ============================================================

class DynamicSystemCNN(nn.Module):
    def __init__(self, input_size, num_classes, filters=32, kernel_size=5, dropout=0.2):
        super(DynamicSystemCNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.bn2 = nn.BatchNorm1d(filters * 2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * 2, num_classes)

    def forward(self, x):
        # wejście: (batch, seq_len, features)
        # Conv1d wymaga: (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)

        x = self.dropout(x)
        x = self.fc(x)

        return x


# ============================================================
# NORMALIZACJA GLOBALNA — taka jak w treningu
# ============================================================

def normalize_global(X):
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Nie znaleziono pliku skalera: {SCALER_PATH}\n"
            f"Najpierw uruchom trening, żeby zapisał cnn_scaler_fixed.npz."
        )

    scaler = np.load(SCALER_PATH)

    mean = scaler["mean"]
    std_safe = scaler["std_safe"]

    X = ((X - mean) / std_safe).astype(np.float32)

    return X


# ============================================================
# PREDYKCJA JEDNEGO PLIKU
# ============================================================

def predict_file(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PREDYKCJA JEDNEGO PRZEBIEGU CZASOWEGO")
    print("=" * 70)
    print(f"Plik: {file_path}")
    print(f"Urządzenie: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")
    print(f"SEQ_LEN={SEQ_LEN}, STRIDE={STRIDE}")
    print(f"FILTERS={FILTERS}, KERNEL_SIZE={KERNEL_SIZE}, DROPOUT={DROPOUT}")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Wczytanie pliku
    # ------------------------------------------------------------

    data = load_single_file(file_path)

    if data is None:
        raise ValueError("Nie udało się wczytać pliku albo plik jest za krótki.")

    print("\n[1] Dane wejściowe:")
    print(f"Raw shape: {data.shape}")

    if data.shape[1] == 2:
        print("Wykryto 2 cechy — prawdopodobnie Ueda: x, y.")
        print("Trzecia kolumna zostanie uzupełniona zerami, tak jak podczas treningu.")
    elif data.shape[1] == 3:
        print("Wykryto 3 cechy: x, y, z.")
    else:
        print(f"Uwaga: wykryto nietypową liczbę cech: {data.shape[1]}.")

    print("Pierwsze 5 wierszy:")
    print(data[:5])

    # ------------------------------------------------------------
    # 2. Padding do 3 cech
    # ------------------------------------------------------------

    data = pad_to_max_dim(data, max_dim=3)

    print("\n[2] Po pad_to_max_dim:")
    print(f"Shape: {data.shape}")
    print(f"Min:  {data.min(axis=0)}")
    print(f"Max:  {data.max(axis=0)}")
    print(f"Mean: {data.mean(axis=0)}")
    print(f"Std:  {data.std(axis=0)}")

    # ------------------------------------------------------------
    # 3. Segmentacja
    # ------------------------------------------------------------

    segments = create_segments(
        time_series=data,
        seq_len=SEQ_LEN,
        stride=STRIDE
    )

    if len(segments) == 0:
        raise ValueError(
            f"Plik jest za krótki. Ma {data.shape[0]} próbek, "
            f"a wymagane minimum to SEQ_LEN={SEQ_LEN}."
        )

    X = np.array(segments, dtype=np.float32)

    print("\n[3] Segmentacja:")
    print(f"Liczba segmentów: {len(segments)}")
    print(f"X shape przed normalizacją: {X.shape}")

    # ------------------------------------------------------------
    # 4. Normalizacja globalna
    # ------------------------------------------------------------

    X = normalize_global(X)

    print("\n[4] Po normalizacji globalnej:")
    print(f"X shape: {X.shape}")
    print(f"Min:  {X.min(axis=(0, 1))}")
    print(f"Max:  {X.max(axis=(0, 1))}")
    print(f"Mean: {X.mean(axis=(0, 1))}")
    print(f"Std:  {X.std(axis=(0, 1))}")

    # ------------------------------------------------------------
    # 5. Tensor
    # ------------------------------------------------------------

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # ------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_PATH}")

    model = DynamicSystemCNN(
        input_size=3,
        num_classes=len(CLASS_TO_IDX),
        filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # 7. Predykcja
    # ------------------------------------------------------------

    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

    # ------------------------------------------------------------
    # 8. Wyniki dla segmentów
    # ------------------------------------------------------------

    print("\n[5] Predykcje segmentów:")

    for i, pred in enumerate(predictions):
        probs = probabilities[i].cpu().numpy()

        print(f"\nSegment {i + 1}: {IDX_TO_CLASS[pred]}")
        for class_idx, prob in enumerate(probs):
            print(f"  {IDX_TO_CLASS[class_idx]:<12}: {prob:.4f}")

    # ------------------------------------------------------------
    # 9. Głosowanie większościowe
    # ------------------------------------------------------------

    counts = np.bincount(predictions, minlength=len(CLASS_TO_IDX))
    final_class_idx = int(np.argmax(counts))
    final_class_name = IDX_TO_CLASS[final_class_idx]

    avg_probabilities = probabilities.mean(dim=0).cpu().numpy()
    avg_class_idx = int(np.argmax(avg_probabilities))
    avg_class_name = IDX_TO_CLASS[avg_class_idx]

    print("\n" + "=" * 70)
    print("PODSUMOWANIE")
    print("=" * 70)

    print(f"Klasa według głosowania segmentów: {final_class_name}")
    print(f"Klasa według średnich prawdopodobieństw: {avg_class_name}")

    print("\nGłosowanie segmentów:")
    for idx, count in enumerate(counts):
        print(f"{IDX_TO_CLASS[idx]:<12}: {count}")

    print("\nŚrednie prawdopodobieństwa:")
    for idx, prob in enumerate(avg_probabilities):
        print(f"{IDX_TO_CLASS[idx]:<12}: {prob:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Użycie:")
        print("python predict_one.py ścieżka_do_pliku.txt")
        print()
        print("Przykład:")
        print('python predict_one.py "lorenz.txt"')
        sys.exit(1)

    file_path = sys.argv[1]
    predict_file(file_path)