import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import glob
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_pipeline import (
    discover_data_dirs,
    load_single_file,
    pad_to_max_dim,
    create_segments,
    CLASS_TO_IDX
)


# ---------------------------------------------------------------------------
# Model CNN 1D
# ---------------------------------------------------------------------------

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
        # DataLoader zwraca dane jako:
        # (batch, seq_len, features)
        #
        # Conv1d wymaga:
        # (batch, features, seq_len)
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


# ---------------------------------------------------------------------------
# Tworzenie segmentów z konkretnych plików
# ---------------------------------------------------------------------------

def make_segments_from_files(file_paths, class_name, seq_len, stride):
    X_segments = []
    y_labels = []

    label = CLASS_TO_IDX[class_name]

    for file_path in file_paths:
        data = load_single_file(file_path)

        if data is None:
            continue

        data = pad_to_max_dim(data, max_dim=3)

        segments = create_segments(
            time_series=data,
            seq_len=seq_len,
            stride=stride
        )

        for segment in segments:
            X_segments.append(segment)
            y_labels.append(label)

    return X_segments, y_labels


# ---------------------------------------------------------------------------
# Poprawiony pipeline danych: podział na pliki, a nie na segmenty
# ---------------------------------------------------------------------------

def prepare_data_file_split(
    seq_len=200,
    stride=100,
    batch_size=64,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    normalize=True,
    seed=42,
    data_root=None
):
    print("=" * 60)
    print("PIPELINE BEZ WYCIEKU DANYCH")
    print("Podział jest robiony na poziomie plików, nie segmentów.")
    print("=" * 60)

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio musi dawać 1.0")

    rng = np.random.default_rng(seed)

    data_dirs = discover_data_dirs(data_root)

    train_files_by_class = {}
    val_files_by_class = {}
    test_files_by_class = {}

    for class_name, dir_path in data_dirs.items():
        files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
        files = np.array(files, dtype=object)

        rng.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        train_files_by_class[class_name] = train_files
        val_files_by_class[class_name] = val_files
        test_files_by_class[class_name] = test_files

        print(
            f"{class_name:>10}: "
            f"train files={len(train_files):4d}, "
            f"val files={len(val_files):4d}, "
            f"test files={len(test_files):4d}"
        )

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for class_name in data_dirs.keys():
        X_part, y_part = make_segments_from_files(
            train_files_by_class[class_name],
            class_name,
            seq_len,
            stride
        )
        X_train.extend(X_part)
        y_train.extend(y_part)

        X_part, y_part = make_segments_from_files(
            val_files_by_class[class_name],
            class_name,
            seq_len,
            stride
        )
        X_val.extend(X_part)
        y_val.extend(y_part)

        X_part, y_part = make_segments_from_files(
            test_files_by_class[class_name],
            class_name,
            seq_len,
            stride
        )
        X_test.extend(X_part)
        y_test.extend(y_part)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)

    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    print("\nSegmenty po podziale plików:")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Jeden ze zbiorów jest pusty. Sprawdź seq_len, stride albo liczbę plików.")

    if normalize:
        n_features = X_train.shape[2]

        X_train_2d = X_train.reshape(-1, n_features)

        mean = X_train_2d.mean(axis=0)
        std = X_train_2d.std(axis=0)
        std_safe = np.where(std == 0, 1.0, std)

        os.makedirs("models", exist_ok=True)
        np.savez("models/cnn_scaler_fixed.npz", mean=mean, std_safe=std_safe)

        X_train = ((X_train - mean) / std_safe).astype(np.float32)
        X_val = ((X_val - mean) / std_safe).astype(np.float32)
        X_test = ((X_test - mean) / std_safe).astype(np.float32)

        print("\nNormalizacja wykonana tylko na podstawie zbioru TRAIN.")
        print("Parametry normalizacji zapisano jako: models/cnn_scaler_fixed.npz")
        #Normalizacja każdego segmentu z osobna (Z-score dla każdego okna)
    #X_train ma kształt: (liczba_segmentów, seq_len, n_features)
    
    # Obliczamy średnią i std dla każdego segmentu i każdej osi niezależnie
        # mean = X_train.mean(axis=1, keepdims=True)  # kształt: (N, 1, 3)
        # std = X_train.std(axis=1, keepdims=True)    # kształt: (N, 1, 3)
        # std_safe = np.where(std == 0, 1.0, std)
        # X_train = (X_train - mean) / std_safe

        # # To samo robimy dla VAL i TEST (każdy segment normalizuje się sam ze sobą!)
        # X_val = (X_val - X_val.mean(axis=1, keepdims=True)) / np.where(X_val.std(axis=1, keepdims=True) == 0, 1.0, X_val.std(axis=1, keepdims=True))
        # X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / np.where(X_test.std(axis=1, keepdims=True) == 0, 1.0, X_test.std(axis=1, keepdims=True))

        # print("\nNormalizacja wykonana niezależnie dla każdego pojedynczego segmentu (Per-Instance).")
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    info = {
        "num_features": X_train.shape[2],
        "num_classes": len(CLASS_TO_IDX),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {v: k for k, v in CLASS_TO_IDX.items()},
        "seq_len": seq_len,
        "stride": stride,
        "train_files_per_class": {k: len(v) for k, v in train_files_by_class.items()},
        "val_files_per_class": {k: len(v) for k, v in val_files_by_class.items()},
        "test_files_per_class": {k: len(v) for k, v in test_files_by_class.items()},
        "train_segments": len(train_dataset),
        "val_segments": len(val_dataset),
        "test_segments": len(test_dataset),
    }

    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# Ewaluacja accuracy
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0

    return 100 * correct / total


# ---------------------------------------------------------------------------
# Trenowanie
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=0.0003
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"\nRozpoczynam uczenie na urządzeniu: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    os.makedirs("models", exist_ok=True)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total if total > 0 else 0.0

        val_accuracy = evaluate_accuracy(model, val_loader, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "models/best_cnn_model.pth")
            print("  --> Zapisano najlepszy model według walidacji.")

        print(
            f"Epoka [{epoch + 1}/{num_epochs}] | "
            f"Loss: {running_loss / len(train_loader):.4f} | "
            f"Train acc: {train_accuracy:.2f}% | "
            f"Val acc: {val_accuracy:.2f}%"
        )

    training_time = time.time() - start_time

    print("=" * 60)
    print("Uczenie zakończone.")
    print(f"Najlepsza dokładność walidacyjna: {best_val_accuracy:.2f}%")
    print(f"Czas uczenia: {training_time:.2f} s")

    return model, training_time, best_val_accuracy


# ---------------------------------------------------------------------------
# Raport klasyfikacji bez sklearn
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader, info):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    num_classes = info["num_classes"]
    idx_to_class = info["idx_to_class"]

    confusion = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            labels_np = labels.cpu().numpy()
            preds_np = predicted.cpu().numpy()

            for true_label, pred_label in zip(labels_np, preds_np):
                confusion[true_label, pred_label] += 1

    print("\nConfusion matrix na zbiorze TEST:")
    print(confusion)

    print("\nRaport klasyfikacji na zbiorze TEST:")
    print(f"{'Klasa':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")

    total_correct = 0
    total_samples = 0

    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        support = confusion[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        total_correct += tp
        total_samples += support

        print(
            f"{idx_to_class[i]:<12} "
            f"{precision:>10.4f} "
            f"{recall:>10.4f} "
            f"{f1_score:>10.4f} "
            f"{support:>10}"
        )

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print("\nAccuracy test:", f"{accuracy * 100:.2f}%")

    return confusion, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # FILTERS = 32
    # KERNEL_SIZE = 5
    # DROPOUT = 0.2
    FILTERS = 32
    KERNEL_SIZE = 5
    DROPOUT = 0.2

    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0003

    # Dla porównania z LSTM zostaw:
    SEQ_LEN = 100
    STRIDE = 50

    # Do dodatkowych eksperymentów możesz później zmienić np.:
    # SEQ_LEN = 500
    # STRIDE = 250

    BATCH_SIZE = 64

    print("Ładowanie i przygotowywanie danych bez wycieku...")

    train_loader, val_loader, test_loader, info = prepare_data_file_split(
        seq_len=SEQ_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize=True,
        seed=42
    )

    print("\nPliki na klasę:")
    print("Train:", info["train_files_per_class"])
    print("Val:  ", info["val_files_per_class"])
    print("Test: ", info["test_files_per_class"])

    print("\nSegmenty:")
    print("Train:", info["train_segments"])
    print("Val:  ", info["val_segments"])
    print("Test: ", info["test_segments"])

    print("\nInicjalizacja modelu CNN...")

    model = DynamicSystemCNN(
        input_size=info["num_features"],
        num_classes=info["num_classes"],
        filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    )

    trained_model, training_time, best_val_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    print("\nŁadowanie najlepszego modelu według walidacji...")

    trained_model.load_state_dict(
        torch.load("models/best_cnn_model.pth", map_location="cpu")
    )

    confusion, test_accuracy = evaluate_model(
        trained_model,
        test_loader,
        info
    )

    print("\nPodsumowanie:")
    print(f"Model: CNN 1D")
    print(f"Seq len: {SEQ_LEN}")
    print(f"Stride: {STRIDE}")
    print(f"Filtry: {FILTERS}")
    print(f"Kernel size: {KERNEL_SIZE}")
    print(f"Dropout: {DROPOUT}")
    print(f"Epoki: {NUM_EPOCHS}")
    print(f"Najlepsza walidacja: {best_val_accuracy:.2f}%")
    print(f"Accuracy test: {test_accuracy * 100:.2f}%")
    print(f"Czas uczenia: {training_time:.2f} s")
    print("\nSkończone. Najlepszy model zapisany jako: models/best_cnn_model.pth")