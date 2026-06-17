import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_pipeline import prepare_data
from cnn_model import CNN1DClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    return accuracy, matrix, report


def train_one_model(
    filters=32,
    kernel_size=5,
    dropout=0.2,
    use_batch_norm=True,
    seq_len=200,
    epochs=30,
    batch_size=64,
    lr=0.001
):
    train_loader, test_loader, info = prepare_data(
        seq_len=seq_len,
        stride=seq_len // 2,
        train_ratio=0.8,
        batch_size=batch_size,
        normalize=True,
        verbose=True
    )

    model = CNN1DClassifier(
        input_channels=info["num_features"],
        num_classes=info["num_classes"],
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        use_batch_norm=use_batch_norm
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        test_acc, _, _ = evaluate_model(model, test_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Test accuracy: {test_acc:.4f}"
        )

    training_time = time.time() - start_time

    accuracy, matrix, report = evaluate_model(model, test_loader)

    print("\n==============================")
    print("WYNIKI KOŃCOWE CNN")
    print("==============================")
    print(f"Filtry: {filters}")
    print(f"Kernel size: {kernel_size}")
    print(f"Dropout: {dropout}")
    print(f"BatchNorm: {use_batch_norm}")
    print(f"Seq len: {seq_len}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Czas uczenia: {training_time:.2f} s")

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(matrix)

    torch.save(model.state_dict(), "models/best_cnn_model.pth")

    return {
        "filters": filters,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "batch_norm": use_batch_norm,
        "seq_len": seq_len,
        "accuracy": accuracy,
        "training_time": training_time,
        "losses": train_losses,
        "confusion_matrix": matrix
    }


if __name__ == "__main__":
    result = train_one_model(
        filters=32,
        kernel_size=5,
        dropout=0.2,
        use_batch_norm=True,
        seq_len=200,
        epochs=30,
        batch_size=64,
        lr=0.001
    )