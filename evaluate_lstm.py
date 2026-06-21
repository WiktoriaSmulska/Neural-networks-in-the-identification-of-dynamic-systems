import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_pipeline import prepare_data
from lstm_model import DynamicSystemLSTM


MODEL_PATH = "models/best_lstm_model.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, info = prepare_data(
        seq_len=200,
        stride=100,
        batch_size=64,
        normalize=True,
        verbose=False
    )

    model = DynamicSystemLSTM(
        input_size=info["num_features"],
        hidden_size=64,
        num_layers=2,
        num_classes=info["num_classes"]
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    class_names = [info["idx_to_class"][i] for i in range(info["num_classes"])]

    accuracy = accuracy_score(all_labels, all_preds)
    matrix = confusion_matrix(all_labels, all_preds)

    print("=" * 60)
    print("EWALUACJA MODELU LSTM")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Accuracy test: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    print("\nConfusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()