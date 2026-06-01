import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline import prepare_data

# ---------------------------------------------------------------------------
# Zdefiniowanie architektury modelu LSTM
# ---------------------------------------------------------------------------
class DynamicSystemLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        Klasa naszego modelu LSTM do rozpoznawania układów dynamicznych.
        
        Parametry:
        - input_size: Ilość cech w każdym kroku czasowym (w naszym przypadku 3 wymiary)
        - hidden_size: Ilość "neuronów" (cech ukrytych) wewnątrz LSTM, decyduje o pojemności pamięci
        - num_layers: Liczba warstw LSTM (np. 2 oznacza, że drugie LSTM patrzy na wyjście pierwszego)
        - num_classes: Na ile klas dzielimy dane (u nas 5 układów)
        """
        super(DynamicSystemLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Warstwa LSTM
        # batch_first=True jest bardzo ważne, bo nasz DataLoader zwraca dane 
        # w kształcie: (rozmiar_batcha, długość_sekwencji, liczba_cech)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 2. Warstwa w pełni połączona (tzw. Dense/Linear) 
        # Przerabia to, co LSTM "zrozumiało" (o rozmiarze hidden_size) na 5 naszych klas.
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Ta metoda definiuje, jak dane (x) przepływają przez sieć (od wejścia do wyjścia).
        """
        # x wchodzi do LSTM. 
        # Zwracane są dwie rzeczy: `out` (wyjścia dla każdego kroku) 
        # i stany ukryte (które tu nas nie obchodzą, więc ignorujemy `_`)
        out, _ = self.lstm(x)
        
        # `out` ma kształt: (batch_size, sekwencja_czasowa, hidden_size).
        # My chcemy dokonać klasyfikacji po obejrzeniu CAŁEJ sekwencji.
        # Dlatego bierzemy tylko wynik z ostatniego kroku czasowego: `[:, -1, :]`
        out = out[:, -1, :] 
        
        # Przepuszczamy ten ostatni wynik przez klasyfikator liniowy
        out = self.fc(out)
        
        return out

# ---------------------------------------------------------------------------
# Funkcja trenująca model
# ---------------------------------------------------------------------------
def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    """
    Funkcja, która uczy nasz model.
    """
    # Przenosimy model na kartę graficzną (GPU), jeśli jest dostępna (znacznie przyspiesza)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Rozpoczynam uczenie na urządzeniu: {device}")
    
    # Funkcja straty (Loss) do klasyfikacji wieloklasowej
    criterion = nn.CrossEntropyLoss()
    
    # Optymalizator (Adam to standardowy, dobry i szybki wybór)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Zapisujemy najlepszą skuteczność, by na koniec wiedzieć, jak dobrze poszło
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train() # Przełączamy model w tryb treningowy
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Pętla po wszystkich paczkach danych (batchach) treningowych
        for inputs, labels in train_loader:
            # Przenosimy dane na to samo urządzenie co model (GPU/CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Krok 1: Wyzerowanie gradientów z poprzedniego kroku
            optimizer.zero_grad()
            
            # Krok 2: Przepuszczenie danych przez model (Forward pass)
            outputs = model(inputs)
            
            # Krok 3: Obliczenie błędu (Loss)
            loss = criterion(outputs, labels)
            
            # Krok 4: Policzenie gradientów (Backward pass - gdzie model dowiaduje się, co sknocił)
            loss.backward()
            
            # Krok 5: Aktualizacja wag modelu (Optymalizacja)
            optimizer.step()
            
            # Zbieranie statystyk (strat i dokładności do wyświetlenia)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # Bierzemy klasę z największym prawdopodobieństwem
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_accuracy = 100 * correct / total
        
        # Walidacja (Sprawdzanie na danych testowych)
        model.eval() # Przełączamy na tryb ewaluacji (wyłącza pewne funkcje uczące)
        test_correct = 0
        test_total = 0
        
        # 'with torch.no_grad()' mówi, żeby PyTorch nie liczył tu gradientów, 
        # bo to tylko test i chcemy zaoszczędzić pamięć i czas.
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        test_accuracy = 100 * test_correct / test_total
        
        # Aktualizacja najlepszego wyniku i zapisanie modelu
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # Tworzymy folder 'models' jeśli jeszcze go nie ma
            os.makedirs('models', exist_ok=True)
            # Zapisujemy najlepszy model na dysk
            torch.save(model.state_dict(), 'models/best_lstm_model.pth')
            print("  --> Zapisano nowy, lepszy model na dysku (models/best_lstm_model.pth)!")
        print(f"Epoka [{epoch+1}/{num_epochs}] | "
              f"Strata(Loss): {running_loss/len(train_loader):.4f} | "
              f"Dokładność Treningowa: {train_accuracy:.2f}% | "
              f"Dokładność Testowa: {test_accuracy:.2f}%")

    print("=" * 60)
    print(f"Uczenie zakończone! Najlepsza dokładność testowa: {best_accuracy:.2f}%")
    return model


if __name__ == "__main__":
    # Parametry modelu
    HIDDEN_SIZE = 64     # Ile "neuronów" w pamięci
    NUM_LAYERS = 2       # Ile warstw LSTM
    NUM_EPOCHS = 10      # Ile pełnych przejść przez dane treningowe
    LEARNING_RATE = 0.001
    
    # 1. Pobieranie danych z Twojego data_pipeline.py
    # UWAGA: limituję liczbę plików na klasę do (np. 10), żebyś przy testach 
    # nie musiał czekać godziny na załadowanie wszystkiego (możesz usunąć `max_files_per_class=10` by puścić całość).
    print("Ładowanie i przygotowywanie danych...")
    train_loader, test_loader, info = prepare_data(
        seq_len=200,          # Długość fragmentu czasowego
        stride=100,           # Krok przesunięcia okna
        batch_size=64,        # Rozmiar pojedynczej paczki paczki danych
        # max_files_per_class=10, # Odkomentuj to do szybkich testów!
        normalize=True,
        verbose=False         # Żeby nie zaśmiecać konsoli przy każdym włączeniu
    )
    
    # Pobieramy informacje z przygotowanych danych
    INPUT_SIZE = info['num_features']
    NUM_CLASSES = info['num_classes']
    
    # 2. Utworzenie instancji naszego modelu LSTM
    print("\nInicjalizacja modelu LSTM...")
    model = DynamicSystemLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    )
    
    # 3. Uruchomienie trenowania!
    print("Rozpoczynamy cykl uczący!")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    print("Skończyliśmy! Plik z wagami najlepszego modelu to 'models/best_lstm_model.pth'")
