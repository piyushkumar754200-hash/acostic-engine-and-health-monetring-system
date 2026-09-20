import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from app.config import MODELS_DIR

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        # Take output of last timestep
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel:
    def __init__(self, hidden_units: int = 64, num_layers: int = 2, learning_rate: float = 0.001, epochs: int = 15, batch_size: int = 16):
        self.model_id = "lstm"
        self.name = "LSTM Recurrent Neural Network"
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = None
        self.is_trained = False
        self.classes_ = []
        self.input_size = None

    def train(self, X_seq: np.ndarray, y: np.ndarray, classes: list):
        # X_seq shape: (N, seq_len, num_features)
        self.classes_ = classes
        num_classes = len(classes)
        self.input_size = X_seq.shape[2]
        
        self.net = LSTMNet(self.input_size, self.hidden_units, self.num_layers, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self.is_trained = True
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.net is None:
            raise ValueError("Model is not trained.")
        if len(X_seq.shape) == 2:
            X_seq = np.expand_dims(X_seq, axis=0) # (1, seq_len, num_features)
            
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            logits = self.net(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_seq)
        return np.argmax(probs, axis=1)

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "lstm_model.pth")
        if self.is_trained:
            torch.save({
                'state_dict': self.net.state_dict(),
                'input_size': self.input_size,
                'hidden_units': self.hidden_units,
                'num_layers': self.num_layers,
                'classes': self.classes_
            }, filepath)

    def load(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "lstm_model.pth")
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            self.input_size = checkpoint['input_size']
            self.hidden_units = checkpoint['hidden_units']
            self.num_layers = checkpoint['num_layers']
            self.classes_ = checkpoint['classes']
            self.net = LSTMNet(self.input_size, self.hidden_units, self.num_layers, len(self.classes_))
            self.net.load_state_dict(checkpoint['state_dict'])
            self.is_trained = True
            return True
        return False
