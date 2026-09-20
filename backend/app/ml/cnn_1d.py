import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from app.config import MODELS_DIR

class CNN1DNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(CNN1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * (input_dim // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 1, input_dim)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNN1DModel:
    def __init__(self, learning_rate: float = 0.001, epochs: int = 15, batch_size: int = 16):
        self.model_id = "cnn_1d"
        self.name = "1D Convolutional Neural Network"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = None
        self.is_trained = False
        self.classes_ = []
        self.input_dim = None

    def train(self, X: np.ndarray, y: np.ndarray, classes: list):
        self.classes_ = classes
        num_classes = len(classes)
        self.input_dim = X.shape[1]
        
        self.net = CNN1DNet(self.input_dim, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # (N, 1, input_dim)
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.net is None:
            raise ValueError("Model is not trained.")
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            logits = self.net(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "cnn_1d_model.pth")
        if self.is_trained:
            torch.save({
                'state_dict': self.net.state_dict(),
                'input_dim': self.input_dim,
                'classes': self.classes_
            }, filepath)

    def load(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "cnn_1d_model.pth")
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            self.input_dim = checkpoint['input_dim']
            self.classes_ = checkpoint['classes']
            self.net = CNN1DNet(self.input_dim, len(self.classes_))
            self.net.load_state_dict(checkpoint['state_dict'])
            self.is_trained = True
            return True
        return False
