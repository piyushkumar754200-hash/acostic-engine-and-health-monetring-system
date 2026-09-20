import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from app.config import MODELS_DIR

class CNN2DNet(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN2DNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        # x: (batch_size, 1, H, W) e.g., 128x128
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN2DModel:
    def __init__(self, learning_rate: float = 0.001, epochs: int = 12, batch_size: int = 16):
        self.model_id = "cnn_2d"
        self.name = "2D Mel-Spectrogram CNN"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = None
        self.is_trained = False
        self.classes_ = []

    def train(self, X_mel: np.ndarray, y: np.ndarray, classes: list):
        # X_mel shape: (N, 128, 128)
        self.classes_ = classes
        num_classes = len(classes)
        self.net = CNN2DNet(num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X_mel, dtype=torch.float32).unsqueeze(1) # (N, 1, 128, 128)
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

    def predict_proba(self, X_mel: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.net is None:
            raise ValueError("Model is not trained.")
        if len(X_mel.shape) == 2:
            X_mel = np.expand_dims(X_mel, axis=0) # (1, 128, 128)
            
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_mel, dtype=torch.float32).unsqueeze(1)
            logits = self.net(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X_mel: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_mel)
        return np.argmax(probs, axis=1)

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "cnn_2d_model.pth")
        if self.is_trained:
            torch.save({
                'state_dict': self.net.state_dict(),
                'classes': self.classes_
            }, filepath)

    def load(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "cnn_2d_model.pth")
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            self.classes_ = checkpoint['classes']
            self.net = CNN2DNet(len(self.classes_))
            self.net.load_state_dict(checkpoint['state_dict'])
            self.is_trained = True
            return True
        return False
