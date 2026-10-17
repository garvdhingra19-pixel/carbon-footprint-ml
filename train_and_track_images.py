import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker

def measure_block(func, project_name):
    tracker = EmissionsTracker(project_name=project_name, output_dir="output", log_level="error")
    tracker.start()
    t0 = time.time()
    result = func()
    duration = time.time() - t0
    emissions_g = (tracker.stop() * 1000)
    return result, emissions_g, duration

def load_cifar_subset(train_size=5000, test_size=1000):
    """Loads a small subset of CIFAR-10 and flattens images for standard ML models."""
    print("Downloading/Loading CIFAR-10 Dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Take a subset to keep CPU training times reasonable
    train_subset = Subset(train_set, range(train_size))
    test_subset = Subset(test_set, range(test_size))
    
    # Extract tensors
    X_train = torch.stack([x for x, y in train_subset]).numpy()
    y_train = np.array([y for x, y in train_subset])
    X_test = torch.stack([x for x, y in test_subset]).numpy()
    y_test = np.array([y for x, y in test_subset])
    
    # Flatten 3D images (3x32x32) into 1D arrays (3072 features) for Sklearn/LightGBM
    X_train_flat = X_train.reshape(train_size, -1)
    X_test_flat = X_test.reshape(test_size, -1)
    
    return X_train_flat, X_test_flat, y_train, y_test, X_train, X_test

# --- Simple CNN for PyTorch ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def run_image_experiment():
    X_train_flat, X_test_flat, y_train, y_test, X_train_img, X_test_img = load_cifar_subset()
    results = []
    
    print("\n--- Training Logistic Regression (Images) ---")
    lr = LogisticRegression(max_iter=100)
    _, co2, dur = measure_block(lambda: lr.fit(X_train_flat, y_train), "lr_img")
    acc = accuracy_score(y_test, lr.predict(X_test_flat)) * 100
    results.append({"Model": "Logistic Regression", "Accuracy (%)": acc, "Emissions (g)": co2, "Time (s)": dur})

    print("--- Training LightGBM (Images) ---")
    gbm = lgb.LGBMClassifier(n_estimators=50, verbosity=-1)
    _, co2, dur = measure_block(lambda: gbm.fit(X_train_flat, y_train), "lgb_img")
    acc = accuracy_score(y_test, gbm.predict(X_test_flat)) * 100
    results.append({"Model": "LightGBM", "Accuracy (%)": acc, "Emissions (g)": co2, "Time (s)": dur})

    print("--- Training PyTorch CNN (Images) ---")
    device = torch.device("cpu")
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    X_tr_t = torch.tensor(X_train_img, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    def train_cnn():
        net.train()
        for _ in range(5): # 5 epochs
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(net(bx), by)
                loss.backward()
                optimizer.step()
                
    _, co2, dur = measure_block(train_cnn, "cnn_img")
    
    net.eval()
    with torch.no_grad():
        preds = torch.max(net(torch.tensor(X_test_img, dtype=torch.float32)), 1)[1].numpy()
        acc = accuracy_score(y_test, preds) * 100
    results.append({"Model": "PyTorch CNN", "Accuracy (%)": acc, "Emissions (g)": co2, "Time (s)": dur})

    # Save and display
    df = pd.DataFrame(results)
    df.to_csv("experiment_results_images.csv", index=False)
    print("\n=== Image Dataset Results (CIFAR-10 Subset) ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_image_experiment()