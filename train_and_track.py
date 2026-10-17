import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from codecarbon import EmissionsTracker

from dataset import load_data
from hardware_probe import get_system_metadata

# Neural Network Architecture
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

def measure_block(func, project_name):
    """Runs a function under CodeCarbon tracking and returns (result, emissions_grams, duration_seconds)."""
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir="output",
        log_level="error"
    )
    tracker.start()
    t0 = time.time()
    result = func()
    duration = time.time() - t0
    emissions_kg = tracker.stop()
    emissions_g = (emissions_kg * 1000) if emissions_kg else 0.0
    return result, emissions_g, duration

def run_rigorous_benchmark(n_runs=5):
    print("Loading tabular dataset...")
    X_train, X_test, y_train, y_test = load_data(n_samples=25000, n_features=30)
    
    # Generate 10k fixed inference batch
    indices = np.random.choice(len(X_test), size=10000, replace=True)
    X_infer_10k = X_test[indices]

    results = []
    
    # -------------------------------------------------------------
    # Model 1: Logistic Regression
    # -------------------------------------------------------------
    print(f"\n--- Profiling Logistic Regression ({n_runs} runs) ---")
    train_accs, train_co2, train_times = [], [], []
    infer_co2, infer_times = [], []

    for r in range(n_runs):
        clf = LogisticRegression(max_iter=500, random_state=42 + r)
        _, co2_tr, t_tr = measure_block(lambda: clf.fit(X_train, y_train), "lr_train")
        train_co2.append(co2_tr)
        train_times.append(t_tr)
        train_accs.append(accuracy_score(y_test, clf.predict(X_test)) * 100)

        _, co2_inf, t_inf = measure_block(lambda: clf.predict(X_infer_10k), "lr_infer")
        infer_co2.append(co2_inf)
        infer_times.append(t_inf)

    results.append({
        "Model": "Logistic Regression",
        "Train Acc Mean (%)": np.mean(train_accs),
        "Train Acc Std (%)": np.std(train_accs),
        "Train CO2 Mean (g)": np.mean(train_co2),
        "Train CO2 Std (g)": np.std(train_co2),
        "Train Time Mean (s)": np.mean(train_times),
        "Infer CO2 (10k) Mean (g)": np.mean(infer_co2),
        "Infer Time (10k) Mean (s)": np.mean(infer_times)
    })

    # -------------------------------------------------------------
    # Model 2: Random Forest
    # -------------------------------------------------------------
    print(f"\n--- Profiling Random Forest ({n_runs} runs) ---")
    train_accs, train_co2, train_times = [], [], []
    infer_co2, infer_times = [], []

    for r in range(n_runs):
        rf = RandomForestClassifier(n_estimators=100, random_state=42 + r)
        _, co2_tr, t_tr = measure_block(lambda: rf.fit(X_train, y_train), "rf_train")
        train_co2.append(co2_tr)
        train_times.append(t_tr)
        train_accs.append(accuracy_score(y_test, rf.predict(X_test)) * 100)

        _, co2_inf, t_inf = measure_block(lambda: rf.predict(X_infer_10k), "rf_infer")
        infer_co2.append(co2_inf)
        infer_times.append(t_inf)

    results.append({
        "Model": "Random Forest",
        "Train Acc Mean (%)": np.mean(train_accs),
        "Train Acc Std (%)": np.std(train_accs),
        "Train CO2 Mean (g)": np.mean(train_co2),
        "Train CO2 Std (g)": np.std(train_co2),
        "Train Time Mean (s)": np.mean(train_times),
        "Infer CO2 (10k) Mean (g)": np.mean(infer_co2),
        "Infer Time (10k) Mean (s)": np.mean(infer_times)
    })

    # -------------------------------------------------------------
    # Model 3: LightGBM (Gradient Boosting)
    # -------------------------------------------------------------
    print(f"\n--- Profiling LightGBM ({n_runs} runs) ---")
    train_accs, train_co2, train_times = [], [], []
    infer_co2, infer_times = [], []

    for r in range(n_runs):
        gbm = lgb.LGBMClassifier(n_estimators=100, random_state=42 + r, verbosity=-1)
        _, co2_tr, t_tr = measure_block(lambda: gbm.fit(X_train, y_train), "lgbm_train")
        train_co2.append(co2_tr)
        train_times.append(t_tr)
        preds = gbm.predict(X_test)
        train_accs.append(accuracy_score(y_test, preds) * 100)

        _, co2_inf, t_inf = measure_block(lambda: gbm.predict(X_infer_10k), "lgbm_infer")
        infer_co2.append(co2_inf)
        infer_times.append(t_inf)

    results.append({
        "Model": "LightGBM",
        "Train Acc Mean (%)": np.mean(train_accs),
        "Train Acc Std (%)": np.std(train_accs),
        "Train CO2 Mean (g)": np.mean(train_co2),
        "Train CO2 Std (g)": np.std(train_co2),
        "Train Time Mean (s)": np.mean(train_times),
        "Infer CO2 (10k) Mean (g)": np.mean(infer_co2),
        "Infer Time (10k) Mean (s)": np.mean(infer_times)
    })

    # -------------------------------------------------------------
    # Model 4: PyTorch Neural Network
    # -------------------------------------------------------------
    print(f"\n--- Profiling PyTorch Neural Network ({n_runs} runs) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_infer_t = torch.tensor(X_infer_10k, dtype=torch.float32).to(device)

    train_accs, train_co2, train_times = [], [], []
    infer_co2, infer_times = [], []

    for r in range(n_runs):
        torch.manual_seed(42 + r)
        net = SimpleClassifier(X_train.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

        def train_nn():
            net.train()
            for _ in range(8):
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = net(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

        _, co2_tr, t_tr = measure_block(train_nn, "nn_train")
        train_co2.append(co2_tr)
        train_times.append(t_tr)

        net.eval()
        with torch.no_grad():
            out_test = net(X_test_t)
            acc = accuracy_score(y_test, torch.max(out_test, 1)[1].cpu().numpy()) * 100
            train_accs.append(acc)

        def infer_nn():
            net.eval()
            with torch.no_grad():
                _ = net(X_infer_t)

        _, co2_inf, t_inf = measure_block(infer_nn, "nn_infer")
        infer_co2.append(co2_inf)
        infer_times.append(t_inf)

    results.append({
        "Model": "PyTorch Neural Net",
        "Train Acc Mean (%)": np.mean(train_accs),
        "Train Acc Std (%)": np.std(train_accs),
        "Train CO2 Mean (g)": np.mean(train_co2),
        "Train CO2 Std (g)": np.std(train_co2),
        "Train Time Mean (s)": np.mean(train_times),
        "Infer CO2 (10k) Mean (g)": np.mean(infer_co2),
        "Infer Time (10k) Mean (s)": np.mean(infer_times)
    })

    # Save summary dataframe
    df = pd.DataFrame(results)
    df.to_csv("experiment_results_rigorous.csv", index=False)
    print("\n" + "=" * 65)
    print(" RIGOROUS BENCHMARK RESULTS (Mean +/- Std Dev across 5 runs)")
    print("=" * 65)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_rigorous_benchmark(n_runs=5)