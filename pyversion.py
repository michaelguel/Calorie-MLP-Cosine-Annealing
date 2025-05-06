import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def feature_engineering(df):
    df = df.rename(columns={'Gender': 'Sex'})
    df['Sex'] = df['Sex'].astype('category')
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['BMI_sq'] = df['BMI'] ** 2
    df['Age_Weight'] = df['Age'] * df['Weight']
    df['Duration_Weight'] = df['Duration'] * df['Weight']
    df['Sex_Duration'] = df['Duration'] * df['Sex'].cat.codes
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,18,30,50,100], labels=['Teen','Young','Adult','Senior']).astype('category')
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Sex_Adjusted_Duration'] = df['Duration'] * df['Sex'].cat.codes.map({0:0.85,1:1.0})
    df['Sex_Adjusted_BMI'] = df['BMI'] * df['Sex'].cat.codes.map({0:0.85,1:1.0})
    df['BMR'] = 10*df['Weight'] + 6.25*df['Height'] - 5*df['Age'] + df['Sex'].cat.codes.map({0:-161,1:5})
    df['LeanMass'] = df['Weight'] * df['Sex'].cat.codes.map({0:0.72,1:0.82})
    df['LeanMass_Duration'] = df['LeanMass'] * df['Duration']
    df['BMR_Duration'] = df['BMR'] * df['Duration']
    df['BMR_per_min'] = df['BMR'] / df['Duration']
    df['Duration_sq'] = df['Duration'] ** 2
    df['Effort_BMI'] = (df['Duration'] * df['Weight']) / df['BMI']
    return df


def load_data():
    train = pd.read_csv('./train.csv', index_col=0)
    orig  = pd.read_csv('./calories_original.csv', index_col=0)
    orig.rename(columns={'Gender':'Sex'}, inplace=True)
    test  = pd.read_csv('./test.csv', index_col=0)

    full = pd.concat([train, orig], ignore_index=False)
    full = feature_engineering(full)
    test = feature_engineering(test)
    return full, test


class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)


def rmsle_loss(pred, true):
    return torch.sqrt(((torch.log1p(pred.clamp(min=0)) - torch.log1p(true))**2).mean())


def main():
    full_data, test_data = load_data()

    # Prepare features and target
    X = full_data.drop(columns=['Calories']).copy()
    y = full_data['Calories'].copy()
    for col in X.select_dtypes(['category']).columns:
        X[col] = X[col].cat.codes
        test_data[col] = test_data[col].cat.codes

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Tensors
    X_train_cpu = torch.tensor(X_train, dtype=torch.float32)
    y_train_cpu = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_cpu, y_train_cpu),
        batch_size=256,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True
    )

    # Model, optimizer, scheduler (warm restarts)
    model = MLPRegressor(X_train_cpu.shape[1]).to(device)
    initial_lr = 8e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    num_epochs = 100
    # Warm restarts every 25 epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)

    best_val = float('inf')
    epoch_bar = tqdm(range(1, num_epochs+1), desc='Epoch', unit='ep')
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False)
        for xb_cpu, yb_cpu in batch_bar:
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = rmsle_loss(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_bar.set_postfix(train_loss=f"{loss.item():.4f}", refresh=True)

        avg_train = running_loss / len(train_loader)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            avg_val = rmsle_loss(val_pred, y_val_tensor).item()

        # Best model tracking
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), 'best_mlp_model.pt')

        # Step scheduler with warm restarts
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]

        epoch_bar.set_postfix(
            train_rmsle=f"{avg_train:.4f}",
            val_rmsle=f"{avg_val:.4f}",
            best_val=f"{best_val:.4f}",
            lr=f"{current_lr:.2e}"
        )

    # Final save of last model
    torch.save(model.state_dict(), 'mlp_model_final.pt')
    print(f'Training complete. Best Val RMSLE: {best_val:.4f}. Models saved.')

    # Test inference using best model
    model.load_state_dict(torch.load('best_mlp_model.pt'))
    test_X = test_data.drop(columns=['Calories'], errors='ignore').copy()
    test_X_scaled = scaler.transform(test_X)
    test_tensor = torch.tensor(test_X_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        test_pred = model(test_tensor).cpu().numpy().flatten()
    test_pred = np.maximum(test_pred, 0)

    submission = pd.DataFrame({'Calories': test_pred}, index=test_data.index)
    submission.to_csv('mlp_test_predictions.csv')
    print('Test predictions saved to mlp_test_predictions.csv')

if __name__ == '__main__':
    main()