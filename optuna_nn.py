import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from utils import setup_logging


class Model:
    def __init__(self, datasets_path: str) -> None:
        self.log = setup_logging()
        self.full_dataset = self.load_datasets(datasets_path)
        self.create_result_dir()

    def create_result_dir(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.result_dir = Path(f"./results/run_{timestamp}")
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def load_datasets(self, dataset_path) -> pd.DataFrame:
        search_path = os.path.join(dataset_path, "*.csv")
        csv_files = glob.glob(search_path)

        df_list = []
        for filename in csv_files:
            data = pd.read_csv(filename)
            df_list.append(data)
            self.log.info(f"Successfully loaded: {os.path.basename(filename)}")

        full_df = pd.concat(df_list, axis=0, ignore_index=True)
        return full_df

    @staticmethod
    def to_df(df) -> pd.DataFrame:
        # helper method to fix lsp mistaking scikit output as ndarray
        return pd.DataFrame(df)

    def prepare_data(self) -> None:
        self.log.info("Data processing...")
        # Label Encoding
        self.encoder = LabelEncoder()
        self.encoder.fit(self.full_dataset.iloc[:, -1])
        self.full_dataset[" Label"] = self.encoder.transform(
            self.full_dataset[" Label"]
        )
        # Remove +- infinities, nan and columns with zero variance.
        self.full_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.full_dataset.dropna(inplace=True)
        self.full_dataset = self.full_dataset.loc[:, self.full_dataset.nunique() > 1]
        self.log.info("Data cleaned")

        # Split dataset into features and labels
        x = self.full_dataset.iloc[:, :-1]  # - features
        y = self.full_dataset.iloc[:, -1:]  # - labels

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y
        )
        self.log.info("Data split")
        df_train = pd.concat([self.to_df(x_train), self.to_df(y_train)], axis=1)
        df_test = pd.concat([self.to_df(x_test), self.to_df(y_test)], axis=1)

        # Keep dataset split as backup for retesting/training
        df_train.to_csv(path_or_buf=self.result_dir / "train.csv")
        df_test.to_csv(path_or_buf=self.result_dir / "test.csv")
        del df_train  # free memory
        del df_test  # free memory

        self.scaler = StandardScaler()
        self.scaler.fit(x_train)
        x_train_scaled = self.scaler.transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        self.log.info("Data scaled")
        del x_train  # free memory
        del x_test  # free memory

        # Select k features with highest variance/f score
        self.selector = SelectKBest(score_func=f_classif, k=40)
        self.selector.fit(x_train_scaled, np.ravel(y_train))
        # For some reason lsp thinks that scikit return it as either list of ndarray altough its pandas dataframe
        x_train_selected = self.selector.transform(x_train_scaled)
        x_test_selected = self.selector.transform(x_test_scaled)
        self.log.info("Data selected")
        del x_train_scaled  # free memory
        del x_test_scaled  # free memory

        self.x_train: pd.DataFrame = self.to_df(x_train_selected)
        self.y_train: pd.DataFrame = self.to_df(y_train)
        self.x_test: pd.DataFrame = self.to_df(x_test_selected)
        self.y_test: pd.DataFrame = self.to_df(y_test)
        self.log.info("Data Processed")

    def get_train_test(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.x_train, self.y_train, self.x_test, self.y_test


def objective(trial):
    manager = Model(datasets_path="./datasets")
    manager.prepare_data()
    x_train, y_train, x_test, y_test = manager.get_train_test()
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()

    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        dataset, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True
    )

    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(
        test_dataset, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_layers = trial.suggest_int("n_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)

    manager = Model(datasets_path="./datasets")
    manager.prepare_data()
    x_train, y_train, x_test, y_test = manager.get_train_test()
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()

    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    layers = []
    in_features = 40
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 32, 256)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_features = out_features

    layers.append(nn.Linear(in_features, 15))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for val_X, val_y in test_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            pred = model(val_X).argmax(dim=1, keepdim=True)
            correct += pred.eq(val_y.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return accuracy


study = optuna.create_study(
    study_name="ids_pytorch_optimization",
    storage="sqlite:///db.sqlite3",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial

print(f"  Value (Accuracy): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
