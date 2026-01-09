import glob
import json
import os
from datetime import datetime
from pathlib import Path
from pprint import pformat

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pytorch_network import NeuralNetwork
from utils import setup_logging

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
sklearn.set_config(transform_output="pandas")


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

        # Showcase data after processing
        # self.log.info(f"Training data:\n {self.x_train.describe()}")
        # decoded_train = pd.Series(
        #     self.encoder.inverse_transform(np.ravel(self.y_train))
        # )
        # self.log.info(f"\n{decoded_train.value_counts(normalize=True) * 100}")
        #
        # self.log.info(f"Test data:\n {self.x_test.describe()}")
        # decoded_test = pd.Series(self.encoder.inverse_transform(np.ravel(self.y_test)))
        # self.log.info(f"\n{decoded_test.value_counts(normalize=True) * 100}")

    def get_train_test(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.x_train, self.y_train, self.x_test, self.y_test

    def train_model_1(self):
        x_train, y_train, x_test, y_test = self.get_train_test()
        custom_weights = {
            0: 1,
            9: 15,
            6: 2,
            3: 10,
            2: 5,
        }
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            class_weight=custom_weights,
            n_jobs=12,  # cpu num
            verbose=1,
        )  # Ensures your results are reproducible.
        self.log.info("Fitting model: RandomForestClassifier")
        rf_classifier.fit(x_train, np.ravel(y_train))
        self.log.info("Predicting results...")
        y_res = rf_classifier.predict(x_test)

        target_names = self.encoder.classes_
        report = classification_report(
            y_test,
            y_res,
            target_names=target_names,
            zero_division=0.0,
            output_dict=True,
        )

        self.log.info(f"Model Report:\n{pformat(report)}")
        with open(self.result_dir / "report_rfc.json", "w") as f:
            json.dump(report, f, indent=4)

        model_artifacts = {
            "classifier": rf_classifier,
            "scaler": self.scaler,
            "features": self.selector.get_support(),
            "label_encoder": self.encoder,
        }
        joblib.dump(model_artifacts, self.result_dir / "rfc_model.joblib")
        self.log.info("Model artifacts saved")

    def train_model_2(self):
        x_train, y_train, x_test, y_test = self.get_train_test()
        x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()

        dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            dataset, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info(f"Using device: {device}")
        model = NeuralNetwork(input_dim=40, output_dim=len(self.encoder.classes_)).to(
            device
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        epochs = 30
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.log.info(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}"
            )

        model.eval()
        y_pred_list = []

        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).to(device)
            outputs = model(x_test_tensor)
            _, y_pred_tags = torch.max(outputs, dim=1)
            y_pred_list = y_pred_tags.cpu().numpy()
            target_names = self.encoder.classes_
            report = classification_report(
                y_test,
                y_pred_list,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            )

            self.log.info(f"Model Report:\n{pformat(report)}")
            with open(self.result_dir / "report_nn.json", "w") as f:
                json.dump(report, f, indent=4)


if __name__ == "__main__":
    model = Model(datasets_path="./datasets")
    model.prepare_data()
    # model.train_model_1()
    model.train_model_2()
