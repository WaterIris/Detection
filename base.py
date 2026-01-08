import glob
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import setup_logging

pd.set_option("display.max_rows", 10)
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
        output_dir = Path(f"./results/run_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

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

    def prepare_data(self) -> None:
        self.log.info("Data preparation...")
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
        # x - features
        x = self.full_dataset.iloc[:, :-1]
        # y - labels
        y = self.full_dataset.iloc[:, -1:]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        self.log.info("Data split")

        scaler = StandardScaler()
        scaler.fit(x_train)
        joblib.dump(scaler, "scaler.pkl")
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        self.log.info("Data scaled")

        # # Select k features with highest variance/f score
        selector = SelectKBest(score_func=f_classif, k=5)
        # selector = SelectKBest(score_func=mutual_info_classif, k=5) #  Unusable due to time required
        selector.fit(x_train_scaled, np.ravel(y_train))
        # For some reason lsp thinks that scikit return it as either list of ndarray altough its pandas dataframe
        x_train_selected = pd.DataFrame(selector.transform(x_train_scaled))
        x_test_selected = pd.DataFrame(selector.transform(x_test_scaled))
        self.log.info("Data selected")
        self.log.info(f"Train:\n{x_train_selected.describe()}")
        self.log.info(f"Test:\n{x_test_selected.describe()}")


if __name__ == "__main__":
    model = Model(datasets_path="./datasets")
    model.prepare_data()
