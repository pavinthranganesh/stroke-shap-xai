
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler


class ImprovedDataLoader:
    def __init__(self):
        self.data = None
        self.imputer = SimpleImputer(strategy="median")

    def load_dataset(
        self,
        path="data/healthcare-dataset-stroke-data.csv"
    ):
        self.data = pd.read_csv(path)

    def get_data_split(self):
        # Remove identifier and separate target
        X = self.data.drop(columns=["id", "stroke"])
        y = self.data["stroke"]

        # Stratified split before fitting preprocessing
        return train_test_split(
            X, y,
            test_size=0.20,
            random_state=2021,
            stratify=y
        )

    def preprocess_split(self, X_train, X_test):
        X_train = X_train.copy()
        X_test = X_test.copy()

        # Fit BMI imputation on training data only
        X_train["bmi"] = self.imputer.fit_transform(
            X_train[["bmi"]]
        ).ravel()

        X_test["bmi"] = self.imputer.transform(
            X_test[["bmi"]]
        ).ravel()

        categorical_cols = [
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status"
        ]

        # Fit category vocabulary on training data only
        X_train = pd.get_dummies(
            X_train,
            columns=categorical_cols,
            dtype=int
        )

        X_test = pd.get_dummies(
            X_test,
            columns=categorical_cols,
            dtype=int
        )

        # Match test columns to training columns
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

        return X_train, X_test

    def oversample(self, X_train, y_train):
        sampler = RandomOverSampler(
            sampling_strategy="minority",
            random_state=42
        )

        return sampler.fit_resample(
            X_train, y_train
        )
