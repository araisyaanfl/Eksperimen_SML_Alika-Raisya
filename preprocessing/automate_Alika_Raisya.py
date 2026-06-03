import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from joblib import dump
import os


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    
    target = "Loan_Approved"

    X = df.drop(columns=[target])
    y = df[target]

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def save_data(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    os.makedirs("dataset_preprocessing", exist_ok=True)

    pd.DataFrame(X_train).to_csv(
        "dataset_preprocessing/X_train.csv",
        index=False
    )

    pd.DataFrame(X_test).to_csv(
        "dataset_preprocessing/X_test.csv",
        index=False
    )

    pd.DataFrame(y_train).to_csv(
        "dataset_preprocessing/y_train.csv",
        index=False
    )

    pd.DataFrame(y_test).to_csv(
        "dataset_preprocessing/y_test.csv",
        index=False
    )

    dump(
        preprocessor,
        "dataset_preprocessing/preprocessing.pkl"
    )


def main():

    df = load_data("../dataset_raw/loan_prediction_dataset.csv")

    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    save_data(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    main()