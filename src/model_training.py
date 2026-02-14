import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import preprocess
from data_encoding_scaling import (
    encode,
    split_data,
    scale_train,
    scale_test,
)


def train_model(raw_data_path: str):

    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv(raw_data_path)

    df = preprocess(df, training=True)
    df = encode(df, training=True)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled = scale_train(X_train)
    X_test_scaled = scale_test(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "artifacts/model.pkl")

    return model


if __name__ == "__main__":
    train_model("data/raw/telecom_customer_churn.csv")
