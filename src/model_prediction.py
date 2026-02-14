import pandas as pd
import joblib

from data_preprocessing import preprocess
from data_encoding_scaling import encode


def predict_from_dataframe(df: pd.DataFrame):

    df = preprocess(df, training=False)
    df = encode(df, training=False)

    scaler = joblib.load("artifacts/scaler.pkl")
    model = joblib.load("artifacts/model.pkl")

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)

    return prediction, probability


def predict_from_dict(user_input: dict):

    df = pd.DataFrame([user_input])
    return predict_from_dataframe(df)


def predict_from_csv(csv_path: str):

    df = pd.read_csv(csv_path)
    return predict_from_dataframe(df)


if __name__ == "__main__":

    prediction, probability = predict_from_csv(
        "data/raw/telecom_customer_churn.csv"
    )

    print("Predictions:", prediction[:5])
    print("Probabilities:", probability[:5])
