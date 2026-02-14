import pandas as pd

COLS_TO_DROP = [
    "Customer ID",
    "Customer Status",
    "Churn Category",
    "Churn Reason",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
]

INTERNET_DEPENDENT_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
]


def impute_numerical(df: pd.DataFrame) -> pd.DataFrame:
    df["Avg Monthly Long Distance Charges"] = df[
        "Avg Monthly Long Distance Charges"
    ].fillna(0)

    df["Avg Monthly GB Download"] = df["Avg Monthly GB Download"].fillna(0)

    return df


def impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df["Offer"] = df["Offer"].fillna("No Offer")
    df["Multiple Lines"] = df["Multiple Lines"].fillna("No Phone")
    df["Internet Type"] = df["Internet Type"].fillna("No Internet Service")

    for col in INTERNET_DEPENDENT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("No Internet Service")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["Tenure in Months"] = df["Tenure in Months"].replace(0, 1)

    df["Revenue_Per_Month"] = df["Total Revenue"] / df["Tenure in Months"]

    df["Is_Monthly_Contract"] = (
        df["Contract"] == "Month-to-Month"
    ).astype(int)

    return df


def preprocess(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:

    df = df.copy()

    if training:
        df["Churn"] = df["Customer Status"].apply(lambda x: 1 if x == "Churned" else 0)

    df = df.drop(columns=COLS_TO_DROP, errors="ignore")

    df = impute_numerical(df)
    df = impute_categorical(df)
    df = feature_engineering(df)

    return df





