"""
data_preprocessing.py

Handles loading, cleaning, and transforming the raw Telco Customer Churn
dataset into a format suitable for feature engineering and model training.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from pathlib import Path






BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "telecom_customer_churn.csv"

df = pd.read_csv(data_path)
df.head()
