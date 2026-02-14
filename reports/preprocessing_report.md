# Data Preprocessing Report: Customer Churn Prediction

## 1. Dataset Overview

The Telco Customer Churn dataset comprises **7,043 records** across **38 features** spanning customer demographics, service subscriptions, billing information, and churn status. The preprocessing pipeline transforms this raw data into a clean, model-ready format of **31 features** with zero missing values.

---

## 2. Target Variable Engineering

- **Derived** a binary target `Churn` from the multi-class `Customer Status` column (`Churned` → 1, else → 0).
- **Rationale:** Converts the problem into a standard binary classification task, enabling the use of metrics such as precision, recall, F1-score, and AUC-ROC.

---

## 3. Column Removal

The following columns were dropped, grouped by rationale:

| Dropped Columns                             | Reason                                                                                                                                                                                                                       |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Customer ID`                               | Unique identifier; no predictive value. Including it risks overfitting to individual records.                                                                                                                                |
| `Customer Status`                           | Redundant after deriving the binary `Churn` target. Retaining it would constitute **target leakage**.                                                                                                                        |
| `Churn Category`, `Churn Reason`            | Post-hoc labels available only for churned customers. These are **direct leakage variables**—they encode the outcome itself.                                                                                                 |
| `City`, `Zip Code`, `Latitude`, `Longitude` | High-cardinality geographic identifiers. `City` alone has ~1,100 unique values, which inflates dimensionality without proportional information gain; geospatial coordinates add noise without engineered proximity features. |

---

## 4. Numerical Preprocessing

### 4.1 Missing Value Imputation

| Feature                             | Missing Count | Strategy        | Justification                                                                                                                                                          |
| ----------------------------------- | ------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Avg Monthly Long Distance Charges` | 682           | Fill with **0** | Nulls correspond to customers with `Phone Service = "No"`. No phone service implies zero long-distance charges; this is a structurally absent value, not a random gap. |
| `Avg Monthly GB Download`           | 1,526         | Fill with **0** | Nulls correspond to `Internet Service = "No"`. Same structural reasoning—no internet means zero download volume.                                                       |

### 4.2 Retained Numerical Features

`Age`, `Number of Dependents`, `Number of Referrals`, `Tenure in Months`, `Monthly Charge`, `Total Charges`, `Total Refunds`, `Total Extra Data Charges`, `Total Long Distance Charges`, `Total Revenue` — all had **zero missing values** and required no imputation.

---

## 5. Categorical Preprocessing

### 5.1 Missing Value Imputation

All categorical nulls are **structurally missing** (conditional on the absence of a parent service), not randomly absent. Imputation therefore uses domain-meaningful sentinel labels rather than statistical methods (mode, KNN).

| Feature(s)                                                                                                                                                    | Missing    | Strategy                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------- |
| `Offer`                                                                                                                                                       | 3,877      | Fill `"No Offer"` — customer was not enrolled in any promotional offer.    |
| `Multiple Lines`                                                                                                                                              | 682        | Fill `"No Phone"` — absence aligns with `Phone Service = "No"`.            |
| `Internet Type`                                                                                                                                               | 1,526      | Fill `"No Internet Service"` — aligns with `Internet Service = "No"`.      |
| `Online Security`, `Online Backup`, `Device Protection Plan`, `Premium Tech Support`, `Streaming TV`, `Streaming Movies`, `Streaming Music`, `Unlimited Data` | 1,526 each | Fill `"No Internet Service"` — all are internet-dependent add-on services. |

### 5.2 Retained Categorical Features

`Gender`, `Married`, `Phone Service`, `Internet Service`, `Contract`, `Paperless Billing`, `Payment Method` — fully populated; no imputation required.

---

## 6. Outlier Analysis

An **IQR-based outlier detection** method (1.5 × IQR rule) was applied to key numerical features:

| Feature                             | Outliers Detected | Action                                                                                                                                                                                                                                                                                    |
| ----------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Avg Monthly Long Distance Charges` | 0                 | No treatment needed.                                                                                                                                                                                                                                                                      |
| `Avg Monthly GB Download`           | 649 (~9.2%)       | **Retained.** These represent legitimate high-usage customers (e.g., heavy streamers). Removing them would discard valid behavioural signal correlated with churn risk. The distribution was further examined via histogram with KDE overlay to confirm the absence of data-entry errors. |

**Design Decision:** Outliers were not capped or removed. Tree-based models (the primary model family for this project) are inherently robust to outliers, and aggressive truncation can suppress genuine high/low-value customer segments relevant to churn prediction.

---

## 7. Data Leakage Prevention

| Measure                        | Implementation                                                                                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Target leakage**             | Dropped `Customer Status`, `Churn Category`, and `Churn Reason` before any modelling. These columns directly encode or causally derive from the churn event. |
| **Identifier exclusion**       | Removed `Customer ID` to prevent memorisation of individual records.                                                                                         |
| **No future-looking features** | All retained features (tenure, monthly charges, service flags) represent information available _at prediction time_, not post-churn outcomes.                |

---

## 8. Final Preprocessed Dataset

| Property                | Value                           |
| ----------------------- | ------------------------------- |
| Records                 | 7,043                           |
| Features (excl. target) | 30                              |
| Numerical features      | 13                              |
| Categorical features    | 17                              |
| Missing values          | 0                               |
| Target distribution     | ~73.5% Non-Churn / ~26.5% Churn |
