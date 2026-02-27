# Customer Churn Prediction & Agentic Retention Strategy

## From Predictive Analytics to Intelligent Intervention

---

## Project Overview

This project focuses on the design and implementation of an AI-driven customer analytics system for predicting telecom customer churn.

The system is structured in two milestones:

- **Milestone 1:** Classical machine learning techniques applied to historical customer behavior data to predict churn risk and identify key drivers of disengagement.
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about churn risk, retrieves retention best practices (RAG), and plans intervention strategies.

This repository currently implements **Milestone 1 only**.

---

## Milestone 1: ML-Based Churn Prediction

### Objective

Identify customers at risk of churn using structured behavioral and financial data through classical machine learning pipelines (without LLMs).

---

## Business Context

Customer churn directly affects revenue stability and customer lifetime value.

From a cost-sensitive perspective:

- Missing a churner (False Negative) is more costly than incorrectly flagging a loyal customer (False Positive).
- Therefore, the system prioritizes churn recall while maintaining strong overall model performance.

---

## System Architecture

Pipeline Overview:

1. Data Cleaning & Preprocessing  
2. Feature Engineering  
3. Train-Test Split (80/20, stratified)  
4. Model Training  
5. Threshold Optimization  
6. Model Evaluation  
7. Streamlit-Based Local UI  

*(Architecture Diagram Placeholder)*

---

## Data Processing

Key preprocessing steps:

- Removal of target leakage columns  
- Contextual handling of structural null values  
- Feature engineering (e.g., Revenue per Month, Monthly Contract flag)  
- One-hot encoding for categorical variables  
- Binary mapping for yes/no features  
- Feature scaling using StandardScaler  

Dataset size: 7,043 customers  
Train-Test split: 80% / 20%  
Random seed: 42  

---

## Models Implemented

### Logistic Regression

- Probability-based classifier  
- Threshold tuning applied (0.5 → 0.3)  
- Selected as final model  

### Decision Tree

- Depth tuning performed (2–20)  
- Class weighting applied  
- Structural regularization via max_depth  

---

## Model Evaluation

### Logistic Regression – Threshold Comparison

| Threshold | Accuracy | Churn Recall | Churn F1 |
|------------|------------|--------------|----------|
| 0.5 (Baseline) | 0.82 | 0.68 | 0.64 |
| 0.4 | 0.83 | 0.75 | 0.70 |
| **0.3 (Selected)** | **0.8077** | **0.83** | **0.70** |

Final Logistic Model (Threshold = 0.3):

- ROC-AUC: 0.8876  
- False Negatives: 63  
- Strong minority class detection  

---

### Decision Tree – Depth Tuning

Baseline (Depth = 2):

- Test Accuracy: 0.781  
- Churn Recall: 0.23  

Best Performing Depths:

| Depth | Test Accuracy | Churn Recall | Churn F1 |
|--------|---------------|--------------|----------|
| 8 | 0.829 | 0.64 | 0.67 |
| 13 | 0.829 | 0.63 | 0.66 |
| 14 | 0.833 | 0.61 | 0.66 |

Although Decision Tree achieved slightly higher raw accuracy at depth 14, it showed lower churn recall compared to Logistic Regression.

---

## Final Model Selection

**Selected Model: Logistic Regression (Threshold = 0.3)**  

Reasons:

- Highest ROC-AUC  
- Strongest churn recall (0.83)  
- Lowest false negatives  
- Better probability calibration  
- More interpretable  
- Better aligned with business objective  

---

## Local Application

The system includes a working Streamlit-based local interface that allows:

- Uploading customer data  
- Generating churn probability predictions  
- Displaying classification results  

*(Application Screenshot Placeholder)*

---

## Constraints & Requirements

- Team Size: 3–4 Students  
- API Budget: Free Tier Only  
- Framework (Milestone 2): LangGraph (Planned)  
- Hosting: Required for final submission  

---

## Technology Stack

| Component | Technology |
|------------|------------|
| ML Models | Logistic Regression, Decision Tree |
| Library | Scikit-Learn |
| UI | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Repository Structure
customer-churn/
│
├── app/
│ └── app.py
│
├── artifacts/
│ ├── best_churn_model.pkl
│ ├── feature_columns.pkl
│ └── scaler.pkl
│
├── data/
│ ├── raw/
│ │ └── telecom_customer_churn.csv
│ └── processed/
│ └── .gitkeep
│
├── reports/
│ └── preprocessing_report.md
│
├── src/
│ ├── eda.ipynb
│ └── model_pipeline.py
│
├── .gitignore
├── README.md
└── requirements.txt

---

## Constraints & Requirements

- Team Size: 3–4 Students
- API Budget: Free Tier Only
- Hosting required for final submission
- Milestone 2 will use LangGraph (planned)

---

## Technology Stack

| Component | Technology |
|------------|------------|
| ML Models | Logistic Regression, Decision Tree |
| ML Library | Scikit-Learn |
| UI | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Milestone 2 (Planned)

Future extension will include:

- LangGraph-based agent workflow
- Retrieval-Augmented Generation (RAG)
- Vector database (FAISS / Chroma)
- Public deployment

---


(Deployment Link Placeholder)
