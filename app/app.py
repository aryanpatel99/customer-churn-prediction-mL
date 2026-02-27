"""
app.py

Streamlit entry point for the Customer Churn Prediction web application.
Provides an interactive UI for uploading data, running predictions, and
visualising model results.
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# # ********************
  # Page Configuration
# # ********************

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide"
)



# # ********************
#     # Load Model
# # ********************
# @st.cache_resource
# def load_model():
#     try:
#         with open("models/trained_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         return model
#     except Exception as e:
#         return None

# model = load_model()

st.title("Customer Churn Intelligence Dashboard", anchor=False)
st.caption("Predict and understand customer churn with our interactive dashboard.")


st.subheader("Customer Input", anchor=False)

tab_manual,tab_csv = st.tabs([
    "👤 Manual Multi-Customer",
    "📎 Upload CSV"
])



with tab_csv:

    st.markdown("### Batch Prediction via CSV")

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)

        with st.expander("View Current Customer Data", expanded=True):
            st.write("Preview:", df_uploaded.head())

        if st.button("Predict Batch", type="primary", use_container_width=False):
            st.success(f"Loaded {len(df_uploaded)} customers (connect model)")


with tab_manual:

    st.markdown("### Synthetic Customer Testing")
    st.caption("Generate random customers to stress-test the churn model.")



    input_col, btn_col, _ = st.columns([2, 1, 2])

    with input_col:
        n_rows = st.number_input(
            "Number of customers to generate",
            min_value=1,
            max_value=10000,
            value=5,
            step=1
        )

    with btn_col:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

        generate_btn = st.button(
            "Generate & Predict",
            use_container_width=True
        )


    # ==============================
    #  RANDOM DATA FUNCTION
    # ==============================
    def generate_random_customers(n):

        rng = np.random.default_rng()

        df = pd.DataFrame({
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 80, n),
            "Married": rng.choice(["Yes", "No"], n),
            "Dependents": rng.integers(0, 5, n),
            "Referrals": rng.integers(0, 10, n),
            "Tenure (Months)": rng.integers(0, 72, n),
            "Offer": rng.choice(["None", "Offer A", "Offer B", "Offer C"], n),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], n
            ),
            "Paperless Billing": rng.choice(["Yes", "No"], n),
            "Payment Method": rng.choice(
                ["Credit Card", "Bank Withdrawal", "Mailed Check"], n
            ),
            "Phone Service": rng.choice(["Yes", "No"], n),
            "Multiple Lines": rng.choice(["Yes", "No"], n),
            "Internet Service": rng.choice(
                ["DSL", "Fiber optic", "No"], n
            ),
            "Internet Type": rng.choice(["Cable", "DSL", "Fiber"], n),
            "Online Security": rng.choice(["Yes", "No"], n),
            "Online Backup": rng.choice(["Yes", "No"], n),
            "Device Protection": rng.choice(["Yes", "No"], n),
            "Tech Support": rng.choice(["Yes", "No"], n),
            "Streaming TV": rng.choice(["Yes", "No"], n),
            "Streaming Movies": rng.choice(["Yes", "No"], n),
            "Streaming Music": rng.choice(["Yes", "No"], n),
            "Unlimited Data": rng.choice(["Yes", "No"], n),
            "Avg Long Distance": rng.uniform(0, 50, n),
            "Avg GB Download": rng.uniform(0, 500, n),
            "Monthly Charge": rng.uniform(20, 120, n),
            "Total Charges": rng.uniform(100, 8000, n),
            "Refunds": rng.uniform(0, 50, n),
            "Extra Data Charges": rng.uniform(0, 100, n),
            "Total Long Distance": rng.uniform(0, 200, n),
            "Total Revenue": rng.uniform(100, 10000, n),
        })


        return df


    if generate_btn:

        with st.spinner("Generating synthetic customers..."):
            df_random = generate_random_customers(n_rows)

        st.success(f"Generated {len(df_random)} customers")

        with st.expander("View Current Customer Data", expanded=True):

            st.dataframe(df_random.head(50), use_container_width=True)
       
