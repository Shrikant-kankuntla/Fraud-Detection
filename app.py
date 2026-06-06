import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="centered"
)

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("Fraud_detection_pipeline.pkl")

model = load_model()

# Title and description
st.title("Fraud Detection System")
st.write("Enter the transaction details below to check if it is fraudulent or normal.")
st.divider()

# Input form
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input(
        "Step (hour of simulation)",
        min_value=1,
        max_value=744,
        value=1,
        help="Each step represents 1 hour. Max is 744 (30 days)."
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        help="Type of transaction being made."
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help="Amount of money being transferred."
    )

    old_balance_orig = st.number_input(
        "Sender Balance Before Transaction",
        min_value=0.0,
        value=5000.0,
        step=100.0,
        help="Balance of the sender account before this transaction."
    )

with col2:
    new_balance_orig = st.number_input(
        "Sender Balance After Transaction",
        min_value=0.0,
        value=4000.0,
        step=100.0,
        help="Balance of the sender account after this transaction."
    )

    old_balance_dest = st.number_input(
        "Receiver Balance Before Transaction",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help="Balance of the receiver account before this transaction."
    )

    new_balance_dest = st.number_input(
        "Receiver Balance After Transaction",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help="Balance of the receiver account after this transaction."
    )

st.divider()

# Predict button
if st.button("Check Transaction", use_container_width=True, type="primary"):

    # Build the same features used during training
    balance_diff_orig = old_balance_orig - new_balance_orig
    balance_diff_dest = new_balance_dest - old_balance_dest

    input_data = pd.DataFrame({
        "step":            [step],
        "type":            [transaction_type],
        "amount":          [amount],
        "oldbalanceOrg":   [old_balance_orig],
        "newbalanceOrig":  [new_balance_orig],
        "oldbalanceDest":  [old_balance_dest],
        "newbalanceDest":  [new_balance_dest],
        "balanceDiffOrig": [balance_diff_orig],
        "balanceDiffDest": [balance_diff_dest]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Show result
    st.subheader("Result")

    if prediction == 1:
        st.error("Fraudulent Transaction Detected")
        st.write(
            "This transaction has patterns that match known fraud cases. "
            "It should be flagged for review."
        )
    else:
        st.success("Transaction Looks Normal")
        st.write(
            "This transaction does not show signs of fraud based on the model."
        )

    # Show fraud probability as a progress bar
    st.write("**Fraud Probability:**", f"{probability:.2%}")
    st.progress(float(probability))

  
    with st.expander("See input summary"):
        st.write("**Transaction Type:**", transaction_type)
        st.write("**Amount:**", f"{amount:,.2f}")
        st.write("**Sender Balance Change:**", f"{balance_diff_orig:,.2f}")
        st.write("**Receiver Balance Change:**", f"{balance_diff_dest:,.2f}")
        st.write("**Fraud Probability:**", f"{probability:.4f}")

# Footer
st.divider()
st.caption("Model: Random Forest trained on the Fraud Detection dataset from Kaggle.")