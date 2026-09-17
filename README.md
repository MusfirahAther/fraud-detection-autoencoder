# Credit Card Fraud Detection

A Streamlit application that uses an autoencoder to flag potentially fraudulent credit-card transactions from reconstruction error.

## Run locally

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

For CSV scoring, upload a file containing `Time`, `V1` through `V28`, and `Amount` columns. Training data and uploaded test CSVs are intentionally not stored in this repository.

## Deployment

Deploy the repository through [Streamlit Community Cloud](https://share.streamlit.io) using `app.py` as the entrypoint.
