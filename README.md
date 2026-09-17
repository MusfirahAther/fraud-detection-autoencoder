# Credit Card Fraud Detection

A Streamlit web application that identifies potentially fraudulent credit-card transactions with an autoencoder-based anomaly detection model.

The model learns normal transaction patterns and measures how accurately it can reconstruct a new transaction. A high reconstruction error means the transaction differs from the learned normal pattern and is flagged for review.

## Features

- Score individual transactions from `Time`, `Amount`, and anonymized `V1` to `V28` features
- Upload a CSV to score multiple transactions at once
- View reconstruction errors with clear **Fraud** or **Normal** results
- Uses saved model weights, feature scalers, and a precomputed threshold

## How it works

1. The application scales the `Time` and `Amount` values using saved scalers.
2. All 30 features are passed through the autoencoder.
3. It calculates the mean-squared reconstruction error.
4. Errors above the saved threshold are labelled **Fraud**; all others are **Normal**.

> A model flag is a signal for investigation, not a final financial decision.

## Project structure

```text
fraud-detection-autoencoder/
|- app.py                       # Streamlit application
|- requirements.txt             # Python dependencies
`- model/
   |- autoencoder.weights.h5    # Trained autoencoder weights
   |- scaler_amount.pkl         # Amount feature scaler
   |- scaler_time.pkl           # Time feature scaler
   `- threshold.txt             # Fraud decision threshold
```

## Run locally

Python 3.10 is recommended for the pinned TensorFlow version.

```powershell
git clone https://github.com/MusfirahAther/fraud-detection-autoencoder.git
cd fraud-detection-autoencoder
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown in the terminal, usually `http://localhost:8501`.

## CSV format

For batch scoring, upload a CSV with this schema:

```text
Time, V1, V2, ..., V28, Amount
```

Training data and test CSVs are intentionally excluded from this repository. Uploaded files are processed only for the current app session.

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io).
2. Create an app from this repository.
3. Select branch `master` and entrypoint `app.py`.
4. In **Advanced settings**, choose Python 3.10, then deploy.

## Tech stack

- [Streamlit](https://streamlit.io/)
- TensorFlow / Keras
- scikit-learn
- pandas and NumPy

## Repository hygiene

The `.gitignore` excludes virtual environments, secrets, generated Python files, operating-system files, and CSV datasets. The versioned model artifacts are required for inference and are small enough for standard GitHub hosting.
