# Credit Card Fraud Detection using Deep Autoencoders

An unsupervised deep learning project that detects fraudulent credit card transactions by learning what a **normal** transaction looks like — and flagging anything it can't reconstruct well as suspicious.

**Live App:** [https://fraud-detection-autoencoder-42gb6653m29lhvbvs4mtvw.streamlit.app/](https://fraud-detection-autoencoder-42gb6653m29lhvbvs4mtvw.streamlit.app/)

---

## Problem Statement

Credit card fraud is extremely rare compared to normal transactions — in this dataset, fraud makes up only **0.17%** of all transactions (492 out of 284,807). This extreme imbalance makes standard classification models struggle: a model could achieve 99.8% accuracy by simply predicting "not fraud" every single time, while catching zero actual fraud cases. Fraud patterns also constantly change, making it hard to rely only on past labeled fraud examples.

## Why an Autoencoder Instead of Classification

Rather than training a model to classify "fraud vs normal" directly, this project uses an **Autoencoder** — trained **only on normal transactions**, with no fraud examples shown during training.

The idea: an Autoencoder learns to compress a transaction down to its essential pattern, then rebuild it. Since it only ever practiced on normal transactions, it becomes very good at rebuilding normal patterns — but struggles to rebuild something it's never seen before, like fraud. That struggle shows up as a higher **reconstruction error**, which becomes the fraud signal.

This approach is genuinely useful in the real world because:
- It doesn't need large amounts of labeled fraud data (which is rare and expensive to collect)
- It can catch new, unseen fraud patterns, not just fraud that looks like past examples

## Dataset

**Credit Card Fraud Detection** dataset (Kaggle: [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))

- 284,807 real transactions from European cardholders, over 2 days
- 492 fraud cases (0.17%)
- Features: `V1`-`V28` (anonymized via PCA to protect sensitive data), `Time` (seconds since first transaction), `Amount` (transaction amount), `Class` (0 = normal, 1 = fraud)

- **Encoder**: compresses 30 input features down to a 10-value bottleneck
- **Decoder**: rebuilds the original 30 features from that compressed form
- Trained only on normal transactions, using Mean Squared Error (MSE) as the loss function
- `Amount` and `Time` were scaled using `StandardScaler` before training (all other features were already PCA-normalized)

## Results

| Metric | Normal | Fraud |
|---|---|---|
| Precision | 1.00 | 0.72 |
| Recall | 1.00 | 0.77 |
| F1-score | 1.00 | 0.74 |

**Confusion Matrix**

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 56,714 | 149 |
| **Actual Fraud** | 113 | 379 |

Out of 492 real fraud cases, the model correctly caught **379 (77%)**, with a low false alarm rate of **0.26%** on normal transactions.

## Live Demo

Try it here: **[fraud-detection-autoencoder.streamlit.app](https://fraud-detection-autoencoder-42gb6653m29lhvbvs4mtvw.streamlit.app/)**

### Test Cases

**Should show NORMAL:**
- Amount: `50.00`, Time: `40000`
- V values: `-1.2, 0.3, 1.1, 0.4, -0.3, 0.5, 0.2, 0.1, -0.4, 0.2, -0.6, 0.3, -0.2, -0.5, 0.1, 0.2, -0.1, 0.05, 0.3, -0.1, 0.02, 0.1, -0.05, 0.1, 0.15, -0.1, 0.03, 0.01`

**Should show FLAGGED AS FRAUD:**
- Amount: `9999.00`, Time: `10`
- V values: `-15.5, 12.3, -18.7, 10.2, -20.1, 15.6, -12.4, 18.9, -14.2, 16.7, -19.3, 13.1, -17.8, 11.4, -16.2, 14.9, -13.7, 19.5, -11.8, 17.2, -15.9, 12.6, -18.4, 16.1, -14.7, 13.9, -17.1, 15.3`

You can also use the **Upload CSV** tab to score multiple transactions at once (CSV must contain `Time`, `V1`-`V28`, `Amount` columns).

## Running Locally

```bash
# Clone the repository
git clone https://github.com/musfirahather/fraud-detection-autoencoder.git
cd fraud-detection-autoencoder

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.


## Limitations & Future Improvements

- The model was trained on a static, historical dataset — a real production system would need periodic retraining as fraud patterns evolve
- Could be benchmarked against simpler methods like **Isolation Forest** to confirm whether the added complexity of deep learning is justified for this data
- The manual entry mode requires pasting raw `V1`-`V28` values, which isn't realistic for an end user in a real product — a production system would compute these internally from real transaction data
- Real fraud detection systems typically combine multiple models/signals rather than relying on a single detector

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, Pandas, NumPy, Streamlit
