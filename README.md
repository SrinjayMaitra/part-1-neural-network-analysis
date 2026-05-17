# Part 1 — Neural Network Fundamentals and Training Behavior Analysis

This repository implements a feed-forward neural network for customer churn prediction, with hyperparameter experimentation and a written reflection on training dynamics.

## Problem

**Binary classification** — predict whether a telecom customer will churn (`churn = 1`) or be retained (`churn = 0`) based on 15 demographic, behavioural, and transactional features.

## Dataset

- **Source:** Provided in the shared Google Drive folder — https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing (Part 1 dataset: `customer_churn_nn.csv`)
- File: `customer_churn_nn.csv` (place in the same directory as `notebook.ipynb`, or upload to Colab's working directory)
- 2,000 rows × 17 columns (1 ID + 15 features + 1 target)
- **Severe class imbalance: only 1.55% of customers churned** (31 out of 2,000)
- Categorical features: `region`, `plan_type`, `contract_type`, `payment_method`
- Numerical features: `tenure_months`, `monthly_charges_inr`, `avg_login_days_per_month`, `support_tickets_last_90_days`, `payment_delay_days`, `data_usage_gb`, `satisfaction_score`, `last_complaint_days_ago`, `discount_percent`, `autopay_enabled`, `referral_count`

## Repository Structure

```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.png
    ├── model_comparison_table.csv
    └── evaluation_outputs.png
```

## Tasks Covered

1. **Dataset Understanding** — shape, dtypes, missing-value check, summary statistics, target distribution.
2. **Preprocessing** — drop ID column, one-hot encode categoricals, scale features, stratified 80/20 split, compute class weights for imbalance.
3. **Model Building** — feed-forward NN with one hidden layer (ReLU), sigmoid output, binary cross-entropy loss, Adam optimizer.
4. **Training & Evaluation** — train/test loss and accuracy curves, confusion matrix, classification report, ROC-AUC.
5. **Hyperparameter Experimentation** — 6 configurations varying hidden layers, neurons, learning rate, batch size, and activation function. Results saved to `results/model_comparison_table.csv` and `.png`.
6. **Final Reflection** — written discussion of weights/biases, activation functions, learning rate behaviour, and overfitting/underfitting diagnosis.

## How to Run

### Colab

1. Open `notebook.ipynb` in Google Colab.
2. Upload `customer_churn_nn.csv` to the Colab session (folder icon → upload).
3. Runtime → Run all (≈ 1–2 minutes total).

### Local

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Key Design Notes

- **Stratified train/test split** is essential because churn cases are rare — without stratification the test set could contain zero or near-zero churners.
- **Class weighting** (`compute_class_weight('balanced', ...)`) prevents the network from trivially predicting "no churn" for every customer.
- **Accuracy is not the right metric.** A model predicting `0` for everyone achieves 98.45% accuracy. The notebook reports per-class **precision, recall, F1**, and **ROC-AUC** for the churn class, which are the metrics that actually reflect business value.

## Dependencies

See `requirements.txt`.
