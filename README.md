# Corporate Fraud Prediction Using Machine Learning

This project develops a predictive framework for detecting corporate fraud using structured financial data. The models implemented include **Random Forest** and **XGBoost**, with automatic feature selection and hyperparameter optimization. The framework leverages scikit-learn and XGBoost in Python and includes support for evaluation on validation and test datasets.


## Models Used

- **Random Forest (RF)**  
  Handles high-dimensional data well and manages imbalanced classes via `class_weight='balanced'`.

- **Extreme Gradient Boosting (XGBoost)**  
  Powerful gradient boosting technique that handles overfitting and feature interactions effectively.

## Key Features

- **Feature Standardization** using `StandardScaler`
- **Feature Selection** via `SelectFromModel` using XGBoost feature importance
- **Hyperparameter Tuning** with `RandomizedSearchCV` using AUC-ROC as the scoring metric
- **Prediction Output** to Excel files for easy review and reporting
- **Evaluation Metrics** including classification report and ROC AUC score

## Evaluation

The models are evaluated using:

- **Classification Report** (Precision, Recall, F1-score)
- **ROC AUC Score**

The final model predictions are saved in:

- `validation2_predictions.xlsx`
- `test2_predictions.xlsx`

## 🛠 Requirements

Install required Python packages using:

```bash
pip install -r requirements.txt
