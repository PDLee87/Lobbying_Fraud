import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

# Set input directory
input_dir = "specify the input directory"  # Update this path

# Load data
print("Loading data...")
train_df = pd.read_excel(os.path.join(input_dir, "place the training Excel dataset.xlsx"))
val_df = pd.read_excel(os.path.join(input_dir, "place the validation Excel dataset.xlsx"))
test_df = pd.read_excel(os.path.join(input_dir, "place the testing Excel dataset.xlsx"))

# Extract fraud label (column A) and features (columns B onward)
y_train = train_df.iloc[:, 0]
X_train = train_df.iloc[:, 1:]
y_val = val_df.iloc[:, 0]
X_val = val_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]
X_test = test_df.iloc[:, 1:]

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Feature Selection using XGBoost
print("Selecting important features...")
xgb_selector = XGBClassifier(n_estimators=100, random_state=42)
xgb_selector.fit(X_train, y_train)
selector = SelectFromModel(xgb_selector, threshold="mean", prefit=True)

X_train = selector.transform(X_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)

print(f"Selected {X_train.shape[1]} important features.")

# Hyperparameter tuning for Random Forest
print("Tuning Random Forest model...")
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_leaf": [10, 20, 30]
}
rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_tuned = RandomizedSearchCV(rf_model, rf_params, n_iter=5, cv=3, scoring="roc_auc", verbose=1)
rf_tuned.fit(X_train, y_train)

# Hyperparameter tuning for XGBoost
print("Tuning XGBoost model...")
xgb_params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [10, 20, 30]
}
xgb_model = XGBClassifier(random_state=42)
xgb_tuned = RandomizedSearchCV(xgb_model, xgb_params, n_iter=5, cv=3, scoring="roc_auc", verbose=1)
xgb_tuned.fit(X_train, y_train)

# Best models
rf_best = rf_tuned.best_estimator_
xgb_best = xgb_tuned.best_estimator_

# Generate validation predictions
print("Generating validation predictions...")
y_pred_prob_rf = rf_best.predict_proba(X_val)[:, 1]
y_pred_prob_xgb = xgb_best.predict_proba(X_val)[:, 1]

# Save validation predictions
val_results = pd.DataFrame({
    "Actual": y_val,
    "RF_Score": y_pred_prob_rf,
    "XGBoost_Score": y_pred_prob_xgb
})
val_results.to_excel(os.path.join(input_dir, "validation_predictions.xlsx"), index=False)
print("Validation predictions saved!")

# Generate test predictions
print("Generating final test predictions...")
y_test_prob_rf = rf_best.predict_proba(X_test)[:, 1]
y_test_prob_xgb = xgb_best.predict_proba(X_test)[:, 1]

# Save test predictions for both models
test_results = pd.DataFrame({
    "Actual": y_test,
    "RF_Score": y_test_prob_rf,
    "XGBoost_Score": y_test_prob_xgb
})
test_results.to_excel(os.path.join(input_dir, "test_predictions.xlsx"), index=False)
print("Test predictions saved for all models!")

# Print evaluation metrics
print("\nRandom Forest Validation Report:")
print(classification_report(y_val, (y_pred_prob_rf > 0.5).astype(int)))
print("Random Forest AUC-ROC:", roc_auc_score(y_val, y_pred_prob_rf))

print("\nXGBoost Validation Report:")
print(classification_report(y_val, (y_pred_prob_xgb > 0.5).astype(int)))
print("XGBoost AUC-ROC:", roc_auc_score(y_val, y_pred_prob_xgb))

print("\nTest Set Performance:")
y_test_pred = (y_test_prob_xgb > 0.5).astype(int)  # Use XGBoost for binary classification
print(classification_report(y_test, y_test_pred))
print("Test AUC-ROC:", roc_auc_score(y_test, y_test_prob_xgb))
