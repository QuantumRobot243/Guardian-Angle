import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# Gradient Boosting Models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# Load Dataset
print("Loading dataset...")
try:
    data = pd.read_csv(r"women_night_safety_dataset.csv")
    print("Dataset loaded successfully.")
    print(f"Shape: {data.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display basic info
print("\nDataset info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nTarget distribution:")
print(data["Safe_Road"].value_counts(normalize=True))

# Check for missing values
print("\nMissing values per column:")
print(data.isna().sum())

# Preprocessing
print("\nStarting preprocessing...")

# Handle missing values if any
if data.isna().sum().any():
    print("Handling missing values...")
    # Numerical columns - fill with median
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [col for col in num_cols if col != "Safe_Road"]
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        data[num_cols] = num_imputer.fit_transform(data[num_cols])
    
    # Categorical columns - fill with most frequent
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# Encode categorical variables
categorical_cols = ["Location", "Lighting", "Time"]
label_encoders = {}
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

print("Preprocessing completed.")

# Separate features (X) and target (y)
X = data.drop("Safe_Road", axis=1)
y = data["Safe_Road"]

# Train-Test Split
print("\nSplitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Train-Test split completed.")

# Apply SMOTE to balance classes in training set
print("\nApplying SMOTE to balance class distribution...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    # Check if predict_proba exists
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None
    
    print(f"\n=== {model_name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    
    if y_proba is not None:
        try:
            print("ROC-AUC:", roc_auc_score(y_test, y_proba))
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {e}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Unsafe", "Safe"], 
                yticklabels=["Unsafe", "Safe"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Feature Importance (if available)
    if hasattr(model, "feature_importances_"):
        try:
            feature_importance = pd.DataFrame(
                {"Feature": X.columns, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_importance)
            plt.title(f"{model_name} Feature Importance")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot feature importance: {e}")

# Train XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_resampled, y_train_resampled)
evaluate_model(xgb_model, X_test, y_test, "XGBoost")
print("XGBoost training completed.")

# Train LightGBM
print("\nTraining LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    objective='binary',
    metric='binary_logloss'
)
lgb_model.fit(X_train_resampled, y_train_resampled)
evaluate_model(lgb_model, X_test, y_test, "LightGBM")
print("LightGBM training completed.")

# Train HistGradientBoostingClassifier
print("\nTraining HistGradientBoostingClassifier...")
hgb_model = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5,
    random_state=42,
    scoring='f1'
)
hgb_model.fit(X_train_resampled, y_train_resampled)
evaluate_model(hgb_model, X_test, y_test, "HistGradientBoostingClassifier")
print("HistGradientBoostingClassifier training completed.")

# Save Models and Encoders
print("\nSaving trained models and encoders...")
try:
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(lgb_model, "lgb_model.pkl")
    joblib.dump(hgb_model, "hgb_model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    print("Models and encoders saved successfully.")
except Exception as e:
    print(f"Error saving models: {e}")

print("\nScript completed successfully!")