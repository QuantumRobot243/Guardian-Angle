import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
) 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import shap
from scipy.stats import uniform, randint
from datetime import datetime

# Load Dataset with enhanced error handling
print("Loading dataset...")
try:
    data = pd.read_csv("women_night_safety_dataset.csv")
    print("Dataset loaded successfully.")
    print(f"Shape: {data.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Enhanced data exploration
def explore_data(df):
    print("\n=== Data Exploration ===")
    print("\nBasic info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nTarget distribution:")
    print(df["Safe_Road"].value_counts(normalize=True))
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nDescriptive statistics:")
    print(df.describe())

explore_data(data)

# Feature Engineering
print("\nStarting feature engineering...")
def extract_time_features(df):
    # Convert time to datetime and extract features
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    df['Is_Late_Night'] = df['Hour'].apply(lambda x: 1 if x >= 23 or x <= 4 else 0)
    df['Is_Evening'] = df['Hour'].apply(lambda x: 1 if 18 <= x <= 22 else 0)
    return df

data = extract_time_features(data)

# Preprocessing
print("\nStarting preprocessing...")

# First encode categorical variables before creating interaction terms
categorical_cols = ["Location", "Lighting"]
label_encoders = {}
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Now create interaction features with encoded values
data['Lighting_Police'] = data['Lighting'] * data['Police_Presence']
data['Crime_Population'] = data['Crime_Rate'] * data['Population_Density']
print("Feature engineering completed.")

# Handle missing values if any
if data.isna().sum().any():
    print("Handling missing values...")
    # Numerical columns - fill with median
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [col for col in num_cols if col not in ["Safe_Road", "Hour", "Is_Late_Night", "Is_Evening"]]
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        data[num_cols] = num_imputer.fit_transform(data[num_cols])
    
    # Categorical columns - fill with most frequent
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

print("Preprocessing completed.")

# Separate features (X) and target (y)
X = data.drop(["Safe_Road", "Time"], axis=1)  # Dropping original Time column
y = data["Safe_Road"]

# Ensure all features are numeric
print("\nFeature types after preprocessing:")
print(X.dtypes)

# Train-Test Split
print("\nSplitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Train-Test split completed.")

# Apply SMOTE to balance classes in training set
print("\nApplying SMOTE to balance class distribution...")
try:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())
except Exception as e:
    print(f"Error applying SMOTE: {e}")
    print("Checking for non-numeric values in features...")
    print(X_train.dtypes)
    # Convert any remaining object columns to numeric
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    # Try SMOTE again
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE (after conversion):")
    print(pd.Series(y_train_resampled).value_counts())

# Train model
print("\nTraining model...")
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Save the model
joblib.dump(model, "final_model.pkl")
print("Model saved as final_model.pkl.")