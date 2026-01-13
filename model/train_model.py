# -----------------------------
# Imports
# -----------------------------
# Data handling
import pandas as pd
import numpy as np

# Model persistence
import joblib
import os

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ============================================================
# 1. Data Loading
#    - Try loading real dataset
#    - If not found, generate dummy data for demo/testing
# ============================================================
try:
    # Attempt to load the real dataset
    df = pd.read_csv('predictive_maintenance.csv')
    print("Dataset 'predictive_maintenance.csv' loaded successfully.")

except FileNotFoundError:
    # Fallback: create a dummy dataset with the same schema
    print("Error: 'predictive_maintenance.csv' not found. Creating dummy DataFrame.")

    data = {
        'UDI': range(1, 101),                              # Unique identifier
        'Product ID': [f'PN{i}' for i in range(1, 101)],  # Product identifier
        'Type': np.random.choice(['L', 'M', 'H'], 100),   # Machine type
        'Air temperature [K]': np.random.uniform(298, 308, 100),
        'Process temperature [K]': np.random.uniform(308, 318, 100),
        'Rotational speed [rpm]': np.random.uniform(1000, 2000, 100),
        'Torque [Nm]': np.random.uniform(0, 200, 100),
        'Tool wear [min]': np.random.uniform(0, 240, 100),
        'Machine failure': np.random.randint(0, 2, 100)  # Target variable
    }

    df = pd.DataFrame(data)

    # Introduce missing values to demonstrate preprocessing
    for col in ['Air temperature [K]', 'Torque [Nm]']:
        df.loc[np.random.choice(df.index, 5, replace=False), col] = np.nan

    # Introduce artificial outliers
    df.loc[df.sample(2).index, 'Rotational speed [rpm]'] = 5000
    df.loc[df.sample(2).index, 'Tool wear [min]'] = 500

    print("Dummy DataFrame created with sample missing values and outliers.")


# ============================================================
# 2. Data Preprocessing & Feature Engineering
# ============================================================
print("\n--- Starting Data Preprocessing and Feature Engineering ---")

# Columns that should not be used as model features
id_columns = ['UDI', 'Product ID']

# Target label used for prediction
target_column = 'Machine failure'

# Identify numerical feature columns (excluding IDs and target)
numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
numerical_columns = [
    col for col in numerical_columns
    if col not in id_columns + [target_column]
]

# Identify categorical feature columns (excluding IDs)
categorical_columns = df.select_dtypes(include='object').columns.tolist()
categorical_columns = [
    col for col in categorical_columns
    if col not in id_columns
]

# -----------------------------
# Handle missing values
# -----------------------------
cols_with_missing_values = (
    df[numerical_columns]
    .columns[df[numerical_columns].isnull().any()]
    .tolist()
)

# Fill missing numerical values with column mean
for col in cols_with_missing_values:
    mean_val = df[col].mean()
    df[col] = df[col].fillna(mean_val)
    print(f"Filled missing values in '{col}' with the mean ({mean_val:.2f}).")

# -----------------------------
# Handle outliers using IQR capping
# -----------------------------
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap extreme values to IQR bounds
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# -----------------------------
# Feature engineering
# -----------------------------
# Temperature difference between process and air temperature
if 'Air temperature [K]' in df.columns and 'Process temperature [K]' in df.columns:
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

    # Ensure engineered feature is included in numerical columns
    if 'Temp_Diff' not in numerical_columns and pd.api.types.is_numeric_dtype(df['Temp_Diff']):
        numerical_columns.append('Temp_Diff')

print("Data preprocessing and feature engineering complete.")


# ============================================================
# 3. Model Training
# ============================================================
print("\n--- Starting Model Training ---")

# Separate features (X) and target (y)
features_df = df.drop(id_columns + [target_column], axis=1)
target_series = df[target_column]

# Column-wise preprocessing:
# - Scale numerical features
# - One-hot encode categorical features
data_preprocessor = ColumnTransformer(
    transformers=[
        ('numerical_scaler', StandardScaler(), numerical_columns),
        ('categorical_encoder', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

# Train-test split with stratification on target
X_train_data, X_test_data, y_train_target, y_test_target = train_test_split(
    features_df,
    target_series,
    test_size=0.2,
    random_state=42,
    stratify=target_series
)

# Random Forest model configuration
random_forest_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

# End-to-end ML pipeline
predictive_maintenance_pipeline = Pipeline(steps=[
    ('preprocessor', data_preprocessor),
    ('classifier', random_forest_classifier)
])

# Train the model
print("Training RandomForestClassifier model...")
predictive_maintenance_pipeline.fit(X_train_data, y_train_target)
print("Model training complete.")

best_model = predictive_maintenance_pipeline


# ============================================================
# 4. Model Evaluation
# ============================================================
print("\n--- Evaluating Model on Test Data ---")

# Generate predictions
y_pred = predictive_maintenance_pipeline.predict(X_test_data)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test_target, y_pred):.4f}")
print(f"Precision: {precision_score(y_test_target, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test_target, y_pred, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test_target, y_pred, average='weighted'):.4f}")

print("\nClassification Report:")
print(classification_report(y_test_target, y_pred))


# ============================================================
# 5. Model Export
# ============================================================
print("\n--- Exporting Trained Model and Preprocessor ---")

# Directory to store trained artifacts
save_directory = 'trained_model'
os.makedirs(save_directory, exist_ok=True)

# Save the full pipeline
joblib.dump(best_model, os.path.join(save_directory, 'best_model.joblib'))

# Save preprocessor separately (optional reuse)
joblib.dump(data_preprocessor, os.path.join(save_directory, 'preprocessor.joblib'))

print("All steps completed.")
