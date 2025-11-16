import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- 1. Data Loading (with dummy data creation if CSV not found) ---
try:
    df = pd.read_csv('predictive_maintenance.csv')
    print("Dataset 'predictive_maintenance.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'predictive_maintenance.csv' not found. Creating dummy DataFrame.")
    # Create a dummy dataset as in previous steps
    data = {
        'UDI': range(1, 101),
        'Product ID': [f'PN{i}' for i in range(1, 101)],
        'Type': np.random.choice(['L', 'M', 'H'], 100),
        'Air temperature [K]': np.random.uniform(298, 308, 100),
        'Process temperature [K]': np.random.uniform(308, 318, 100),
        'Rotational speed [rpm]': np.random.uniform(1000, 2000, 100),
        'Torque [Nm]': np.random.uniform(0, 200, 100),
        'Tool wear [min]': np.random.uniform(0, 240, 100),
        'Machine failure': np.random.randint(0, 2, 100),
        'Target': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)

    # Introduce some missing values and outliers for preprocessing demonstration
    for col in ['Air temperature [K]', 'Torque [Nm]']:
        df.loc[np.random.choice(df.index, 5, replace=False), col] = np.nan
    df.loc[df.sample(2).index, 'Rotational speed [rpm]'] = 5000
    df.loc[df.sample(2).index, 'Tool wear [min]'] = 500
    print("Dummy DataFrame created with sample missing values and outliers.")

# --- 2. Data Preprocessing and Feature Engineering ---
print("\n--- Starting Data Preprocessing and Feature Engineering ---")

# Define columns
id_columns = ['UDI', 'Product ID']
target_column = 'Machine failure'

numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
if 'Target' in numerical_columns:
    numerical_columns.remove('Target')
numerical_columns = [col for col in numerical_columns if col not in id_columns + [target_column]]

categorical_columns = df.select_dtypes(include='object').columns.tolist()
categorical_columns = [col for col in categorical_columns if col not in id_columns]

# Handle missing values
cols_with_missing_values = df[numerical_columns].columns[df[numerical_columns].isnull().any()].tolist()
if cols_with_missing_values:
    for col in cols_with_missing_values:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        print(f"Filled missing values in '{col}' with the mean ({mean_val:.2f}).")

# Handle outliers using IQR capping
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Feature engineering: Temp_Diff
if 'Air temperature [K]' in df.columns and 'Process temperature [K]' in df.columns:
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    if 'Temp_Diff' not in numerical_columns and pd.api.types.is_numeric_dtype(df['Temp_Diff']):
         numerical_columns.append('Temp_Diff')

print("Data preprocessing and feature engineering complete.")

# --- 3. Model Training ---
print("\n--- Starting Model Training ---")

# Define features (X) and target (y)
features_df = df.drop(id_columns + [target_column], axis=1)
target_series = df[target_column]

# Define preprocessor
data_preprocessor = ColumnTransformer(
    transformers=[
        ('numerical_scaler', StandardScaler(), numerical_columns),
        ('categorical_encoder', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

# Split data into training and testing sets
X_train_data, X_test_data, y_train_target, y_test_target = train_test_split(
    features_df,
    target_series,
    test_size=0.2,
    random_state=42,
    stratify=target_series
)

# Hyperparameters for RandomForestClassifier
rf_n_estimators = 200
rf_max_depth = 15
rf_random_state = 42

# Instantiate RandomForestClassifier
random_forest_classifier = RandomForestClassifier(
    n_estimators=rf_n_estimators,
    max_depth=rf_max_depth,
    random_state=rf_random_state
)

# Create a pipeline
predictive_maintenance_pipeline = Pipeline(steps=[
    ('preprocessor', data_preprocessor),
    ('classifier', random_forest_classifier)
])

# Train the pipeline
print("Training RandomForestClassifier model...")
predictive_maintenance_pipeline.fit(X_train_data, y_train_target)
print("Model training complete.")

# Assign to best_model for consistency if this were part of a larger notebook
best_model = predictive_maintenance_pipeline

# Get and print feature importances (optional, but good for understanding)
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    final_feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    feature_importance_series = pd.Series(feature_importances, index=final_feature_names)
    feature_importance_series_sorted = feature_importance_series.sort_values(ascending=False)
    print("\nFeature Importances (RandomForestClassifier):")
    print(feature_importance_series_sorted)

# --- 4. Model Export ---
print("\n--- Exporting Trained Model and Preprocessor ---")
save_directory = 'trained_model'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print(f"Directory '{save_directory}' created.")

model_filepath = os.path.join(save_directory, 'best_model.joblib')
preprocessor_filepath = os.path.join(save_directory, 'preprocessor.joblib')

# Save the best_model object (the entire pipeline)
try:
    joblib.dump(best_model, model_filepath)
    print(f"Trained model saved successfully to '{model_filepath}'")
except Exception as e:
    print(f"Error saving the trained model: {e}")

# Save the preprocessor object separately if needed for other applications (though it's in the pipeline too)
try:
    joblib.dump(data_preprocessor, preprocessor_filepath)
    print(f"Preprocessor saved successfully to '{preprocessor_filepath}'")
except Exception as e:
    print(f"Error saving the preprocessor: {e}")

print("All steps completed.")
