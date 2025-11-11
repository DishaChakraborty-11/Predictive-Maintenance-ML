import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load Data
df = pd.read_csv("../data/sensor_data.csv")

X = df.drop("is_failure", axis=1)
y = df["is_failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Create folders
os.makedirs("../static/screenshots", exist_ok=True)

# Confusion Matrix Plot
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix")
plt.savefig("../static/screenshots/confusion_matrix.png")
plt.close()

# Feature Importance Plot
feat_importance = pd.Series(model.feature_importances_, index=df.columns[:-1])
feat_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.savefig("../static/screenshots/feature_importance.png")
plt.close()

print(classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump(model, "../model/failure_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("✅ Model and scaler saved successfully")
