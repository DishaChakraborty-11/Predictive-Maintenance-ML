<div align="center">

# 🛠️⚙️ Predictive Maintenance using Machine Learning  
### **A Production-Ready ML Pipeline for Equipment Failure Prediction**

<br>

<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=github" />
<img src="https://img.shields.io/badge/Machine%20Learning-RandomForest-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Framework-ScikitLearn-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/App-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit" />
<img src="https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge&logo=python" />

<br><br>

✨ **AI-powered system that predicts machine failures using real sensor data.**  
Automatically preprocesses data, trains a RandomForest model, evaluates performance,  
and provides visual insights + an interactive prediction app.

</div>

---

## 🚀 **Features**
✔ End-to-End ML Pipeline (Preprocessing → Training → Evaluation → Export)  
✔ Handles missing values & outliers  
✔ One-Hot Encoding + Scaling with ColumnTransformer  
✔ RandomForestClassifier with feature importance  
✔ Exports trained model + preprocessor (joblib)  
✔ Auto-generated model performance plots  
✔ Ready-to-run Streamlit Web App

---

## 📁 **Project Structure**
 ```
Predictive-Maintenance-ML/
│
├── data/
│   └── sensor_data.csv
│
├── model/
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   └── train_model.py
│
├── static/
│   └── screenshots/
│         ├── accuracy_graph.png
│         ├── confusion_matrix.png
│         └── feature_importance.png
│
├── app/
│   └── app.py
│
└── README.md
```


---

## 📊 **Model Performance Visuals**

### 📈 Model Accuracy
<img src="static/accuracy_graph.png" width="400"/>

### 🔎 Confusion Matrix
<img src="static/confusion_matrix.png" width="500"/>

### 🌟 Feature Importance
<img src="static/feature_importance.png" width="650"/>

---

## ⚙️ **How It Works (ML Pipeline)**

### **1️⃣ Data Preprocessing**
- Missing values filled  
- Outlier handling  
- Scaling numerical features  
- One-hot encoding categorical features  
- Automatic feature engineering (Temp Diff, etc.)

### **2️⃣ Model Training**
- RandomForestClassifier  
- Handles nonlinear relationships  
- Extracts top predictive features  

### **3️⃣ Model Export**
Outputs:



best_model.joblib
preprocessor.joblib


### **4️⃣ Visual Insights**
Automatically generated:

- Confusion Matrix  
- Feature Importance  
- Accuracy Plot  

---


## 📊 Dataset Preparation

No external dataset is required to run this project.

For demonstration purposes, the training script automatically generates a **dummy dataset** that simulates:
- Sensor readings
- Missing values
- Outliers
- Machine failure labels

This allows users to:
- Run the full pipeline out of the box
- Understand preprocessing, feature engineering, and model training flow

### Using a Custom Dataset (Optional)

If you want to use your own dataset:
1. Replace the dummy data generation logic in `train_model.py`
2. Ensure your dataset includes relevant sensor features and a target label
3. Update column names in the preprocessing step accordingly


## 📦 **Installation**

### Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher  
- pip (comes with Python)  
- Git  

Check versions:
```bash
python --version
pip --version
git --version


```bash
git clone https://github.com/yourusername/Predictive-Maintenance-ML.git
cd Predictive-Maintenance-ML
pip install -r requirements.txt

▶️ Run Training Script
cd model
python train_model.py

This script:
- Cleans the data
- Trains a RandomForest model
- Saves the trained model and preprocessor to `trained_model/`


💻 Run the Streamlit App
cd app
streamlit run app.py

This launches a web interface where you can interactively test predictions using the trained model.


📜 Training Logs (Expand to View)
<details> <summary>Click to expand training output</summary>
Dummy DataFrame created with sample missing values and outliers.

--- Starting Data Preprocessing and Feature Engineering ---
Filled missing values in 'Air temperature [K]' with mean (302.59)
Filled missing values in 'Torque [Nm]' with mean (101.92)
Data preprocessing complete.

--- Starting Model Training ---
Training RandomForestClassifier...
Model training complete.

Feature Importances:
Temp_Diff               0.179
Tool wear               0.153
Process temperature     0.153
Torque                  0.139
Air temperature         0.134
Rotational speed        0.134
Type_L                  0.033
Type_M                  0.028
Target                  0.024
Type_H                  0.017

--- Exporting Trained Model ---
model saved to trained_model/best_model.joblib
preprocessor saved to trained_model/preprocessor.joblib

</details>
🧠 Tech Stack

Python

Scikit-Learn

Pandas

NumPy

Matplotlib / Seaborn

Streamlit

Joblib

## 📈 Interpreting the Output

After training, you will see:

- **Feature Importances**: Shows which sensor readings most influence failure prediction.
- **Model saved to** `trained_model/best_model.joblib`
- **Preprocessor saved to** `trained_model/preprocessor.joblib`

Higher feature importance values indicate stronger influence on predictions.


🌟 Future Improvements

🔹 Deploy on cloud (Render / AWS / GCP)
🔹 Hyperparameter tuning
🔹 LSTM-based time-series forecasting
🔹 Real-time sensor data ingestion

<div align="center">
❤️ Like this project? Star ⭐ the repo!
Built with hard work & caffeine by Disha Chakraborty ☕✨
</div> ```
