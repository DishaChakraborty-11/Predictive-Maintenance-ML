Predictive Maintenance using Machine Learning

A real-world ML project that predicts whether a machine is likely to fail soon using sensor data such as temperature, pressure, vibration, voltage, and runtime.

This project demonstrates end-to-end ML development, including data preprocessing, feature engineering, model training, evaluation, and a working Streamlit web app for real-time predictions.

🚀 Project Overview

Predictive Maintenance helps industries avoid unexpected equipment failures by analyzing sensor data and predicting when maintenance is required.

Industries using PdM:
✅ Manufacturing
✅ Railways
✅ Heavy Machinery
✅ Power Plants
✅ Automobile & Aerospace

This ML model predicts:

1 → Machine failure likely

0 → Machine is running normally

✅ Tech Stack

Machine Learning

Random Forest Classifier

Feature Engineering

StandardScaler

Evaluation Metrics

Python Libraries

pandas, numpy

scikit-learn

seaborn, matplotlib

joblib

streamlit

Others

Streamlit (Frontend)

Jupyter Notebook (Exploration)

📁 Folder Structure
Predictive-Maintenance-ML/
│
├── data/
│   └── sensor_data.csv
│
├── notebooks/
│   └── predictive_maintenance.ipynb
│
├── model/
│   ├── train_model.py
│   ├── failure_model.pkl
│   └── scaler.pkl
│
├── app/
│   ├── app.py
│   ├── styles.css
│   └── helper.py
│
├── static/
│   └── screenshots/
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       └── accuracy_graph.png
│
├── requirements.txt
├── README.md
└── .gitignore

🔍 Model Training & Evaluation

The ML pipeline includes:

✅ 1. Data Preprocessing

Handling missing values

Scaling continuous features

Train-test split

✅ 2. Model

RandomForestClassifier (200 trees)

Handles non-linear behavior

Great for sensor-based predictions

✅ 3. Evaluation Metrics

Confusion Matrix

Precision, Recall, F1

Feature Importance

📊 Results & Visualizations
✅ Confusion Matrix
<img src="static/screenshots/confusion_matrix.png" width="450">
✅ Feature Importance
<img src="static/screenshots/feature_importance.png" width="450">

(Add more screenshots if you want!)

🖥️ Streamlit Web App

The project includes an interactive UI to test machine health.

✅ Features:

Input sensor data manually

Real-time ML prediction

Risk-based result display

Works offline

✅ To Run App:
cd app
streamlit run app.py

🛠️ How to Run the Project Locally
1️⃣ Clone Repo
git clone <your-repo-link>

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model (optional)
python model/train_model.py

4️⃣ Run Web App
streamlit run app/app.py

📈 Future Improvements

✅ Add real industrial IoT sensor data
✅ Deploy using AWS / Azure / Streamlit Cloud
✅ Add LSTM for time-series prediction
✅ Add anomaly detection module
✅ Connect to a live dashboard (Grafana / MQTT)

👩‍💻 Developed By

Disha Chakraborty
AI & Machine Learning Enthusiast
