# 📱 Teen Phone Addiction Risk Predictor

This project predicts the **risk of phone addiction among teenagers** using Machine Learning 🤖.  
It analyzes factors such as **daily phone usage, sleep hours, and anxiety levels** to classify addiction risk.

The project also includes an **interactive Streamlit web app** where users can input details and get predictions.

---

## 📌 Project Overview

Smartphone addiction among teenagers is increasing rapidly.  
This project uses a **Random Forest Classifier** to analyze behavioral patterns and predict addiction risk.

Features used in the model:
- 👤 Age
- 📱 Daily Phone Usage Hours
- 😴 Sleep Hours
- 😟 Anxiety Level

---

## 🛠 Technologies Used

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 🤖 Scikit-learn
- 🌐 Streamlit
- 💾 Joblib

---

## 📂 Project Structure

```
teen-phone-addiction-predictor
│
├── app.py
├── addiction_pipeline.pkl
├── teen_phone_addiction_dataset.csv
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

Install required libraries:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py
