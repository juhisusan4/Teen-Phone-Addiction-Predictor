import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIG
st.set_page_config(page_title="Teen Phone Addiction Dashboard")

# LOAD MODEL & DATA
model = joblib.load("addiction_pipeline.pkl")
df = pd.read_csv("teen_phone_addiction_dataset.csv")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["About Project", "Dataset Overview", "EDA Analysis", "Model Performance", "Prediction"]
)
if page == "About Project":
    st.title("📱 Teen Phone Addiction Prediction System")

    st.write("""
    This project predicts the level of phone addiction among teenagers 
    using Machine Learning techniques.

    The model analyzes behavioral, psychological, and usage pattern features 
    to classify addiction levels into:
    - Low
    - Medium
    - High

    Objective:
    - Identify risky phone usage behavior
    - Understand addiction patterns
    - Provide early risk indication
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Target Classes", df["Addiction_Level"].nunique())

elif page == "Dataset Overview":
    st.title("Dataset Overview")

    st.subheader("First 5 Rows")
    st.write(df.head())

    st.subheader("Dataset Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Average Daily Usage", round(df["Daily_Usage_Hours"].mean(),2))


elif page == "EDA Analysis":

    st.title("Exploratory Data Analysis")

    # Convert addiction score to categories
    df["Addiction_Level_Label"] = pd.cut(
        df["Addiction_Level"],
        bins=[0,4,7,10],
        labels=["Low","Medium","High"]
    )

    # Addiction distribution
    st.subheader("Addiction Level Distribution")

    addiction_counts = df["Addiction_Level_Label"].value_counts().reindex(["Low","Medium","High"])
    st.bar_chart(addiction_counts)

    # Usage vs addiction
    st.subheader("Average Daily Usage by Addiction Level")

    usage_data = df.groupby("Addiction_Level_Label")["Daily_Usage_Hours"].mean()
    st.bar_chart(usage_data)

    # Gender vs addiction
    st.subheader("Addiction Level by Gender")

    gender_addiction = pd.crosstab(df["Gender"], df["Addiction_Level_Label"])
    st.bar_chart(gender_addiction)

# =============================
# =============================
# MODEL PERFORMANCE
# =============================
elif page == "Model Performance":

    st.title("Model Performance")

    st.write("Model Used: Random Forest Classifier")
    st.write("Evaluation Metrics: Accuracy, Precision, Recall, F1-score")

    model_rf = model.named_steps["model"]

    # Get feature names from pipeline
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    importances = model_rf.feature_importances_

    # Clean names (remove num__ / cat__)
    feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    top_features = feature_importance_df.sort_values(
        by="Importance", ascending=False
    ).head(10)

    st.subheader("Top 10 Important Features")

    st.bar_chart(top_features.set_index("Feature"))



elif page == "Prediction":

    st.title("📱 Teen Phone Addiction Predictor")

    age = st.number_input("Age", 10, 20, 15)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Location", ["Urban", "Rural", "Suburban"])
    school_grade = st.selectbox("School Grade", ["8th", "9th", "10th", "11th", "12th"])

    daily_usage = st.number_input("Daily Usage Hours", 0.0, 24.0, 2.0)
    sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 8.0)
    academic = st.slider("Academic Performance (1-10)", 1, 10, 5)
    social = st.slider("Social Interactions (1-10)", 1, 10, 5)
    exercise = st.number_input("Exercise Hours", 0.0, 5.0, 1.0)

    anxiety = st.slider("Anxiety Level (1-10)", 1, 10, 5)
    depression = st.slider("Depression Level (1-10)", 1, 10, 5)
    self_esteem = st.slider("Self Esteem (1-10)", 1, 10, 5)
    parental = st.slider("Parental Control (1-10)", 1, 10, 5)

    screen_before_bed = st.number_input("Screen Time Before Bed", 0.0, 5.0, 1.0)
    phone_checks = st.number_input("Phone Checks Per Day", 0, 300, 50)
    apps_used = st.number_input("Apps Used Daily", 0, 50, 10)

    social_media = st.number_input("Time on Social Media", 0.0, 12.0, 2.0)
    gaming = st.number_input("Time on Gaming", 0.0, 12.0, 1.0)
    education = st.number_input("Time on Education", 0.0, 12.0, 2.0)

    purpose = st.selectbox("Phone Usage Purpose", ["Social Media", "Gaming", "Education", "Mixed"])
    family_comm = st.slider("Family Communication (1-10)", 1, 10, 5)

    weekend_usage = st.number_input("Weekend Usage Hours", 0.0, 24.0, 4.0)

    if st.button("Predict Addiction Level"):

        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Location": [location],
            "School_Grade": [school_grade],
            "Daily_Usage_Hours": [daily_usage],
            "Sleep_Hours": [sleep_hours],
            "Academic_Performance": [academic],
            "Social_Interactions": [social],
            "Exercise_Hours": [exercise],
            "Anxiety_Level": [anxiety],
            "Depression_Level": [depression],
            "Self_Esteem": [self_esteem],
            "Parental_Control": [parental],
            "Screen_Time_Before_Bed": [screen_before_bed],
            "Phone_Checks_Per_Day": [phone_checks],
            "Apps_Used_Daily": [apps_used],
            "Time_on_Social_Media": [social_media],
            "Time_on_Gaming": [gaming],
            "Time_on_Education": [education],
            "Phone_Usage_Purpose": [purpose],
            "Family_Communication": [family_comm],
            "Weekend_Usage_Hours": [weekend_usage]
        })

        prediction = model.predict(input_data)[0]

        if prediction == "Low":
            st.success("Predicted Addiction Level: LOW")
        elif prediction == "Medium":
            st.warning("Predicted Addiction Level: MEDIUM")
        else:
            st.error("Predicted Addiction Level: HIGH")

        prob = model.predict_proba(input_data)[0]

        risk_df = pd.DataFrame({
            "Addiction Level": model.classes_,
            "Probability": prob
        })

        st.subheader("Addiction Risk Graph 📊")
        st.bar_chart(risk_df.set_index("Addiction Level"))