import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

st.write("## Personal Fitness Tracker")
st.write("In this WebApp, you can predict the number of kilocalories burned based on input parameters such as `Age`, `Gender`, `BMI`, etc.")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 20.0)
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 38.0)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    
    gender_encoded = 1 if gender == "Male" else 0
    
    data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_encoded
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

st.write("---")
st.header("Your Parameters")
st.write(df)

# Load data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge datasets
exercise_df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
exercise_df["BMI"] = round(exercise_df["Weight"] / (exercise_df["Height"] / 100) ** 2, 2)

# Select relevant features
exercise_df = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_df = pd.get_dummies(exercise_df, drop_first=True)

# Split dataset
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
X_train, y_train = train_data.drop("Calories", axis=1), train_data["Calories"]
X_test, y_test = test_data.drop("Calories", axis=1), test_data["Calories"]

# Train model
model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=1)
model.fit(X_train, y_train)

# Ensure input matches training data columns
df = df.reindex(columns=X_train.columns, fill_value=0)

# Predict
prediction = model.predict(df)[0]

st.write("---")
st.header("Prediction")
st.write(f"**{round(prediction, 2)} kilocalories**")

st.write("---")
st.header("Similar Results")
similar_data = exercise_df[(exercise_df["Calories"] >= prediction - 10) & (exercise_df["Calories"] <= prediction + 10)]
st.write(similar_data.sample(min(5, len(similar_data))))

st.write("---")
st.header("General Information")
st.write(f"You are older than {round((exercise_df['Age'] < df['Age'].values[0]).mean() * 100, 2)}% of other people.")
st.write(f"Your exercise duration is higher than {round((exercise_df['Duration'] < df['Duration'].values[0]).mean() * 100, 2)}% of other people.")
st.write(f"You have a higher heart rate than {round((exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100, 2)}% of other people during exercise.")
st.write(f"You have a higher body temperature than {round((exercise_df['Body_Temp'] < df['Body_Temp'].values[0]).mean() * 100, 2)}% of other people during exercise.")
