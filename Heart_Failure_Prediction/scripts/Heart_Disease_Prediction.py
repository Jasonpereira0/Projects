#First, import the necessary libraries for data manipulation, analysis, and visualization.

import pickle
import pandas as pd
import numpy as np


def generate_prediction():
    # Load the saved model and necessary objects
    with open('../output/best_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('../output/selector.pkl', 'rb') as f:
        loaded_selector = pickle.load(f)
    with open('../output/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    with open('../output/onehot_features.pkl', 'rb') as f:
        onehot_features = pickle.load(f)

    # Ask for user input
    age = float(input("Enter your age: "))
    sex = input("Enter your sex (M/F): ")
    chest_pain_type = input("Enter your chest pain type (ATA/NAP/ASY/TA): ")
    resting_bp = float(input("Enter your resting blood pressure: "))
    cholesterol = float(input("Enter your cholesterol level: "))
    fasting_bs = float(input("Enter your fasting blood sugar (0/1): "))
    resting_ecg = input("Enter your resting ECG result (Normal/ST/LVH): ")
    max_hr = float(input("Enter your maximum heart rate: "))
    exercise_angina = input("Do you have exercise angina? (Y/N): ")
    oldpeak = float(input("Enter your old peak value: "))
    st_slope = input("Enter your ST slope (Up/Flat/Down): ")

    # Display user inputs
    print("\n--- User Input ---")
    print(f"Age: {age}")
    print(f"Sex: {sex}")
    print(f"Chest Pain Type: {chest_pain_type}")
    print(f"Resting Blood Pressure: {resting_bp}")
    print(f"Cholesterol Level: {cholesterol}")
    print(f"Fasting Blood Sugar: {fasting_bs}")
    print(f"Resting ECG Result: {resting_ecg}")
    print(f"Maximum Heart Rate: {max_hr}")
    print(f"Exercise Angina: {exercise_angina}")
    print(f"Oldpeak Value: {oldpeak}")
    print(f"ST Slope: {st_slope}")

    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # One-hot encode categorical variables
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    user_data = pd.get_dummies(user_data, columns=categorical_cols, drop_first=True)

    # Create age groups
    user_data['AgeGroup'] = pd.cut(user_data['Age'], bins=[0, 30, 45, 60, np.inf], labels=['Young', 'Adult', 'MiddleAged', 'Senior'])
    user_data = pd.get_dummies(user_data, columns=['AgeGroup'], drop_first=True)

    # Create cholesterol level categories
    user_data['CholesterolLevel'] = pd.cut(user_data['Cholesterol'], bins=[0, 200, 240, np.inf], labels=['Low', 'Medium', 'High'])
    user_data = pd.get_dummies(user_data, columns=['CholesterolLevel'], drop_first=True)

    # Scale numerical features
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    user_data[numerical_cols] = loaded_scaler.transform(user_data[numerical_cols])

    # Add missing features with zeros
    for feature in onehot_features:
        if feature not in user_data.columns:
            user_data[feature] = 0

    # Reorder features to match the original dataset
    user_data = user_data[onehot_features]

    # Display the feature-engineered data without False data
    filtered_columns = user_data.loc[:, (user_data != False).any(axis=0)].drop(columns=['Age', 'MaxHR', 'Oldpeak'])
    print("\n--- Feature-Engineered Data (Filtered) ---")
    print(filtered_columns.to_string(index=False))

    # Select features
    user_data_selected = loaded_selector.transform(user_data)

    # Generate prediction
    prediction = loaded_model.predict(user_data_selected)

    # Display prediction result
    prediction_label = "Risk of Heart Disease Detected" if prediction[0] == 1 else "Low Risk of Heart Disease Detected"

    print("\n--- Prediction ---")
    print(prediction_label)

# Call the function to generate a prediction
generate_prediction()
