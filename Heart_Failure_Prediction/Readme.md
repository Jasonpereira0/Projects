# Heart Failure Prediction - README - Jason Pereira

## Overview
This Jupyter Notebook provides a comprehensive workflow for predicting heart disease using machine learning models. It includes data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and evaluation. The notebook also demonstrates how to deploy the trained model for real-world predictions.

## Table of Contents
1. **Introduction**  
    The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It contains clinical data that can be used to predict heart disease.

2. **Data Preparation**  
    - Importing Libraries  
    - Loading the Dataset  
    - Exploratory Data Analysis  
    - Importing Libraries
    - Loading the Dataset
    - Exploratory Data Analysis
3. **Data Preprocessing**
    - Handling Missing and Zero Values
    - Visualizing Numerical Features
4. **Feature Engineering**
    - One-Hot Encoding
    - Feature Scaling
    - Creating New Features
5. **Model Training**
    - Splitting the Data
    - Training Multiple Models
    - Feature Selection
6. **Hyperparameter Tuning**
    - Grid Search for Optimal Parameters
    - Visualizing Hyperparameter Heatmaps
7. **Model Evaluation**
    - Comparing Metrics Across Models
    - Visualizing ROC Curves and Confusion Matrices
    - SHAP Analysis for Feature Importance
8. **Model Deployment**
    - Preset Examples (Moderate and Severe Cases)
    - User Input Prediction
9. **Saving and Loading Models**
10. **Conclusion**

## Key Features
- **Exploratory Data Analysis (EDA):** Visualizes distributions and relationships between features to understand the dataset.
- **Feature Engineering:** Includes one-hot encoding, feature scaling, and creation of new features like age groups and cholesterol levels.
- **Model Training:** Implements multiple machine learning models, including Logistic Regression, Random Forest, XGBoost, KNN, and Decision Tree.
- **Hyperparameter Tuning:** Uses GridSearchCV to optimize model performance.
- **Evaluation Metrics:** Compares models using accuracy, precision, recall, F1-score, and AUC-ROC.
- **SHAP Analysis:** Explains model predictions by identifying the most impactful features.
- **Deployment:** Demonstrates how to use the trained model for predictions with real-world data.

## Models Used
- Logistic Regression
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)
- Decision Tree

## Results
- Logistic Regression emerged as the best-performing model with the highest AUC-ROC score (0.89) and balanced metrics across accuracy, precision, recall, and F1-score.
- SHAP analysis highlighted key features such as `ST_Slope_Up`, `MaxHR`, and `Oldpeak` as the most impactful predictors of heart disease.

## Deployment
The notebook includes examples of how to use the trained model for predictions:
1. **Preset Examples:** Predicts heart disease risk for moderate and severe cases.
2. **User Input Prediction:** Allows users to input their own data for prediction.

## How to Use
1. Clone the repository and ensure all dependencies are installed.
2. Run the notebook step-by-step to reproduce the results.
3. Use the deployment section to test the model with your own data.

## Dependencies
- Python 3.x
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, shap
- pickle

All required libraries are listed in the `requirements.txt` file. Install them using the following command:
```bash
pip install -r requirements.txt
```

## File Structure
- `data/heart.csv`: Input dataset.
- `output/`: Directory for saving plots, models, and other outputs.
    - `best_model.pkl`: Saved model for deployment.
    - `selector.pkl`: Feature selector for preprocessing.
    - `scaler.pkl`: Scaler for numerical features.
    - `onehot_features.pkl`: One-hot encoded feature names.
- `script/`: Folder containing all Python scripts.
- `notebook/`: Folder containing Jupyter Notebooks.

## Conclusion
This notebook provides a complete pipeline for heart disease prediction, from data preprocessing to model deployment. It is designed to be interpretable and user-friendly, making it suitable for both data scientists and healthcare professionals.

## License
This project is licensed under the MIT License.