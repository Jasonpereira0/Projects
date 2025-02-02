# McDonald's Menu Analysis and Prediction

## Overview
This project analyzes McDonald's menu items to uncover nutritional insights and build a predictive model to classify high-calorie items. The dataset includes detailed nutritional information for various food and beverage items.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Project Objective](#project-objective)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Model](#machine-learning-model)
- [Visualizations](#visualizations)
- [Instructions for Replication](#instructions-for-replication)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Dataset Description
The dataset, `McDonalds_menu_data.csv`, contains nutritional information for menu items across categories like Breakfast, Beef & Pork, Chicken & Fish, Salads, Desserts, Beverages, etc.

### Key Columns:
- `Category`: The type of menu item (e.g., Breakfast, Desserts).
- `Item`: Name of the menu item.
- `Calories`: Total calories in the item.
- `Total Fat`, `Sodium`, `Carbohydrates`, etc.: Nutritional breakdown.

## Project Objective
The primary goals of this project are:
1. Conduct exploratory data analysis to identify trends and patterns in McDonald's menu items.
2. Engineer new features to enhance data insights.
3. Build a machine learning model to classify menu items as high-calorie or low-calorie based on nutritional data.

## Exploratory Data Analysis (EDA)
Key findings from EDA include:
- **High-Calorie Categories**: Items in the "Smoothies & Shakes" category have the highest average calorie count.
- **Sodium Levels**: Many items exceed recommended daily sodium intake levels.
- **Correlation Insights**: Calories are strongly correlated with Total Fat and Carbohydrates.

![Correlation Heatmap](outputs/correlation_heatmap.png)

## Feature Engineering
New features created include:
- `Calories_from_Fat_Percentage`: Percentage of calories derived from fat.
- One-hot encoding for the categorical variable `Category`.

## Machine Learning Model
A Random Forest Classifier was used to predict whether an item is high-calorie (>500 calories). 

### Model Performance:
- Accuracy: 85%
- Precision: 0.88
- Recall: 0.82

## Visualizations
Key visualizations include:
1. Distribution of Calories across categories.
2. Correlation heatmap of numerical features.

![Calories Distribution](outputs/calories_distribution.png)

## Instructions for Replication
To replicate this project locally:
1. Clone this repository:git clone https://github.com/yourusername/McDonalds_Menu_Analysis.git
2. Navigate to the project directory: cd McDonalds_Menu_Analysis
3. Install required dependencies:pip install -r requirements.txt
4. Run the Jupyter notebooks in the `/notebooks` folder for EDA and modeling.

## Future Work
Potential improvements include:
1. Adding more advanced machine learning models like XGBoost or Neural Networks.
2. Exploring additional datasets with customer preferences or sales data.

## Acknowledgments
Dataset provided by McDonald's Nutrition Information (publicly available). Special thanks to contributors who inspired this analysis.




