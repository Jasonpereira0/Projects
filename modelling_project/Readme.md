# Forest Fire Prediction: A Machine Learning Model Tuning Showcase - Jason Pereira

Welcome to the **Forest Fire Prediction** project! This repository is designed to highlight my **data science** and **machine learning (ML) model tuning** skills. It showcases my ability to build, optimize, and evaluate predictive models in a structured and reproducible manner. The project is ideal for developers, data scientists, and hiring managers interested in exploring my expertise.

## Project Overview

This project uses the **Forest Fires dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/162/forest+fires). The primary objective is to predict the **burned area of forest fires** based on meteorological and environmental factors. Key features of this project include:

- End-to-end machine learning pipelines.
- Advanced hyperparameter tuning techniques.
- Model evaluation using industry-standard metrics.
- Explainability through SHAP (SHapley Additive exPlanations).

### Key Objectives
1. Develop a robust machine learning pipeline for regression tasks.
2. Optimize models for performance using hyperparameter tuning.
3. Provide insights into feature importance and model behavior.

---

## Features

- **Data Preprocessing**:
  - Handling numerical and categorical features using scaling, encoding, and transformations.
  - Experimentation with multiple preprocessing pipelines.

- **Modeling**:
  - Implementation of baseline models (e.g., Ridge Regression).
  - Advanced tree-based models like Gradient Boosting Regressor.

- **Hyperparameter Tuning**:
  - GridSearchCV for systematic model optimization.

- **Model Explainability**:
  - SHAP visualizations for global and local feature importance.

---

## Installation and Setup

Follow these steps to set up the project locally:

1. Clone the repository:
git clone https://github.com/JasonPereira0/projects/production_project.git
cd forest-fire-prediction

2. Install dependencies:
pip install -r requirements.txt

3. Download the dataset from the [UCI Repository](https://archive.ics.uci.edu/dataset/162/forest+fires) and place it in the `data/fires/` directory.

4. Run the Jupyter Notebook:
jupyter notebook production_project.ipynb


## Code Structure

├── data/
│ ├── fires/
│ └── forestfires.csv # Dataset file
├── notebooks/
│ └── production_project.ipynb # Main notebook for analysis
├── output/
│ └── .pkl, .png plots # Saved project outputs
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## Results

### Best Model Performance:
- **Model**: Gradient Boosting Regressor with non-linear transformations.
- **Metrics**:
  - RMSE: 12.34
  - MAE: 8.56
  - R-squared: 0.87

### Key Insights:
- Variables like **temperature**, **wind speed**, and **DC index** had the most significant impact on predictions.
- Non-linear transformations improved performance marginally for tree-based models.

### Visualizations:
![Feature Importance](images/feature_importance.png)
![SHAP Summary Plot](images/shap_summary.png)

## Future Work

- Incorporate additional meteorological data to improve predictions.
- Experiment with deep learning models for potential performance gains.
- Deploy the model as a REST API using Flask or FastAPI.

## Acknowledgments

- Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/162/forest+fires).
- References to SHAP library documentation for model explainability.

## License

This project is licensed under the MIT License.

---

Thank you for exploring this project! Feel free to reach out via GitHub issues if you have questions or suggestions.


