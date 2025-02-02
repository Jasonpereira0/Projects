# **Sales Forecasting for Large Retail Store**

## **Project Overview**
This project focuses on building a predictive model to forecast sales for a large retail store using advanced machine learning techniques. The dataset contains sales data for various products across multiple outlets, and the goal is to predict future sales based on historical data and other features. 

The project demonstrates skills in data preprocessing, exploratory data analysis (EDA), feature engineering, and model implementation using machine learning algorithms such as Random Forest, XGBoost, and LightGBM.

---

## **Key Features**
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and creating new features.
- **Exploratory Data Analysis (EDA):** Visualizing trends, relationships, and distributions in the dataset.
- **Machine Learning Models:** Implementation of Random Forest, XGBoost, and LightGBM for regression tasks.
- **Model Evaluation:** Comparison of models using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
- **Actionable Insights:** Recommendations for improving model performance and business decision-making.

---

## **Dataset**
The dataset used in this project is the [BigMart Sales Dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data). It contains information about:
- Product attributes (e.g., `Item_Type`, `Item_MRP`, etc.)
- Outlet attributes (e.g., `Outlet_Type`, `Outlet_Size`, etc.)
- Sales figures (`Item_Outlet_Sales`) for each product-outlet combination.

### **Dataset Columns**
| Column Name                | Description                                   |
|----------------------------|-----------------------------------------------|
| `Item_Identifier`          | Unique identifier for each product           |
| `Item_Weight`              | Weight of the product                        |
| `Item_Fat_Content`         | Fat content of the product                   |
| `Item_Visibility`          | Percentage visibility of the product         |
| `Item_Type`                | Category of the product                      |
| `Item_MRP`                 | Maximum Retail Price (MRP) of the product    |
| `Outlet_Identifier`        | Unique identifier for each outlet            |
| `Outlet_Establishment_Year`| Year the outlet was established              |
| `Outlet_Size`              | Size of the outlet (Small/Medium/High)       |
| `Outlet_Location_Type`     | Location type of the outlet (Tier 1/2/3)     |
| `Outlet_Type`              | Type of outlet (e.g., Grocery Store)         |
| `Item_Outlet_Sales`        | Sales figure for each product-outlet pair    |

---

## **Project Workflow**

### **1. Data Preprocessing**
- Handled missing values in columns such as `Item_Weight` and `Outlet_Size`.
- Standardized inconsistent categorical values in columns like `Item_Fat_Content`.
- Created new features such as `Years_Since_Establishment`.

### **2. Exploratory Data Analysis (EDA)**
- Visualized sales distribution across different outlets and item categories.
- Explored relationships between features and sales using boxplots, histograms, and scatter plots.

### **3. Model Implementation**
Implemented three machine learning models:
1. **Random Forest Regressor:** A robust ensemble method that uses decision trees.
2. **XGBoost Regressor:** A gradient boosting algorithm optimized for speed and performance.
3. **LightGBM Regressor:** A fast, efficient gradient boosting framework designed for large datasets.

### **4. Model Evaluation**
Evaluated model performance using:
- **Mean Absolute Error (MAE):** Measures average prediction error.
- **Root Mean Squared Error (RMSE):** Penalizes larger errors more heavily.
- **R² Score:** Indicates how well the model explains variance in sales data.

---

## **Results**

### **Model Performance Comparison**
| Metric          | Random Forest       | XGBoost            | LightGBM          |
|------------------|---------------------|--------------------|-------------------|
| **MAE**         | 765.72              | N/A                | 731.16           |
| **RMSE**        | 1094.25             | N/A                | 1055.39          |
| **R² Score**    | 0.559               | 0.511              | 0.590            |

### **Key Insights**
- LightGBM outperformed both Random Forest and XGBoost across all metrics.
- It achieved the lowest MAE (**731.16**) and RMSE (**1055.39**) while explaining approximately 59% of the variance in sales data (**R² = 0.590**).

---

## **Technologies Used**
- Python
- Jupyter Notebook
- Libraries: 
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - LightGBM
  - XGBoost

---

## **How to Run the Project**

### Prerequisites
1. Install Python (3.8 or higher).
2. Install required libraries: pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost


### Steps to Run
1. Clone this repository: git clone https://github.com/Jasonpereira0/Projects/sales_forecasting_model_retail.git
2. Navigate to the project directory: cd sales_forecasting_model_retail
3. Open the Jupyter Notebook file (`Sales_Forecasting.ipynb`) in VS Code or Jupyter Lab.
4. Run all cells sequentially to preprocess data, train models, and evaluate results.

---

## **Future Work**
To further improve this project:
1. Perform hyperparameter tuning for all models using Grid Search or Bayesian Optimization.
2. Experiment with ensembling techniques like stacking or blending.
3. Incorporate external datasets (e.g., economic indicators or seasonal trends) to enhance predictions.
4. Address potential outliers or imbalances in the dataset.

---

## **Acknowledgments**
This project uses the BigMart Sales Dataset available on Kaggle: [BigMart Sales Dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data).

Special thanks to Kaggle contributors for providing this valuable dataset!

---

## **Contact**
For questions or collaboration opportunities, feel free to reach out:
- Name: Jason Pereira
- Email: jason.pere@gmail.com
- GitHub: [JasonPereira0](https://github.com/jasonpereira0)




