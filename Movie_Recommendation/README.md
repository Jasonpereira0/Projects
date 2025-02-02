# **Movie Recommendation System using the MovieLens Dataset**

## **Overview**
This project demonstrates the development of a personalized movie recommendation system using the **MovieLens dataset**. The system leverages collaborative filtering techniques to predict user preferences and recommend movies they are likely to enjoy. The project is designed to showcase practical data science and machine learning skills, including data preprocessing, exploratory data analysis, model building, and evaluation.

---

## **Table of Contents**
- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Installation and Setup](#installation-and-setup)
- [Modeling Approach](#modeling-approach)
- [Results and Insights](#results-and-insights)
- [Key Features](#key-features)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## **Dataset Description**
The MovieLens dataset is a widely used dataset for building recommendation systems. It contains information on:
- **Ratings**: User ratings for movies (on a scale of 0.5 to 5).
- **Movies**: Metadata about movies, including titles and genres.
- **Users**: Anonymized user IDs.

For this project, we used the **MovieLens 100K dataset**, which includes:
- 100,000 ratings from 943 users on 1,682 movies.
- Data files: `ratings.csv` and `movies.csv`.

You can download the dataset from the [GroupLens website](https://grouplens.org/datasets/movielens/).

---

## **Project Workflow**
The project follows an end-to-end data science workflow:
1. **Data Loading and Exploration**:
   - Loaded and merged the ratings and movies datasets.
   - Explored data distributions, missing values, and trends.
2. **Data Preprocessing**:
   - Filtered outliers and handled missing data.
   - Created user-item interaction matrices.
3. **Exploratory Data Analysis (EDA)**:
   - Visualized rating distributions and user behavior.
   - Analyzed popular genres and highly rated movies.
4. **Modeling**:
   - Built a collaborative filtering-based recommendation system using the Singular Value Decomposition (SVD) algorithm.
   - Evaluated model performance using Root Mean Square Error (RMSE).
5. **Insights and Recommendations**:
   - Generated personalized movie recommendations for users.
   - Visualized top recommendations.

---

## **Installation and Setup**
Follow these steps to set up the project locally:

1. Clone the repository: git clone https://github.com/JasonPereira0/Projects/tree/master/Movie_Recommendation

cd movie_recommendation

2. Install dependencies: pip install numpy pandas matplotlib seaborn scikit-learn surprise


3. Download the MovieLens dataset (100K version) from [here](https://grouplens.org/datasets/movielens/) and place it in the `data/` folder.

4. Run the Jupyter Notebook or Python script: jupyter notebook Movie_Recommendation_System.ipynb


---

## **Modeling Approach**
The recommendation system uses a **collaborative filtering approach**, specifically:
1. **Algorithm**: Singular Value Decomposition (SVD).
2. **Train-Test Split**: The dataset was split into training (75%) and testing (25%) sets.
3. **Evaluation Metric**: RMSE was used to measure prediction accuracy.

### Code Snippet for Model Training:
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)


---

## **Results and Insights**
### Key Results:
1. Achieved an RMSE of approximately `0.87` on the test set.
2. Successfully generated personalized recommendations for users.

### Insights:
- Popular genres like *Drama* and *Comedy* received higher average ratings.
- Users with more interactions tend to have stable rating patterns.
- The model effectively recommends movies based on user preferences.

### Example Recommendations for User `1`:
| Movie Title               | Predicted Rating |
|---------------------------|------------------|
| The Godfather             | 4.8              |
| Schindler's List          | 4.7              |
| Pulp Fiction              | 4.6              |

---

## **Key Features**
- End-to-end implementation of a collaborative filtering recommendation system.
- Clear visualizations of rating distributions and trends.
- Personalized movie recommendations based on user preferences.
- Modular code structure for easy understanding and reuse.

---

## **Future Work**
Potential improvements include:
1. Implementing content-based filtering using movie metadata (e.g., genres).
2. Exploring hybrid models that combine collaborative and content-based approaches.
3. Deploying the model as a web application using Flask or Streamlit.

---

## **Contributing**
Contributions are welcome! If you have suggestions or find issues, feel free to open an issue or submit a pull request.

---

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

---



