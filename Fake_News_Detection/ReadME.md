# **Fake News Detection Using Machine Learning**

## **Overview**
This project demonstrates the development of a machine learning model to classify news articles as either *fake* or *real*. The project uses Natural Language Processing (NLP) techniques for text preprocessing and a Logistic Regression model for classification. The dataset consists of labeled news articles, and the model achieves high accuracy in distinguishing between fake and real news.

---

## **Project Structure**
The project is organized into the following structure:

Fake_News_Detection/
├── data/ # Contains dataset files (True.csv and Fake.csv)
├── output/ # Saved models and vectorizers after training
├── scripts/ # Python scripts for preprocessing, training, and evaluation
└── README.md # Project documentation


- **data/**: Contains the dataset files (`True.csv` for real news and `Fake.csv` for fake news).
- **output/**: Stores the trained Logistic Regression model (`fake_news_model.pkl`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`).
- **scripts/**: Contains Python scripts used for preprocessing, training, and evaluation.

---

## **Dataset Description**
The dataset used in this project consists of two CSV files:
1. **True.csv**: Articles labeled as real news.
2. **Fake.csv**: Articles labeled as fake news.

### Dataset Statistics:
- Total articles: 44,898
  - Real news: ~21,417
  - Fake news: ~23,481
- Columns:
  - `title`: The title of the news article.
  - `text`: The body of the news article.
  - `subject`: The category of the news article (e.g., politics, world news).
  - `date`: The publication date of the article.
- Average text length: ~1,993 characters per article.

---

## **Objective**
The goal of this project is to:
1. Preprocess text data using NLP techniques (e.g., tokenization, stopword removal).
2. Convert text data into numerical features using TF-IDF Vectorization.
3. Train a Logistic Regression model to classify articles as fake or real.
4. Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

---

## **Technologies Used**
- Programming Language: Python
- Libraries:
  - **Pandas**: Data manipulation and analysis.
  - **SpaCy**: Natural Language Processing for text preprocessing.
  - **Scikit-learn**: Machine learning algorithms and evaluation metrics.
  - **Joblib**: Model serialization.
  - **Time**: Execution time measurement.

---

## **How It Works**
### Step-by-Step Process:
1. **Data Loading**:
   - Load `True.csv` and `Fake.csv` into Pandas DataFrames.
   - Combine both datasets into a single DataFrame with labels (`1` for real news, `0` for fake news).

2. **Text Preprocessing**:
   - Convert text to lowercase.
   - Tokenize text into individual words.
   - Remove punctuation, numbers, and stopwords (e.g., "the," "is").
   - Truncate articles to the first 500 words to reduce processing time.

3. **Feature Extraction**:
   - Use TF-IDF Vectorization to transform cleaned text into numerical features with a maximum of 2,000 features.

4. **Model Training**:
   - Split the dataset into training (80%) and testing (20%) sets.
   - Train a Logistic Regression model on the training set.

5. **Evaluation**:
   - Evaluate the model on the test set using accuracy score and classification report.

6. **Save Model**:
   - Save the trained model (`fake_news_model.pkl`) and vectorizer (`tfidf_vectorizer.pkl`) for future use.

---

## **Results**
### Model Performance:
The Logistic Regression model achieved the following results on the test set:

| Metric       | Value |
|--------------|-------|
| Accuracy     | 97.10% |
| Precision    | 97%   |
| Recall       | 97%   |
| F1-score     | 97%   |

### Classification Report:


          precision    recall  f1-score   support

       0       0.97      0.97      0.97       469
       1       0.97      0.97      0.97       429

accuracy                           0.97       898


macro avg 0.97 0.97 0.97 898
weighted avg 0.97 0.97 0.97 898


### Preprocessing Time:
- Total preprocessing time: ~291 seconds (~4 minutes and 51 seconds).

### Key Findings:
- The model performs equally well on both classes (fake and real news), achieving an F1-score of **97%** for both classes.
- The high accuracy indicates that TF-IDF features combined with Logistic Regression are effective for this classification task.

---

## **How to Run**
Follow these steps to run the project locally:

### Prerequisites:
1. Install Python (>=3.7) on your system.
2. Install required libraries using pip:
pip install pandas spacy scikit-learn joblib
python -m spacy download en_core_web_sm


### Instructions:
1. Clone this repository to your local machine:
git clone https://github.com/Jasonpereira0/Projects.git
cd Projects/Fake_News_Detection/

2. Place the dataset files (`True.csv` and `Fake.csv`) in the `data/` directory.

3. Run the script to train the model: python scripts/fake_news_detection.py

4. Check the `output/` directory for saved models:
output/
├── fake_news_model.pkl # Trained Logistic Regression model
└── tfidf_vectorizer.pkl # TF-IDF Vectorizer used for feature extraction

5. Use these models to make predictions on new data.

---

## **Future Work**
This project can be extended in several ways:
1. Implement deep learning models like LSTMs or Transformers (e.g., BERT) for better performance on more complex datasets.
2. Include additional features like metadata (e.g., publication date, subject).
3. Deploy as a web application using Flask or Streamlit for real-time predictions.

---

## **Contributing**
Contributions are welcome! If you have ideas or find issues, feel free to open an issue or submit a pull request.

---

## **License**
This project is licensed under the MIT License—see the LICENSE file for details.

---
