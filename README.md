# 🧠 Twitter Sentiment Analysis

A Natural Language Processing (NLP) project for analyzing and classifying tweet sentiments using traditional machine learning models — **Logistic Regression**, **Naive Bayes**, **LightGBM**, and **XGBoost**.  
The project combines linguistic feature engineering, lexicon-based scores, and modern ML techniques to improve text sentiment prediction.

---

## 📘 Overview
This project focuses on classifying Twitter posts into positive and negative sentiments through:
- Advanced **text cleaning and preprocessing**
- **Feature engineering** (Afinn, POS, bigrams, LDA topics)
- **Exploratory Data Analysis (EDA)** to discover patterns
- **Model training and comparison** using multiple classifiers

---

## 🧩 Key Objectives
- Clean and preprocess noisy real-world Twitter data  
- Analyze sentiment patterns via visual EDA  
- Engineer features capturing both syntax and semantics  
- Evaluate and compare ML models for optimal performance  

---

## 🧹 Data Preprocessing
Standard NLP cleaning pipeline applied:
- Removed duplicates and missing values  
- Lowercased all text for normalization  
- Removed URLs, user mentions, and special characters  
- Handled negations (`not_good → not_good`)  
- Tokenized and lemmatized words (WordNetLemmatizer)  
- Removed English stopwords (NLTK corpus)  
- Added **Part-of-Speech (POS)** tags using spaCy  

---

## 📊 Exploratory Data Analysis (EDA)
Key insights derived from visual analysis:
- **Correlation heatmap** — checked feature interdependence  
- **Afinn sentiment score distribution**  
- **Tweet length vs sentiment** relation  
- **Top 20 bigrams** for positive & negative tweets  
- **VADER sentiment distributions** for overall tone balance  

<p align="center">
  <img src="images/eda_heatmap.png" width="400"/>
  <img src="images/bigrams.png" width="400"/>
</p>

---

## 🧠 Feature Engineering

| Feature | Type | Description |
|----------|------|-------------|
| `afinn_score` | Numeric | Sentiment score using Afinn lexicon |
| `positive_words_count` | Numeric | Count of predefined positive words |
| `negative_words_count` | Numeric | Count of predefined negative words |
| `punctuation_marks_count` | Numeric | Count of "!" and "?" |
| `likes_count`, `retweets_count`, `replies_count` | Numeric | Counts of interaction-related words |
| `pos_tags` | Numeric | Frequency of POS tags (NN, JJ, VB, etc.) |
| `mentions_count` | Numeric | Count of “@” mentions |
| `Neg`, `Pos`, `Neu`, `Compound` | Normalized | Sentiment scores from VADER |
| `tweet_length` | Numeric | Cleaned text length |
| `TF-IDF Features` | Sparse | Weighted word representation |
| `Topic_0`–`Topic_9` | Numeric | LDA topic distributions |

---

## ⚙️ Models Used
- **LightGBM:** Fast, scalable gradient boosting with leaf-wise tree growth.  
- **Logistic Regression:** Baseline probabilistic classifier for binary outcomes.  
- **Naive Bayes:** Simple yet effective probabilistic model for text classification.  
- **XGBoost:** High-performance gradient boosting with regularization and feature importance.

---

## 📈 Model Evaluation
Evaluation was done via **random train-test split** and **classification metrics**:
| Model | Accuracy | F1 Score |
|--------|-----------|----------|
| Logistic Regression | 0.749 | 0.787 |
| Naive Bayes | 0.737 | 0.778 |
| LightGBM | 0.78+ | 0.79+ |

Metrics:
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)  
- **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix** used for deeper evaluation.

---

## 📦 Installation
Install all dependencies:

```bash
pip install numpy pandas nltk scikit-learn matplotlib seaborn joblib spacy afinn wordcloud xgboost lightgbm
python -m spacy download en_core_web_sm

## Files
HW2.ipynb                  → Jupyter Notebook with code
HW2.html                   → Rendered notebook for quick preview
Intro to Data Science HW2 REPORT.pdf → Full project report
Readme.txt                 → Package installation list

## 💡 Insights and Learning

Combining lexicon-based and statistical features improved results.

LightGBM achieved stable performance with large feature sets.

Afinn & POS tags provided valuable interpretability to model decisions.

Manual feature engineering outperformed raw TF-IDF baseline.
