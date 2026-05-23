# movie-ratings-prediction
Movie rating prediction with Linear Regression and feature engineering.

# 🎬 Movie Rating Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-red.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project predicts movie ratings (`Vote_Average`) using regression models with advanced **feature engineering** techniques. The goal is to demonstrate a complete machine learning pipeline from raw data preprocessing to model evaluation and comparison.

### Key Features
- ✅ Date extraction (year, month, day of week)
- ✅ Language aggregation (top 10 languages + 'other')
- ✅ Genre multi-label encoding using `MultiLabelBinarizer` (top 15 genres + `has_other_genres` flag)
- ✅ Support for multiple regression algorithms (Linear, Lasso, Ridge, ElasticNet)
- ✅ Clean, modular code structure
- ✅ Production-ready preprocessing pipeline

## 📊 Dataset

**Source:** MyMovieDB dataset from Kaggle

**Original features:**
- Title, Release Date, Overview, Poster URL
- Original Language, Genre, Popularity
- Vote Average (target variable)

**After preprocessing:**
- 20+ numerical and binary features
- No missing values
- Ready for machine learning

## 🎯 Model Performance

| Model | Train R² | Test R² | RMSE |
|-------|----------|---------|------|
| Linear Regression | 0.XX | 0.XX | X.XX |
| Lasso | 0.XX | 0.XX | X.XX |
| Ridge | 0.XX | 0.XX | X.XX |
| ElasticNet | 0.XX | 0.XX | X.XX |

*Results will vary based on your specific data split*

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **pandas** | Data manipulation |
| **numpy** | Numerical operations |
| **scikit-learn** | ML models & preprocessing |
| **matplotlib/seaborn** | Visualization |
| **joblib** | Model persistence |

## 📁 Project Structure

movie-ratings-prediction/
│
├── data/
│   ├── raw/                     # Original mymoviedb.csv
│   └── processed/               # Cleaned dataset
│       └── cleaned_movies.csv
│
├── src/
│   ├── 01_data_preprocessing.py # Feature engineering
│   ├── 02_model_training.py     # Train & evaluate models
│   └── 03_predict.py            # Demo predictions
│
├── models/
│   ├── linear_model.pkl
│   ├── lasso_model.pkl
│   ├── ridge_model.pkl
│   └── scaler.pkl
│
├── results/
│   ├── metrics.csv
│   └── model_comparison.png
│
├── requirements.txt
├── .gitignore
└── README.md

