# Loan Default Prediction: Model Comparison and Evaluation

## Overview

This project aims to predict loan default status using machine learning models, addressing a highly imbalanced dataset where most loans are non-default. The primary objective is to assess the effectiveness of various machine learning algorithms and ensemble techniques in predicting loan default while handling class imbalance effectively.

### Key Features:
- **Dataset**: The dataset contains loan-related information with features like loan amount, annual income, CIBIL score, and more.
- **Target Variable**: `loan_status` (0 = Non-Default, 1 = Default)
- **Goal**: Build a predictive model to determine whether a loan will default based on the available features.

## Project Structure

This repository contains two Jupyter Notebooks and two essential visualizations:

1. **Exploratory Data Analysis (EDA) Notebook(eda.ipynb)**: Contains an in-depth analysis of the dataset, data cleaning, feature engineering, and preliminary insights.
2. **Modeling and Comparison Notebook(ML_Model.ipynb)**: Implements various models, compares their performance, and identifies the best-performing model using ensemble methods.
3. **Visualizations**: 
    - A bar graph comparing the accuracy of each model.
    - A bar graph comparing the ROC AUC score of each model.

## Approach

### 1. **EDA and Data Preprocessing**:
The first notebook focuses on exploring the dataset and performing necessary preprocessing steps:
- Data cleaning, such as handling missing values and converting categorical variables into numerical ones.
- Feature engineering to create derived features like debt-to-income and employment stability ratio.
- Class balancing using the SMOTEENN technique to handle imbalanced classes.

### 2. **Modeling**:
The second notebook involves building and comparing multiple machine learning models, including:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost**
- **LightGBM**
- **Voting Classifier** (Ensemble Model)
- **Stacking Classifier** (Ensemble Model with Logistic Regression as Meta-Model)

Each model's performance is evaluated using accuracy, ROC AUC score, confusion matrix, and classification report. The final goal is to determine the best model based on these metrics.

### 3. **Winner Announcement**:
After training and evaluating the models, the **Voting Classifier** emerged as the best-performing model, achieving the highest accuracy and ROC AUC score. It combined the predictions of individual models (Logistic Regression, Random Forest, XGBoost, and LightGBM) through soft voting, improving the overall performance.

## Visualizations

- **Accuracy Comparison**: A bar graph comparing the accuracy of each model.
- **ROC AUC Score Comparison**: A bar graph comparing the ROC AUC scores of each model.

You can view these visualizations in the repository to understand the comparative performance of all models.

## Installation and Requirements

To run this project, you need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm


Evaluation Metrics
The models were evaluated based on the following metrics:

Accuracy: Measures the overall correctness of the model.
ROC AUC Score: Measures the model's ability to distinguish between the two classes (default and non-default).
Confusion Matrix: Shows the true positives, false positives, true negatives, and false negatives.
Classification Report: Provides precision, recall, and F1-score for both classes.
Results
Model Performance:
Logistic Regression: Achieved an accuracy of 66.02% with an ROC AUC score of 0.71.
Random Forest: Achieved an accuracy of 65.71% with an ROC AUC score of 0.70.
XGBoost: Achieved an accuracy of 73.99% with an ROC AUC score of 0.70.
LightGBM: Achieved an accuracy of 73.98% with an ROC AUC score of 0.71.
Voting Classifier (Ensemble): Achieved the best performance with an accuracy of 74.64% and ROC AUC score of 0.71.
Stacking Classifier: Achieved an accuracy of 67.69% with an ROC AUC score of 0.71.
Winner: Voting Classifier
The Voting Classifier outperformed other models, providing the highest accuracy and balanced performance across the confusion matrix, making it the best choice for this task.

Conclusion
This project demonstrates the power of ensemble methods in improving predictive performance, especially in imbalanced datasets. By leveraging a Voting Classifier, we were able to significantly improve prediction accuracy. Further optimizations and hyperparameter tuning could further enhance performance.

Contact
For any questions or comments, please feel free to contact me:

Email: karanrpjain@gmail.com
GitHub: github.com/KaranJain09
