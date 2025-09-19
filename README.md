# cc_approvals
Predicting Credit Card Approvals with Logistic Regression

# Predicting Credit Card Approvals with Logistic Regression

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blueviolet.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A data science project focused on building a tuned Logistic Regression model to automate credit card approval decisions based on applicant data.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source](#data-source)
3. [Methodology](#methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training & Hyperparameter Tuning](#model-training--hyperparameter-tuning)
    - [Model Evaluation](#model-evaluation)
4. [Results & Conclusion](#results--conclusion)


---

## Project Overview

### Problem Statement
In the financial industry, the credit card approval process is a critical task. It involves assessing an applicant's risk profile to decide whether an application should be approved or denied. A manual process can be slow and subjective. An automated system powered by machine learning can provide a faster, more consistent, and data-driven approach.

### Goal
The goal of this project is to develop and tune a Logistic Regression model to accurately predict the outcome of credit card applications. This model aims to:
- Minimize financial risk by correctly identifying high-risk applicants.
- Automate the initial screening process to improve operational efficiency.
- Ensure fair and consistent decision-making.

---

## Data Source
The dataset used for this project is the **Credit Card Approval Prediction** dataset, commonly sourced from the UCI Machine Learning Repository.

- **Source File:** `cc_approvals.data`
- **Description:** This is a confidential dataset where feature names and values have been anonymized for privacy. It contains a mix of continuous and categorical features that are relevant for credit assessment. The target variable indicates whether a credit card application was approved (`+`) or denied (`-`).

---

## Methodology

### Data Preprocessing
To prepare the data for modeling, the following steps were taken based on the script:

- **Handling Missing Values:**
    - The dataset contained `'?'` characters representing missing data, which were first converted to `NaN`.
    - Categorical features with missing values were imputed using the **most frequent value (mode)** of their respective columns.
    - Numerical features with missing values were imputed using the **mean** of their respective columns.

- **Encoding Categorical Variables:**
    - All categorical features were converted into numerical format using **one-hot encoding** via `pandas.get_dummies()`.
    - The `drop_first=True` parameter was used to avoid multicollinearity.

- **Feature Scaling:**
    - The feature set (`X`) was scaled using `StandardScaler` to normalize the range of the data. This is crucial for the performance of Logistic Regression. The scaler was fit on the training data and used to transform both the training and test sets.

---

### Model Training & Hyperparameter Tuning
This project focuses on the **Logistic Regression** model, a robust and interpretable classifier ideal for binary classification tasks.

- **Initial Model:** A baseline Logistic Regression model was first trained to establish initial performance.
- **Hyperparameter Tuning:** To find the optimal parameters for the model, `GridSearchCV` was employed with 5-fold cross-validation. The following hyperparameters were tuned:
    - `tol`: [0.01, 0.001, 0.0001]
    - `max_iter`: [100, 150, 200]

The best-performing model from the grid search was then selected for final evaluation.
Best: 0.818163 using {'max_iter': 100, 'tol': 0.01}
---

### Model Evaluation
The primary metric used to evaluate the final model in the script is 0.79386. The best model from the `GridSearchCV` was evaluated on the unseen test set.

- **Final Test Set Accuracy:**  79.386%

**Further Evaluation:**
While accuracy provides a good overview, a more detailed evaluation is necessary to understand the model's performance in a business context.

- **Confusion Matrix on Test Set:**
Calculate and display the final confusion matrix for the **test set** to show how the model handles false positives and false negatives.

[[ 81  22]
 [ 25 100]]

- **Classification Report:**

              precision    recall  f1-score   support

           0       0.76      0.79      0.78       103
           1       0.82      0.80      0.81       125

    accuracy                           0.79       228
   macro avg       0.79      0.79      0.79       228
weighted avg       0.79      0.79      0.79       228
