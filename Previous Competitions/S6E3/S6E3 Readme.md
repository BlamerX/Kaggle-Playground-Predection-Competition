# Predicting Customer Churn
## Playground Series - Season 6, Episode 3

![Header](https://www.kaggle.com/competitions/125197/images/header)

## Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict the likelihood of customer churn.

## Evaluation

Submissions are evaluated on **area under the ROC curve** between the predicted probability and the observed target.

### Submission File

For each id in the test set, you must predict a probability for the Churn variable. The file should contain a header and have the following format:

```csv
id,Churn
594194,0.1
594195,0.3
594196,0.2
```

## Timeline

- **Start Date** - March 1, 2026
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - March 31, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the customer churn prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Files

- `train.csv` - the training set, where Churn is the target
- `test.csv` - the test set
- `sample_submission.csv` - a sample submission file in the correct format

---

## Competition Dataset

### Overview

The competition dataset (train.csv and test.csv) is a synthetically generated dataset based on the Telco Customer Churn dataset.

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| id | Integer | Unique identifier |
| gender | String | Gender of the customer (Male, Female) |
| SeniorCitizen | Integer | Whether the customer is a senior citizen or not (1, 0) |
| Partner | String | Whether the customer has a partner or not (Yes, No) |
| Dependents | String | Whether the customer has dependents or not (Yes, No) |
| tenure | Integer | Number of months the customer has stayed with the company |
| PhoneService | String | Whether the customer has a phone service or not (Yes, No) |
| MultipleLines | String | Whether the customer has multiple lines or not |
| InternetService | String | Customer’s internet service provider (DSL, Fiber optic, No) |
| OnlineSecurity | String | Whether the customer has online security or not |
| OnlineBackup | String | Whether the customer has online backup or not |
| DeviceProtection | String | Whether the customer has device protection or not |
| TechSupport | String | Whether the customer has tech support or not |
| StreamingTV | String | Whether the customer has streaming TV or not |
| StreamingMovies | String | Whether the customer has streaming movies or not |
| Contract | String | The contract term of the customer (Month-to-month, One year, Two year) |
| PaperlessBilling | String | Whether the customer has paperless billing or not (Yes, No) |
| PaymentMethod | String | The customer’s payment method |
| MonthlyCharges | Float | The amount charged to the customer monthly |
| TotalCharges | Float | The total amount charged to the customer |
| Churn | String | Target variable: Whether the customer churned or not (Yes, No) |

---

## Original Dataset: Telco Customer Churn Dataset

### About Dataset
#### Dataset Overview

This dataset represents customer-level information from a fictional telecommunication centre and is provided publicly by IBM Analytics as a sample dataset. It contains demographic details, subscribed services, account information, and billing data for customers, along with a churn indicator that shows whether a person left the company.

This dataset is widely use for practice of Data Analytics and Machine Learning, especially for Customer Churn analysis.

#### Purpose of the Dataset

The primary purpose of this dataset is to:

- Understand customer behavior in the telecom industry
- Analyze factors contributing to customer churn
- Practice data cleaning, preprocessing, exploratory data analysis (EDA), and predictive modeling
- Build and evaluate machine learning models for classification problems

It is suitable for beginners to intermediate learners in data science and analytics.

#### What Can Be Analyzed

Through using this dataset, you can analyze:

- Churn vs non-churn customer patterns
- Impact of contract type, tenure, and pricing on churn
- Effect of internet services, payment methods, and add-on services
- Customer lifetime behavior based on tenure and monthly charges
- Relationships between demographic features and churn probability

#### Data Cleaning & Preprocessing Applications

Before modeling, the following preprocessing steps can be applied:

- Handling missing or invalid values (e.g., TotalCharges)
- Converting categorical variables into numerical format using Label Encoding, One-Hot Encoding
- Scaling numerical features (StandardScaler / MinMaxScaler)
- Feature engineering (e.g., average charge per month, tenure groups)
- Removing or analyzing class imbalance
- Dropping non-informative identifiers (e.g., customerID)

#### Acknowledgment

This dataset is a publicly available sample dataset provided by IBM Analytics for educational and demonstration purposes. All credit for the original data structure and concept goes to IBM.

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| customerID | String | Unique customer identifier |
| gender | String | Gender of the customer (Male, Female) |
| SeniorCitizen | Integer | Whether the customer is a senior citizen or not (1, 0) |
| Partner | String | Whether the customer has a partner or not (Yes, No) |
| Dependents | String | Whether the customer has dependents or not (Yes, No) |
| tenure | Integer | Number of months the customer has stayed with the company |
| PhoneService | String | Whether the customer has a phone service or not (Yes, No) |
| MultipleLines | String | Whether the customer has multiple lines or not |
| InternetService | String | Customer’s internet service provider (DSL, Fiber optic, No) |
| OnlineSecurity | String | Whether the customer has online security or not |
| OnlineBackup | String | Whether the customer has online backup or not |
| DeviceProtection | String | Whether the customer has device protection or not |
| TechSupport | String | Whether the customer has tech support or not |
| StreamingTV | String | Whether the customer has streaming TV or not |
| StreamingMovies | String | Whether the customer has streaming movies or not |
| Contract | String | The contract term of the customer (Month-to-month, One year, Two year) |
| PaperlessBilling | String | Whether the customer has paperless billing or not (Yes, No) |
| PaymentMethod | String | The customer’s payment method |
| MonthlyCharges | Float | The amount charged to the customer monthly |
| TotalCharges | Float | The total amount charged to the customer |
| Churn | String | Target variable: Whether the customer churned or not (Yes, No) |

