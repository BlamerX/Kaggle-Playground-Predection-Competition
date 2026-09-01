# Predicting Heart Disease
## Playground Series - Season 6, Episode 2

![Header](https://www.kaggle.com/competitions/125192/images/header)

## Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict the likelihood of heart disease.

## Evaluation

Submissions are evaluated on **area under the ROC curve** between the predicted probability and the observed target.

### Submission File

For each id in the test set, you must predict a probability for the Heart Disease target. The file should contain a header and have the following format:

```csv
id,Heart Disease
630000,0.2
630001,0.3
630002,0.1
...
```

## Timeline

- **Start Date** - February 1, 2026
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - February 28, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

The dataset for this Playground competition was generated synthetically from a deep learning model. We use synthetic data to create a fun, beginner-friendly "sandbox" where we can provide interesting datasets with named features while keeping the test labels private.

Please Note: While our generation techniques have improved over years, synthetic data is not a perfect replica of the real world. You may encounter "artifacts"—patterns or correlations that do not exist in reality—and slight gaps between the synthetic and original distributions. Our goal is to provide a high-quality learning experience, and we rely on your feedback to help us close the gap between synthetic and real data in future competitions.

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Heart disease prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Files

- `train.csv` - the training set, where Heart Disease is the target
- `test.csv` - the test set
- `sample_submission.csv` - a sample submission file in the correct format

---

## Competition Dataset

### Overview

The competition dataset is a synthetically generated dataset based on the Heart Disease Prediction Dataset. It mimics real-world clinical attributes used to analyze and predict the presence or absence of heart disease.

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| id | Integer | Unique identifier |
| Age | Integer | Age of the patient (in years) |
| Sex | Integer | Gender of the patient (1 = Male, 0 = Female) |
| Chest pain type | Integer | Type of chest pain (1-4) |
| BP | Integer | Resting blood pressure (mm Hg) |
| Cholesterol | Integer | Serum cholesterol level (mg/dL) |
| FBS over 120 | Integer | Fasting blood sugar > 120 mg/dL (1 = True, 0 = False) |
| EKG results | Integer | Resting electrocardiogram results (0, 1, 2) |
| Max HR | Integer | Maximum heart rate achieved |
| Exercise angina | Integer | Exercise-induced angina (1 = Yes, 0 = No) |
| ST depression | Float | ST depression induced by exercise relative to rest |
| Slope of ST | Integer | Slope of the peak exercise ST segment |
| Number of vessels fluro | Integer | Number of major vessels (0–3) colored by fluoroscopy |
| Thallium | Integer | Thallium stress test result (categorical medical indicator) |
| Heart Disease | String | Target variable: Presence / Absence |

---

## Original Dataset: Heart Disease Prediction Dataset

### Overview

This dataset contains real-world clinical attributes used to analyze and predict the presence or absence of heart disease.

- **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease/data)

### Features

| Column Name | Description |
| :--- | :--- |
| **Age** | Age of the patient (in years) |
| **Sex** | Gender of the patient (1 = Male, 0 = Female) |
| **Chest pain type** | Type of chest pain:<br>1 = Typical angina<br>2 = Atypical angina<br>3 = Non-anginal pain<br>4 = Asymptomatic |
| **BP** | Resting blood pressure (mm Hg) |
| **Cholesterol** | Serum cholesterol level (mg/dL) |
| **FBS over 120** | Fasting blood sugar > 120 mg/dL (1 = True, 0 = False) |
| **EKG results** | Resting electrocardiogram results:<br>0 = Normal<br>1 = ST-T wave abnormality<br>2 = Left ventricular hypertrophy |
| **Max HR** | Maximum heart rate achieved |
| **Exercise angina** | Exercise-induced angina (1 = Yes, 0 = No) |
| **ST depression** | ST depression induced by exercise relative to rest |
| **Slope of ST** | Slope of the peak exercise ST segment |
| **Number of vessels fluro** | Number of major vessels (0–3) colored by fluoroscopy |
| **Thallium** | Thallium stress test result (categorical medical indicator) |
| **Heart Disease** | Target variable:<br>Presence = Heart disease detected<br>Absence = No heart disease |

### Encoding Notes in Original

- Categorical variables are numerically encoded for ML compatibility.
- Target column uses text labels (Presence / Absence) for better interpretability.
- Dataset is ready for Logistic Regression, Tree-based models, and Ensembles.


