# Diabetes Prediction Challenge
## Playground Series - Season 5, Episode 12

![Header](https://www.kaggle.com/competitions/91723/images/header)

## Overview

Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict the probability that a patient will be diagnosed with diabetes.

## Evaluation

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Diabetes Health Indicators Dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance. `diagnosed_diabetes` is the target, and for the testing data, you should predict the probability of `diagnosed_diabetes`.

## Files

- `train.csv` - the training set
- `test.csv` - the test set
- `sample_submission.csv` - a sample submission file in the correct format
- `Original Dataset.csv` - the original dataset (Diabetes Health Indicators Dataset)

---

## Competition Dataset

### Overview

The competition dataset (train.csv and test.csv) is a synthetically generated dataset based on the Diabetes Health Indicators Dataset. It contains patient records with various demographic and health-related features used to predict the probability of a diabetes diagnosis.

### Features

| Column | Description |
| :--- | :--- |
| id | Unique identifier for each record |
| age | Age of the patient |
| alcohol_consumption_per_week | Number of drinks consumed per week |
| physical_activity_minutes_per_week | Minutes of physical activity per week |
| diet_score | Score representing diet quality |
| sleep_hours_per_day | Average hours of sleep per day |
| screen_time_hours_per_day | Average hours of screen time per day |
| bmi | Body Mass Index |
| waist_to_hip_ratio | Ratio of waist circumference to hip circumference |
| systolic_bp | Systolic blood pressure |
| diastolic_bp | Diastolic blood pressure |
| heart_rate | Resting heart rate |
| cholesterol_total | Total cholesterol level |
| hdl_cholesterol | HDL cholesterol level |
| ldl_cholesterol | LDL cholesterol level |
| triglycerides | Triglycerides level |
| gender | Gender of the patient |
| ethnicity | Ethnic background of the patient |
| education_level | Highest level of education completed |
| income_level | Income category of the patient |
| smoking_status | Smoking history/status of the patient |
| employment_status | Current employment status |
| family_history_diabetes | Family history of diabetes (0 = No, 1 = Yes) |
| hypertension_history | History of hypertension (0 = No, 1 = Yes) |
| cardiovascular_history | History of cardiovascular disease (0 = No, 1 = Yes) |
| diagnosed_diabetes | Target variable: Diabetes diagnosis (0 = No, 1 = Yes) (Train only) |

---

## Original Dataset: Diabetes Health Indicators Dataset

### Overview

This dataset contains 100,000 patient records designed for diabetes risk prediction, analysis, and machine learning applications. The dataset is clean, preprocessed, and ready for use in classification, regression, feature engineering, statistical analysis, and data visualization.

- **Rows:** 100,000
- **Columns:** 35+
- **File:** `Original Dataset.csv`

### Dataset Description

The dataset includes patient profiles with features based on demographics, lifestyle habits, family history, and clinical measurements that are well-established indicators of diabetes risk. All data is generated using statistical distributions inspired by real-world medical research, ensuring privacy preservation while reflecting realistic health patterns.

### Features

| Column | Type | Description | Values/Range |
| :--- | :--- | :--- | :--- |
| patient_id | Integer | Unique patient identifier | 1–100000 |
| age | Integer | Age of patient in years | 18–90 |
| gender | String | Patient gender | 'Male', 'Female', 'Other' |
| ethnicity | String | Ethnic background | 'White', 'Hispanic', 'Black', 'Asian', 'Other' |
| education_level | String | Highest completed education | 'No formal', 'Highschool', 'Graduate', 'Postgraduate' |
| income_level | String | Income category | 'Low', 'Medium', 'High' |
| employment_status | String | Employment type | 'Employed', 'Unemployed', 'Retired', 'Student' |
| smoking_status | String | Smoking behavior | 'Never', 'Former', 'Current' |
| alcohol_consumption_per_week | Float | Drinks consumed per week | 0–30 |
| physical_activity_minutes_per_week | Integer | Physical activity (weekly minutes) | 0–600 |
| diet_score | Integer | Diet quality (higher = healthier) | 0–10 |
| sleep_hours_per_day | Float | Average daily sleep hours | 3–12 |
| screen_time_hours_per_day | Float | Average daily screen time hours | 0–12 |
| family_history_diabetes | Integer | Family history of diabetes | 0 = No, 1 = Yes |
| hypertension_history | Integer | Hypertension history | 0 = No, 1 = Yes |
| cardiovascular_history | Integer | Cardiovascular history | 0 = No, 1 = Yes |
| bmi | Float | Body Mass Index (kg/m²) | 15–45 |
| waist_to_hip_ratio | Float | Waist-to-hip ratio | 0.7–1.2 |
| systolic_bp | Integer | Systolic blood pressure (mmHg) | 90–180 |
| diastolic_bp | Integer | Diastolic blood pressure (mmHg) | 60–120 |
| heart_rate | Integer | Resting heart rate (bpm) | 50–120 |
| cholesterol_total | Float | Total cholesterol (mg/dL) | 120–300 |
| hdl_cholesterol | Float | HDL cholesterol (mg/dL) | 20–100 |
| ldl_cholesterol | Float | LDL cholesterol (mg/dL) | 50–200 |
| triglycerides | Float | Triglycerides (mg/dL) | 50–500 |
| glucose_fasting | Float | Fasting glucose (mg/dL) | 70–250 |
| glucose_postprandial | Float | Post-meal glucose (mg/dL) | 90–350 |
| insulin_level | Float | Blood insulin level (µU/mL) | 2–50 |
| hba1c | Float | HbA1c (%) | 4–14 |
| diabetes_risk_score | Integer | Risk score (calculated, 0–100) | 0–100 |
| diabetes_stage | String | Stage of diabetes | 'No Diabetes', 'Pre-Diabetes', 'Type 1', 'Type 2', 'Gestational' |
| diagnosed_diabetes | Integer | Target: Diabetes diagnosis | 0 = No, 1 = Yes |