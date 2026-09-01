# Predicting Student Test Scores
## Playground Series - Season 6, Episode 1

![Header](https://www.kaggle.com/competitions/playground-series-s6e1/header)

## Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict students' test scores.

## Evaluation

Submissions are evaluated using the **Root Mean Squared Error (RMSE)** between the predicted and the observed target.

### Submission File

For each ID in the test set, you must predict a probability for the exam_score variable. The file should contain a header and have the following format:

```csv
id,exam_score
630000,97.5
630001,89.2
630002,85.5
...
```

## Timeline

- **Start Date** - January 1, 2026
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - January 31, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Exam Score Prediction Dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

### Dataset Verification & Insights
Based on analysis:
- **No Missing Values**: The dataset is clean.
- **Correlations**: `study_hours` (0.76) and `class_attendance` (0.36) strongly correlate with success.
- **Distributions**:
    - `age`: 17-24 (Discrete, small range).
    - `gender`: Balanced (Male, Female, Other).
    - `study_hours`: Continuous (0.1 - 8.0).
    - `exam_score`: Normal-like distribution (20-100).

## Files

- `train.csv` - the training set
- `test.csv` - the test set
- `sample_submission.csv` - a sample submission file in the correct format
- `Exam_Score_Prediction.csv` - the original dataset

---

## Competition Dataset

### Overview

The competition dataset (train.csv and test.csv) contains student records with various demographic and academic features used to predict the student's exam score.

- **Train Set**: 630,000 rows
- **Test Set**: 270,000 rows

### Features

| Column | Type | Values / Range | Description |
| :--- | :--- | :--- | :--- |
| id | Integer | 0 - 629,999 | Unique identifier |
| age | Integer | 17 - 24 | Median: 21. **Binning:** Q1 is 19. Can be binned into "Teen" (17-19) and "Young Adult" (20-24). |
| gender | Categorical | Male, Female, Other | Balanced distribution across 3 categories. |
| course | Categorical | 7 Unique Courses | Includes B.Tech, B.Sc, B.Com, BBA, BCA, BA, Diploma. |
| study_hours | Float | 0.08 - 7.91 | Median: 4.0. **Binning:** Quantiles suggest Low (<2), Medium (2-6), High (>6). |
| class_attendance | Float | 40.6 - 99.4 | Median: 72.6%. **Binning:** Skewed towards higher attendance. |
| internet_access | Binary | Yes, No | Majority have internet access. |
| sleep_hours | Float | 4.1 - 9.9 | Median: 7.1. **Binning:** Normal distribution centered around 7 hours. |
| sleep_quality | Ordinal | Poor, Average, Good | Subjective rating. |
| study_method | Categorical | 5 Unique Methods | Includes Online videos, Coaching, etc. |
| facility_rating | Ordinal | Low, Medium, High | Evenly distributed. |
| exam_difficulty | Ordinal | Easy, Moderate, Hard | Perceived difficulty. |
| exam_score | Continuous | 19.6 - 100.0 | **Target**: Final score. Median 62.6. |

---

## Original Dataset: Exam Score Prediction Dataset

### Overview

This dataset provides the foundational data used for the synthetic generation. It contains 20,000 records.

### Features

| Column | Type | Values / Range |
| :--- | :--- | :--- |
| student_id | Integer | Unique ID |
| age | Integer | 17 - 24 |
| gender | Categorical | Male, Female, Other |
| course | Categorical | 7 unique courses |
| study_hours | Float | 0 - 8 hours |
| class_attendance | Float | 40 - 100% |
| internet_access | Binary | Yes, No |
| sleep_hours | Float | 4 - 10 hours |
| sleep_quality | Ordinal | Poor, Average, Good |
| study_method | Categorical | 5 methods |
| facility_rating | Ordinal | Low, Medium, High |
| exam_difficulty | Ordinal | Easy, Moderate, Hard |
| exam_score | Continuous | 0 - 100 |


## Citation

Yao Yan, Walter Reade, Elizabeth Park. Predicting Student Test Scores. https://kaggle.com/competitions/playground-series-s6e1, 2025. Kaggle.