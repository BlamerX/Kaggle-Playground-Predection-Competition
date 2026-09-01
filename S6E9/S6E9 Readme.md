# Predicting Electric Vehicle Interest
## Playground Series - Season 6, Episode 9

![Header](https://www.kaggle.com/competitions/125219/images/header)

## Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict whether a potential car buyer will purchase an electric vehicle.

## Evaluation

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

### Submission File

For each id in the test set, you must predict a probability for the `Will_Buy_EV` variable. The file should contain a header and have the following format:

```csv
id,Will_Buy_EV
668665,0.2
668666,0.3
668667,0.2
```

## Timeline

- **Start Date** - September 1, 2026
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - September 30, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series six years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Dataset Description

The dataset for this competition (both train and test) was inspired by the EV Adoption and Range Anxiety dataset. Feature distributions are close to, but not exactly the same, as the original. The synthetic data is designed to mimic realistic car-buying behavior and EV adoption signals while keeping the target label hidden in the public test set.

## Files

- `train.csv` - the training set, with `Will_Buy_EV` as target
- `test.csv` - the test set, used to predict the probability for `Will_Buy_EV`
- `sample_submission.csv` - a sample submission file in the correct format
- `EV_Adoption_and_Range_Anxiety_Dataset.csv` - the original real-world inspired data used to generate the synthetic competition data

---

## Competition Dataset

### Overview

The competition dataset (train.csv and test.csv) is a synthetically generated dataset based on EV adoption behavior patterns. It contains demographic, economic, infrastructural, and behavioral variables associated with purchase likelihood for electric vehicles.

### 📘 Dataset Description

This dataset contains realistic buyer information for a synthetic population of potential electric vehicle buyers. It captures factors such as age, income, city type, commute distance, current vehicle type, charging access, environmental concern, and range anxiety. The challenge is to infer the likelihood that each buyer will purchase an EV.

### 🔍 Use Cases

- Predicting EV purchase intent
- Modeling range anxiety and adoption barriers
- Studying the effect of charging accessibility and subsidies
- Analyzing consumer segmentation for green mobility
- Building tabular ML and explainability workflows

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| id | Integer | Unique identifier |
| Age | Integer | Age of the buyer |
| Annual_Income_USD | Float | Annual household income in USD |
| Daily_Commute_km | Float | Average daily commute distance in kilometers |
| Number_of_Cars_Owned | Integer | Number of cars currently owned |
| Charging_Stations_Near_Home | Integer | Number of nearby charging stations close to home |
| Charging_Stations_Near_Work | Integer | Number of nearby charging stations close to work |
| Environmental_Concern_Level | Float | Self-reported environmental concern score |
| Gender | String | Buyer gender |
| City_Type | String | Urban, Suburban, or Rural |
| Current_Car_Type | String | Type of current vehicle |
| Home_Charging_Possible | String | Whether home charging is possible (Yes/No) |
| Subsidy_Available | String | Whether a subsidy or incentive is available (Yes/No) |
| Range_Anxiety_Level | String | Self-reported range anxiety level (Low/Medium/High) |
| Will_Buy_EV | Integer | **Target variable**: Whether the buyer will purchase an EV |

---

## Original Dataset: EV Adoption & Range Anxiety Dataset

### About Dataset

#### Dataset Overview

This dataset provides a synthetic snapshot of 10,000 potential car buyers and is designed to help data scientists analyze the demographic, geographic, and psychological factors that influence EV adoption and range anxiety.

#### Data Processing

The dataset includes realistic buyer distributions and intentionally introduces a small percentage of missing values in certain columns such as income and commute distance to allow for data cleaning and imputation practice.

#### Notes

- Synthetic but inspired by real EV adoption behavior patterns
- Includes demographic, infrastructure, and psychological signals
- Optimized for tabular classification and EDA exercises
- Suitable for beginners and advanced ML workflows

#### Potential Use Cases

- Purchase prediction (binary classification)
- Range anxiety analysis (multi-class classification)
- Product and policy analysis for EV adoption strategies
- Exploratory data analysis on urban vs rural charging access

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| Buyer_ID | String | Unique buyer identifier |
| Age | Integer | Buyer age |
| Gender | String | Buyer gender |
| Annual_Income_USD | Float | Annual income in USD |
| City_Type | String | Urban, Suburban, or Rural |
| Daily_Commute_km | Float | Daily commute distance |
| Number_of_Cars_Owned | Integer | Number of cars owned |
| Current_Car_Type | String | Type of current vehicle |
| Charging_Stations_Near_Home | Integer | Charging stations nearby at home |
| Charging_Stations_Near_Work | Integer | Charging stations nearby at work |
| Home_Charging_Possible | String | Whether home charging is available |
| Environmental_Concern_Level | Float | Environmental concern score |
| Subsidy_Available | String | Whether subsidy is available |
| Range_Anxiety_Level | String | Range anxiety level (Low/Medium/High) |
| Will_Buy_EV | String | Binary target: whether the buyer will buy an EV |

