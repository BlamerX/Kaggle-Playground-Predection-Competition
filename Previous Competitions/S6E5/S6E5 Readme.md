# Predicting F1 Pit Stops
## Playground Series - Season 6, Episode 5

![Header](https://www.kaggle.com/competitions/125221/images/header)

## Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict whether a Formula 1 driver will pit on the next lap.

## Evaluation

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

### Submission File

For each id in the test set, you must predict a probability for the PitNextLap target. The file should contain a header and have the following format:

```csv
id,PitNextLap
439140,0.2
439141,0.3
439142,0.9
```

## Timeline

- **Start Date** - May 1, 2026
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - May 31, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Dataset Description

The dataset for this competition (both train and test) was inspired by F1 strategy dataset. Feature distributions are close to, but not exactly the same, as the original, and we intentionally remove Normalized_TyreLife which makes the prediction trivial. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Files

- `train.csv` - the training set, with PitNextLap as target
- `test.csv` - the test set, used to predict the likelihood for PitNextLap
- `sample_submission.csv` - a sample submission file in the correct format

---

## Competition Dataset

### Overview

The competition dataset (train.csv and test.csv) is a synthetically generated dataset based on the F1 Strategy Dataset.

### 📘 Dataset Description

This dataset provides a lap-level view of Formula 1 races, designed specifically for race strategy analysis and machine learning applications. It transforms raw telemetry data into a structured format with engineered features that capture tire degradation, race progression, and driver performance dynamics.

### 🔍 Use Cases

- Predicting pit stop decisions
- Modeling race strategies
- Analyzing tire degradation patterns
- Driver performance and consistency analysis
- Time-series and classification tasks

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| id | Integer | Unique identifier |
| Driver | String | Driver name or abbreviation |
| Compound | String | Tire compound type |
| Race | String | Race name/location |
| Year | Integer | Year of the race |
| PitStop | Integer | Pit stop indicator |
| LapNumber | Integer | Current lap number |
| Stint | Integer | Current stint number |
| TyreLife | Float | Number of laps completed on the current tire set |
| Position | Integer | Driver's race position |
| LapTime (s) | Float | Lap time in seconds |
| LapTime_Delta | Float | Delta of lap time compared to previous lap/best lap |
| Cumulative_Degradation | Float | Cumulative tire degradation |
| RaceProgress | Float | Percentage of race completed |
| Position_Change | Float | Change in position |
| PitNextLap | Float | **Target variable**: Predict whether a driver will pit next lap |

---

## Original Dataset: F1 Strategy Dataset | Pit Stop Prediction

### About Dataset

#### Dataset Overview

This dataset provides a lap-level view of Formula 1 races, designed specifically for race strategy analysis and machine learning applications. It transforms raw telemetry data into a structured format with engineered features that capture tire degradation, race progression, and driver performance dynamics.

#### Data Processing

The dataset is built using FastF1 and includes custom feature engineering such as:
- Lap time deltas
- Cumulative degradation
- Normalized tire life
- Race progress metrics

#### Notes

- Data is aggregated from multiple races (multi-race dataset)
- Missing or unreliable entries have been cleaned
- Suitable for both beginners and advanced ML workflows

#### Future Updates

- Additional seasons and races
- Advanced strategy features (undercut/overcut modeling)
- Weather and track condition integration

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| Driver | String | Driver name or abbreviation |
| Compound | String | Tire compound type |
| Race | String | Race name/location |
| Year | Integer | Year of the race |
| PitStop | Integer | Pit stop indicator |
| LapNumber | Integer | Current lap number |
| Stint | Integer | Current stint number |
| TyreLife | Float | Number of laps completed on the current tire set |
| Normalized_TyreLife | Float | Normalized version of tire life |
| Position | Integer | Driver's race position |
| LapTime (s) | Float | Lap time in seconds |
| LapTime_Delta | Float | Delta of lap time compared to previous lap/best lap |
| Cumulative_Degradation | Float | Cumulative tire degradation |
| RaceProgress | Float | Percentage of race completed |
| Position_Change | Float | Change in position |
| PitNextLap | Integer | **Target variable**: Predict whether a driver will pit next lap |

### 📁 Kaggle Data Locations
```python
# Raw data
train = pd.read_csv('/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv')
test = pd.read_csv('/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv')
orig = pd.read_csv('/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv')
```
