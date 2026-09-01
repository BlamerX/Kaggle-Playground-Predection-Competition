# Predicting Irrigation Need
## Playground Series - Season 6, Episode 4

![Header](https://www.kaggle.com/competitions/125220/images/header)

## Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict the irrigation need.

## Evaluation

Submissions are evaluated on **balanced accuracy** between the predicted class and observed target.

### Submission File

For each id in the test set, you must predict a class label (Low, Medium, High) for the Irrigation_Need target. The file should contain a header and have the following format:

```csv
id,Irrigation_Need
630000,Low
630001,High
630002,Low
```

## Timeline

- **Start Date** - April 1, 2026
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - April 30, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Irrigation Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Files

- `train.csv` - the training set, with Irrigation_Need as target
- `test.csv` - the test set, used to predict the category for Irrigation_Need
- `sample_submission.csv` - a sample submission file in the correct format

---

## Competition Dataset

### Overview

The competition dataset (train.csv and test.csv) is a synthetically generated dataset based on the Irrigation Prediction dataset.

### 📘 Dataset Description

The Smart Irrigation Water Requirement Prediction Dataset aims to optimize water usage in agriculture by predicting the amount of irrigation required under varying environmental conditions. Efficient water management is critical for sustainable farming, especially in water-scarce regions.

The dataset includes crop type, soil type, temperature, rainfall, and evapotranspiration parameters, enabling accurate estimation of irrigation needs using machine learning models. It supports regression tasks and smart irrigation system development.

### 🔍 Use Cases

- Smart irrigation systems
- Water resource optimization
- Climate-aware agriculture solutions
- Machine learning regression projects

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| id | Integer | Unique identifier |
| Soil_Type | String | Type of soil (Loamy, Clay, Sandy, Silt) |
| Soil_pH | Float | Ph level of the soil |
| Soil_Moisture | Float | Moisture content in the soil |
| Organic_Carbon | Float | Organic carbon content |
| Electrical_Conductivity | Float | Soil electrical conductivity |
| Temperature_C | Float | Ambient temperature in Celsius |
| Humidity | Float | Relative humidity percentage |
| Rainfall_mm | Float | Precipitation in millimeters |
| Sunlight_Hours | Float | Daily sunlight duration |
| Wind_Speed_kmh | Float | Wind speed in km/h |
| Crop_Type | String | Type of crop (Sugarcane, Wheat, Rice, Potato, Cotton, Maize) |
| Crop_Growth_Stage | String | Growth phase (Sowing, Vegetative, Flowering, Harvest) |
| Season | String | Agricultural season (Zaid, Kharif, Rabi) |
| Irrigation_Type | String | Method of irrigation (Drip, Rainfed, Sprinkler, Canal) |
| Water_Source | String | Source of water (Rainwater, River, Reservoir, Groundwater) |
| Field_Area_hectare | Float | Size of the field in hectares |
| Mulching_Used | String | Whether mulching was applied (Yes, No) |
| Previous_Irrigation_mm | Float | Amount of previous irrigation |
| Region | String | Geographical region (East, South, North, West, Central) |
| Irrigation_Need | String | **Target variable**: Irrigation requirement level (Low, Medium, High) |

---

## Original Dataset: Irrigation Prediction Dataset

### About Dataset

#### Dataset Overview

The Smart Irrigation Water Requirement Prediction Dataset aims to optimize water usage in agriculture by predicting the amount of irrigation required under varying environmental conditions. Efficient water management is critical for sustainable farming, especially in water-scarce regions.

The dataset includes crop type, soil type, temperature, rainfall, and evapotranspiration parameters, enabling accurate estimation of irrigation needs using machine learning models.

#### Purpose of the Dataset

The primary purpose of this dataset is to:

- Understand irrigation requirements under different environmental conditions
- Analyze the impact of crop type, soil type, and weather on water needs
- Practice data cleaning, preprocessing, exploratory data analysis (EDA), and predictive modeling
- Build and evaluate machine learning models for classification problems

It is suitable for beginners to intermediate learners in data science and analytics.

#### What Can Be Analyzed

Through using this dataset, you can analyze:

- Irrigation need patterns across different crop and soil types
- Impact of temperature, rainfall, and evapotranspiration on water requirements
- Relationships between environmental factors and irrigation demand
- Climate-aware strategies for water resource optimization
- Smart irrigation system performance under varied conditions

#### Data Cleaning & Preprocessing Applications

Before modeling, the following preprocessing steps can be applied:

- Handling categorical variables into numerical format using Label Encoding or One-Hot Encoding
- Scaling numerical features (StandardScaler / MinMaxScaler)
- Feature engineering (e.g., interaction between rainfall and soil type)
- Removing or analyzing class imbalance
- Dropping non-informative identifiers

#### Acknowledgment

This dataset is a publicly available dataset provided for educational and demonstration purposes.

### Features

| Column | Type | Description |
| :--- | :--- | :--- |
| Soil_Type | String | Type of soil (Loamy, Clay, Sandy, Silt) |
| Soil_pH | Float | Ph level of the soil |
| Soil_Moisture | Float | Moisture content in the soil |
| Organic_Carbon | Float | Organic carbon content |
| Electrical_Conductivity | Float | Soil electrical conductivity |
| Temperature_C | Float | Ambient temperature in Celsius |
| Humidity | Float | Relative humidity percentage |
| Rainfall_mm | Float | Precipitation in millimeters |
| Sunlight_Hours | Float | Daily sunlight duration |
| Wind_Speed_kmh | Float | Wind speed in km/h |
| Crop_Type | String | Type of crop (Sugarcane, Wheat, Rice, Potato, Cotton, Maize) |
| Crop_Growth_Stage | String | Growth phase (Sowing, Vegetative, Flowering, Harvest) |
| Season | String | Agricultural season (Zaid, Kharif, Rabi) |
| Irrigation_Type | String | Method of irrigation (Drip, Rainfed, Sprinkler, Canal) |
| Water_Source | String | Source of water (Rainwater, River, Reservoir, Groundwater) |
| Field_Area_hectare | Float | Size of the field in hectares |
| Mulching_Used | String | Whether mulching was applied (Yes, No) |
| Previous_Irrigation_mm | Float | Amount of previous irrigation |
| Region | String | Geographical region (East, South, North, West, Central) |
| Irrigation_Need | String | Target variable: Irrigation requirement level (Low, Medium, High) |

