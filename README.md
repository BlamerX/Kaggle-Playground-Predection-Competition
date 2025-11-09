# Kaggle Playground Prediction Competitions

This repository contains my solutions for various Kaggle Playground Series prediction competitions. Each competition is organized into its own directory, containing the notebooks, scripts, and data related to that specific challenge.

## Competitions

Here's a breakdown of the competitions included in this repository:

### 1. Playground Series - Season 5, Episode 10: Road Accidents

*   **Goal:** Predict the severity of road accidents.
*   **Directory:** `Playground Series - Season 5, Episode 10/`
*   **Description:** This project involves analyzing a dataset of road accidents to build a model that can accurately predict the severity of an accident based on various factors like weather, road conditions, and time of day.
*   **Files of Interest:**
    *   `road-accidents.ipynb`: Jupyter Notebook with the complete analysis, from EDA to model training and submission.
    *   `main_final.py`: Python script for the final pipeline.
    *   `Dataset/`: Contains the training, testing, and sample submission files.

### 2. Playground Series - Season 5, Episode 11: Loan Payback Prediction

*   **Goal:** Predict whether a loan will be paid back or not.
*   **Directory:** `Playground Series - Season 5, Episode 11/`
*   **Description:** This project focuses on building a classification model to predict the probability of a borrower defaulting on a loan. The solution explores different models, including XGBoost and LightGBM, with hyperparameter tuning.
*   **Files of Interest:**
    *   `loan-payback.ipynb`: Initial EDA and modeling.
    *   `Xgboost and LGBMR Hypertuned.ipynb`: Notebook with hyperparameter tuned XGBoost and LGBM models.
    *   `Python/`: Directory containing a structured Python project for the loan payback prediction task.
    *   `Dataset/`: Contains the training, testing, and sample submission files.

## General Approach

For each competition, I follow a general machine learning workflow:

1.  **Problem Definition:** Understanding the competition's objective and evaluation metrics.
2.  **Data Loading and Exploration (EDA):** Analyzing the dataset to understand its structure, identify patterns, and visualize relationships between features.
3.  **Data Preprocessing and Feature Engineering:** Cleaning the data, handling missing values, and creating new features to improve model performance.
4.  **Model Selection and Training:** Experimenting with different machine learning models (e.g., XGBoost, LightGBM, RandomForest) and training them on the prepared data.
5.  **Hyperparameter Tuning:** Optimizing the models' hyperparameters to achieve the best possible performance.
6.  **Submission:** Generating the submission file in the format required by the competition.

## How to Use

To explore a specific competition, navigate to its directory. You can then open the Jupyter Notebooks to see the analysis and code. If the project is structured as a Python package, you can install the dependencies from the `requirements.txt` file and run the main script.

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.