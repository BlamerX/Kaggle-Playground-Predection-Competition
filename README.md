# Kaggle Playground Prediction Competitions

This repository contains my solutions for various Kaggle Playground Series prediction competitions. Each competition is organized into its own directory, containing the notebooks, scripts, and data related to that specific challenge.

## Competitions

Here's a breakdown of the competitions included in this repository:

### 1. Playground Series - Season 5, Episode 10: Road Accidents

<table>
  <tr>
    <td style="width: 40%; vertical-align: top;">
      <img src="https://www.kaggle.com/competitions/91721/images/header" alt="Road Accidents Competition Header" width="100%">
    </td>
    <td style="width: 60%; vertical-align: top; padding-left: 20px;">
      <strong>Goal:</strong> Predict the severity of road accidents.
      <br><br>
      <strong>Directory:</strong> <code>Playground Series - Season 5, Episode 10/</code>
      <br><br>
      <strong>Description:</strong> This project involves analyzing a dataset of road accidents to build a model that can accurately predict the severity of an accident based on various factors like weather, road conditions, and time of day.
      <br><br>
      <strong>Files of Interest:</strong>
      <ul>
        <li><code>road-accidents.ipynb</code>: Jupyter Notebook with the complete analysis, from EDA to model training and submission.</li>
        <li><code>main_final.py</code>: Python script for the final pipeline.</li>
        <li><code>Dataset/</code>: Contains the training, testing, and sample submission files.</li>
      </ul>
    </td>
  </tr>
</table>

### 2. Playground Series - Season 5, Episode 11: Loan Payback Prediction

<table>
  <tr>
    <td style="width: 40%; vertical-align: top;">
      <img src="https://www.kaggle.com/competitions/91722/images/header" alt="Loan Payback Prediction Competition Header" width="100%">
    </td>
    <td style="width: 60%; vertical-align: top; padding-left: 20px;">
      <strong>Goal:</strong> Predict whether a loan will be paid back or not.
      <br><br>
      <strong>Directory:</strong> <code>Playground Series - Season 5, Episode 11/</code>
      <br><br>
      <strong>Description:</strong> This project focuses on building a classification model to predict the probability of a borrower defaulting on a loan. The solution explores different models, including XGBoost and LightGBM, with hyperparameter tuning.
      <br><br>
      <strong>Files of Interest:</strong>
      <ul>
        <li><code>loan-payback.ipynb</code>: Initial EDA and modeling.</li>
        <li><code>Xgboost and LGBMR Hypertuned.ipynb</code>: Notebook with hyperparameter tuned XGBoost and LGBM models.</li>
        <li><code>Python/</code>: Directory containing a structured Python project for the loan payback prediction task.</li>
        <li><code>Dataset/</code>: Contains the training, testing, and sample submission files.</li>
      </ul>
    </td>
  </tr>
</table>

---

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
