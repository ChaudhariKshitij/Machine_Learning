# Big Mart Sales Prediction 🛒📈

## Overview

This project involves predicting the sales of products at different outlets for BigMart. The dataset includes 2013 sales data for 1559 products across 10 stores in different cities. The primary goal is to build predictive models using ensemble techniques like XGBoost and Stacking to forecast sales accurately.

## Dataset

- **Training Data:** `train_v9rqX0R.csv`
- **Test Data:** `test_AbJTz2l.csv`

# Sales Prediction Script Description 🚀

The provided script (`sales_prediction_script.py`) performs the following tasks:

## Data Loading:

- Reads training and test datasets using Pandas.

## Data Cleaning:

- Handles missing values in the 'Item_Weight' column.
- Standardizes 'Item_Fat_Content' values for consistency.

## Feature Engineering:

- Concatenates dataframes and calculates mean item weights.
- Fills missing 'Item_Weight' values based on the calculated means.

## Outlet Information:

- Extracts unique outlet information and fills missing 'Outlet_Size' values.

## Encoding Categorical Values:

- Utilizes OneHotEncoder for categorical encoding.

## XGBoost Model:

- Applies XGBoost with hyperparameter tuning using GridSearchCV.
- Outputs predictions to 'Sales_xgb.csv'.

## XGBoost with RandomForest Model:

- Implements XGBoost with RandomForestRegressor.
- Outputs predictions to 'Sales_xgbForest.csv'.

## Stacking Ensemble:

- Employs StackingRegressor with base models (Linear Regression, ElasticNet, DecisionTree).
- Outputs predictions to 'Sales_stack.csv'.

## Results 📊

### XGBoost Model

- Best Parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}
- Best Score: 0.6008544653330191

### XGBoost with RandomForest Model

- Best Parameters: {'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 25}
- Best Score: 0.3020503893501891

### Stacking Ensemble

- Best Parameters: {'ELA__alpha': 1, 'ELA__l1_ratio': 0.001, 'TREE__max_depth': None, 'final_estimator__max_features': 5, 'passthrough': True}
- Best Score: 0.5803439126292098

## Submission on Analytics Vidhya's DataHack 🏆

This project was submitted on Analytics Vidhya's DataHack platform for the Sales Prediction for Big Mart Outlets challenge.

## Future Work 🔮

- Experiment with additional models and feature engineering techniques.
- Explore ensemble methods for further performance improvements.

## Acknowledgments 🙌

The script uses popular libraries such as Pandas, NumPy, XGBoost, and scikit-learn for data manipulation, modeling, and evaluation.

