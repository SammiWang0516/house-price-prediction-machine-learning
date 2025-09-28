# House Price Prediction – Machine Learning Solution

## Overview
This project presents a comprehensive machine learning solution for the Kaggle House Prices: Advanced Regression Techniques competition. 
The goal is to predict house sale prices based on property features such as location, size, quality, and condition. 
The solution integrates extensive data preprocessing, exploratory data analysis (EDA), feature engineering, and advanced regression and ensemble models to deliver accurate and interpretable predictions.

## Key Features

- **Robust Data Cleaning & Feature Engineering**: Systematic handling of missing values using strategies such as mean, median, and mode imputation depending on feature type (numeric, categorical, ordinal).
- **Exploratory Data Analysis (EDA)**: Visual and statistical analysis to reveal correlations between predictors (e.g., OverallQual, GrLivArea, YearBuilt) and sale price.
- **Feature Engineering**: Creation and transformation of ordinal features, log-transformations of skewed variables, and scaling/encoding to prepare data for regression and ensemble models.
- **Pipeline & Cross-Validation**: Leveraging `scikit-learn` pipelines and GridSearchCV for clean, repeatable preprocessing and hyperparameter tuning.
- **Advanced Regression Models**: Linear Regression and its regularized variants: Ridge Regression (L2 regularization), Lasso Regression (L1 regularization), Elastic Net (combination of L1 & L2)
- **Ensemble Models**: Tree-based approaches to capture nonlinear interactions and feature importance: Random Forest Regressor, XGBoost Regressor
- **Evaluation Metrics & Visualization**: Thorough model validation with RMSE, MAE, and r2_score

## Technologies & Tools

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn (preprocessing, model selection, pipelines, linear models, cross-validation)
- Statistical libraries (scipy)
- XGBoost (gradient boosting framework)
- Jupyter Notebook (interactive development and visualization)

## How to Use

1. Clone the repository:
git clone https://github.com/SammiWang0516/house-price-prediction.git

2. Install dependencies:
pip install -r requirements.txt

3. Place the training data file (`train.csv`, `test.csv`) inside the `data/` directory.

4. Run the pipeline script to train and tune the Random Forest model:
`scripts/Kaggle_Titanic_Pipeline.py`

5. After successful run, the best model and grid search results will be saved automatically inside the `models/` folder.

6. You can explore, modify, or extend the pipeline in the `Kaggle_Titanic_Pipeline.py` script or examine detailed analysis and further experiments in the Jupyter notebook.
  a. `01_DataCleaning_FeatureEngineering.ipynb`: basic data cleaning procedure with feature engineering. Data visualization and statistical significance for each feature are shown.
  b. `02_LinearRegression_Modeling.ipynb`: importing post-cleaning dataframe, preprocessor using pipeline, and lasso, ridge, and elastic net model deployment. After fitting, cross-validation is carried out, along with metrics and feature importance.
  c. `02_RandomForest_Modeling.ipynb`: importing post-cleaning dataframe, preprocessor using pipeline, and tree-based random forest model deployment. After fitting, cross-validation is carried out, along with metrics and feature importance.
  d. `02_XGBoost_Modeling.ipynb`: importing post-cleaning dataframe, preprocessor using pipeline, and tree-based XGBoostRegressor model deployment. After fitting, cross-validation is carried out, along with metrics and feature importance.
  e. `03_Predict_On_Test.ipynb`: importing best model (in `models/best_model.pkl`) and feature engineering function from scripts (in `scripts/Kaggle_HousePrice_Pipeline.py`). Conduct feature engineering on test data and present prediction on it.

7. To experiment interactively, launch Jupyter Notebook:
`notebooks/02_LinearRegression_Modeling.ipynb`
`notebooks/02_RandomForest_Modeling.ipynb`
`notebooks/02_XGBoost_Modeling.ipynb`

This streamlined workflow ensures easy reproduction of model training, tuning, and evaluation with modular and maintainable code.

## Project Structure

- `notebooks/` — Full modeling workflow: data cleaning, EDA, feature engineering, model training, hyperparameter tuning, and evaluation with different models.
- `Kaggle_HousePrice_Pipeline.py` — Modular pipeline script to preprocess data and train regression/ensemble models with cross-validation and GridSearchCV.
- `data/` — Directory for storing dataset files.
- `models/` — Folder for storing serialized models and GridSearchCV results (e.g., pickle files) generated after training.
- `submissions/` — Kaggle=ready CSV submission files with predicted SalePrice values.

## Results

- Achieved strong predictive performance across multiple models, with Ridge, Lasso, and Elastic Net providing interpretable regression baselines.
- Random Forest and XGBoost improved accuracy by capturing nonlinear feature interactions and complex relationships.
- RMSE scores validated through cross-validation and Kaggle submissions confirmed competitive performance.
- Developed a modular, reproducible pipeline to streamline experimentation and model deployment.
- Delivered interpretable results by analyzing feature importances from both regression coefficients and tree-based models.

---

🚀 **Dive into the notebook and see how each step is crafted with best practices in data science and machine learning!**

For questions or collaborations, feel free to open an issue or contact me directly through GitHub.

---

© 2025 by Sammi Wang | Data Science Enthusiast | Kaggle Competitor 
