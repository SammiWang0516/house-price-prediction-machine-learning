# import basic libraries
import pandas as pd
import numpy as np

# file location
import os

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# input missing value
from sklearn.impute import SimpleImputer

# data preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder

# data splitting
from sklearn.model_selection import train_test_split

# for evaluation of statistical significance of categorical vs. categorical features
from scipy.stats import chi2_contingency

# for evaluation of statistical significance of categorical vs numeric features
from scipy.stats import f_oneway

# evaluating skewness of numeric feature
from scipy.stats import skew

# pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# gridsearch
from sklearn.model_selection import GridSearchCV

# modeling
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor

# model evaluation (metrics)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# for saving model as pickle file
import joblib

# loading data to Pandas DataFrame
def load_data(data_path):

    df = pd.read_csv(data_path)
    return df

# feature engineering
def feature_engineering(df):

    # domain-based replacement - Alley Feature
    df['Alley'] = df['Alley'].replace(np.nan, 'NoAlley')

    # domain-based replacement - FirePlace Feature
    df['FireplaceQu'] = df['FireplaceQu'].replace(np.nan, 'NoFirePlace')

    # domain-based replacement - Garage Feature
    df['GarageType'] = df['GarageType'].replace(np.nan, 'NoGarage')
    df['GarageFinish'] = df['GarageFinish'].replace(np.nan, 'NoGarage')
    df['GarageQual'] = df['GarageQual'].replace(np.nan, 'NoGarage')
    df['GarageCond'] = df['GarageCond'].replace(np.nan, 'NoGarage')

    # domain-based replacement - Pool Feature
    df['PoolQC'] = df['PoolQC'].replace(np.nan, 'NoPool')

    # domain-based replacement - Fence Feature
    df['Fence'] = df['Fence'].replace(np.nan, 'NoFence')

    # domain-based replacement - Feature
    df['MiscFeature'] = df['MiscFeature'].replace(np.nan, 'NoFeature')

    # domain-based fill missing value - GarageYrBlt
    # if there is no garage, fill the nan with the year when the house was built
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    # Style_Type: Combining BldgType & HouseStyle
    # one is type of dwelling and the other is style
    # each show little statistic significance, and combining p_value is greater
    df['Style_Type'] = df['BldgType'] + '_' + df['HouseStyle']

    # Roof_Style_Mat: Combining RoofStyle & RoofMatl
    # they are all about roof, thus combining together
    # which include not only style but material used
    df['Roof_Style_Mat'] = df['RoofStyle'] + '_' + df['RoofMatl']

    # is_MasVnr: check if there is any masonry veneer area in the house
    # MasVnrArea has lots of 0 value (meaning no masonry veneer)
    df['is_MasVnr'] = (df['MasVnrArea'] != 0).astype(int)

    # Year_Month_Sold: Combining YrSold + MoSold
    # since both MoSold and YrSold do not show much significance
    df['Year_Month_Sold'] = df['YrSold'].astype(str) + '-' + df['MoSold'].astype(str)

    # Season_Sold: Dividing MoSold into 4 seasons
    # ames, Iowa four seasons
    winter = [12, 1, 2]
    spring = [3, 4, 5]
    summer = [6, 7, 8]
    fall = [9, 10, 11]
    def season(row):
        if row['MoSold'] in winter:
            return 'winter'
        elif row['MoSold'] in spring:
            return 'spring'
        elif row['MoSold'] in summer:
            return 'summer'
        else:
            return 'fall'
    df['Season_Sold'] = df.apply(season, axis = 1)

    # TotalFlrSF: Combining 1stFlrSF & 2ndFlrSF
    df['TotalFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']

    # Total_Porch_Area: Combining OpenPorchSF & EnclosedPorch &
    # 3SsnPorch & ScreenPorch
    df['Total_Porch_Area'] = df['OpenPorchSF'] + df['EnclosedPorch'] + \
                             df['3SsnPorch'] + df['ScreenPorch']
    
    # Year_Avg: Combining YearBuilt & YearRemodAdd
    # take the average of those 2
    df['Year_Avg'] = (df['YearBuilt'] + df['YearRemodAdd']) / 2

    # ExteriorSame: Checking if there is only one exterior material
    df['ExteriorSame'] = (df['Exterior1st'] == df['Exterior2nd']).astype(int)

    # is_LowQualFinSF: check if there is any low quality area in the house
    df['is_LowQualFinSF'] = (df['LowQualFinSF'] > 0).astype(int)

    # Has_Open_Porch: check if there is any open porch or not
    df['Has_Open_Porch'] = (df['OpenPorchSF'] > 0).astype(int)

    # Has_Wooden_Deck: check if there is any wooden deck
    df['Has_Wooden_Deck'] = (df['WoodDeckSF'] > 0).astype(int)

    # Has_Enclose_Porch: check if there is any enclose porch
    df['Has_Enclose_Porch'] = (df['EnclosedPorch'] > 0).astype(int)

    # Has_3Sn_Porch: check if there is any 3 seasons porch
    df['Has_3Sn_Porch'] = (df['3SsnPorch'] > 0).astype(int)

    # Has_Screen_Porch: check if there is any screen porch
    df['Has_Screen_Porch'] = (df['ScreenPorch'] > 0).astype(int)

    # Has_Pool: check if there is any pool
    df['Has_Pool'] = (df['PoolArea'] > 0).astype(int)

    return df

# pipeline include preprocessing and model initiation
# OneHotEncoder, SimpleImputer for categorical features
# PowerTransformer, StandardScaler, SimpleImputer for numeric features
def build_preprocessor():

    onehot_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 
                       'LandSlope', 'Neighborhood', 'Condition1', 'Style_Type', 'Roof_Style_Mat', 'Exterior1st', 
                       'Exterior2nd', 'is_MasVnr', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 
                       'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                       'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 
                       'YrSold', 'Year_Month_Sold', 'Season_Sold', 'SaleType', 'SaleCondition', 'BsmtQual',
                       'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'Electrical']
    
    impute_power_features = ['LotFrontage', 'MasVnrArea', 'LotArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 
                             '1stFlrSF', '2ndFlrSF', 'TotalFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                             'OpenPorchSF', 'Total_Porch_Area', 'Year_Avg', 'GarageYrBlt']

    ordinal_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
                        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

    binary_features = ['ExteriorSame', 'is_LowQualFinSF', 'Has_Open_Porch', 'Has_Wooden_Deck', 'Has_Enclose_Porch', 'Has_3Sn_Porch',
                       'Has_Screen_Porch', 'Has_Pool']
    
    preprocessor = ColumnTransformer (
        transformers = [
            ('impute_onehot', Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
            ]), onehot_features),

            ('impute_power', Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'median')),
                ('power_scaler', PowerTransformer(method = 'yeo-johnson', standardize = True))
            ]), impute_power_features),

            ('ordinal', SimpleImputer(strategy = 'median'), ordinal_features),

            ('binary', SimpleImputer(strategy = 'most_frequent'), binary_features)
        ],
        remainder = 'drop'
    )

    # returl model after preprocessor
    return preprocessor

def build_models(preprocessor):

    lasso_linear_model = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('lasso', Lasso())
    ])

    ridge_linear_model = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('ridge', Ridge())
    ])

    elastic_net_linear_model = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('elastic_net', ElasticNet())
    ])

    return {'lasso': lasso_linear_model, 
            'ridge': ridge_linear_model,
            'elastic_net': elastic_net_linear_model}

def build_param_grid():

    # define the parameter grid for Lasso Regression Model
    lasso_param_grid = [
    {'lasso__alpha': [0.01, 0.1, 1, 10, 100],
     'lasso__max_iter': [1000, 5000, 10000]}
    ]

    # define the parameter grid for Ridge Regression Model
    ridge_param_grid = [
    {'ridge__alpha': [0.01, 0.1, 1, 10, 100],
     'ridge__max_iter': [1000, 5000, 10000]}
    ]

    # define the parameter grid for Elastic Net Regression Model
    elastic_param_grid = [
    {'elastic_net__alpha': [0.01, 0.1, 1, 10, 100],
     'elastic_net__l1_ratio': [0.1, 0.5, 0.9],
     'elastic_net__max_iter': [1000, 5000, 10000]}
    ]

    # return grid search model
    return {'lasso': lasso_param_grid,
            'ridge': ridge_param_grid,
            'elastic_net': elastic_param_grid}

def grid_search(model, param_grid, cv, x_train, y_train):

    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        cv = cv,
        scoring = 'neg_root_mean_squared_error',
        n_jobs = -1,
        verbose = 1
    )

    grid_search.fit(x_train, y_train)

    # return a grid search model which has already trained
    return grid_search

def best_model(grid):

    return grid.best_estimator_

def evaluation(bestModel, x_val, y_val):

    y_pred_log = bestModel.predict(x_val)
    
    # log mean squared error and log RMSE
    log_mse = mean_squared_error(y_val, y_pred_log)
    log_rmse = np.sqrt(log_mse)

    # raw mean absolute error
    mae = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred_log))
    log_mae = mean_absolute_error(y_val, y_pred_log)

    # r2 score: showing what percentage of variance is explained
    r2 = r2_score(np.expm1(y_val), np.expm1(y_pred_log))
    log_r2 = r2_score(y_val, y_pred_log)

    print(f'log_mse: {round(log_mse, 4)}')
    print(f'log_rmse: {round(log_rmse, 4)}')
    print(f'mae: {round(mae, 4)}')
    print(f'log_mae {round(log_mae, 4)}')
    print(f'r2: {round(r2, 4)}')
    print(f'log_r2: {round(log_r2, 4)}\n')

    return log_rmse

def save_model(bestModel, location):

    joblib.dump(bestModel, location)
    print('Model Saved!')

def main():

    # read train.csv file
    current_script = os.path.dirname(os.path.abspath(__file__))
    train_file_path = os.path.join(current_script, '..', 'data', 'train.csv')
    df = load_data(train_file_path)

    # make a copy of the raw data
    df_clean = df.copy()

    # feature engineering (no statistic involved)
    df_clean = feature_engineering(df_clean)

    # target: SalePrice
    y = df_clean['SalePrice']
    x = df_clean.drop(columns = ['SalePrice'])
    # SalePrice is extremely skewed. Take log of y
    log_y = np.log1p(y)
    # data splitting
    x_train, x_val, y_train, y_val = train_test_split(x, log_y, test_size = 0.2, random_state = 42)

    # preprocessor with pipeline
    preprocessor = build_preprocessor()

    # modeling
    lasso_model = build_models(preprocessor)['lasso']
    ridge_model = build_models(preprocessor)['ridge']
    elastic_net_model = build_models(preprocessor)['elastic_net']

    # param_grid
    lasso_param_grid = build_param_grid()['lasso']
    ridge_param_grid = build_param_grid()['ridge']
    elastic_param_grid = build_param_grid()['elastic_net']

    # GridSearchCV
    lasso_grid_search = grid_search(lasso_model, lasso_param_grid, 5, x_train, y_train)
    ridge_grid_search = grid_search(ridge_model, ridge_param_grid, 5, x_train, y_train)
    elastic_net_grid_search = grid_search(elastic_net_model, elastic_param_grid, 5, x_train, y_train)

    # best model
    lasso_best_model = best_model(lasso_grid_search)
    ridge_best_model = best_model(ridge_grid_search)
    elastic_net_best_model = best_model(elastic_net_grid_search)

    # evaluation of lasso model
    print('lasso model\n')
    lasso_log_rmse = evaluation(lasso_best_model, x_val, y_val)

    # evaluation of ridge model
    print('ridge model\n')
    ridge_log_rmse = evaluation(ridge_best_model, x_val, y_val)

    # evaluation of elastic net model
    print('elastic net model\n')
    elastic_net_log_rmse = evaluation(elastic_net_best_model, x_val, y_val)

    log_rmse = {lasso_best_model: lasso_log_rmse, ridge_best_model: ridge_log_rmse, elastic_net_best_model: elastic_net_log_rmse}

    model_with_smallest_log_rmse = min(log_rmse, key = log_rmse.get)

    # save the model in pkl file
    model_path = os.path.join(current_script, '..', 'models', 'best_model.pkl')
    save_model(model_with_smallest_log_rmse, model_path)

if __name__ == '__main__':
    main()