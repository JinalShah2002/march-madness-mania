"""

author: Jinal Shah

This script will preprocess all the data
(training, testing, and submission) for prediction.

"""
# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Getting the data
training_data = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/modeling-data/training.csv',index_col=0)
testing_data = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/modeling-data/testing.csv',index_col=0)
submission_data = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/final_submission_processed.csv',index_col=0)

# Dropping unnecessary features
training_data.drop(['lower_TeamID','higher_TeamID'],axis=1,inplace=True)
testing_data.drop(['lower_TeamID','higher_TeamID'],axis=1,inplace=True)
submission_data.drop(['lower_TeamID','higher_TeamID'],axis=1,inplace=True)

# Converting Season into numbers
training_data['Season_converted'] = training_data['Season'] - 2023.0
training_data.drop(['Season'],axis=1,inplace=True)
testing_data['Season_converted'] = testing_data['Season'] - 2023.0
testing_data.drop(['Season'],axis=1,inplace=True)
submission_data['Season_converted'] = submission_data['Season'] - 2023.0
submission_data.drop(['Season'],axis=1,inplace=True)

# Splitting data into numerical and categorical
categorical = ['Bracket']
numerical = list(training_data.columns)
numerical.remove('Bracket')
numerical.remove('LowerWin?')

# Adding LowerWin? in the submission so python doesn't yell at me
submission_data['LowerWin?'] = 0

# Building a pipeline to perform all the appropriate transformations
preprocessing_pipeline = ColumnTransformer(transformers=[
    ('scaler',StandardScaler(with_mean=True,with_std=True),numerical),
    ('encoder',OneHotEncoder(),categorical)
],remainder='passthrough',n_jobs=-1,verbose=True)

# Running the data through the pipeline
training_data_preprocessed = pd.DataFrame(preprocessing_pipeline.fit_transform(training_data))
testing_data_preprocessed = pd.DataFrame(preprocessing_pipeline.transform(testing_data))
submission_data_preprocessed = pd.DataFrame(preprocessing_pipeline.transform(submission_data))

# Dropping the last column in submission_data
submission_data_preprocessed.drop([189],axis=1,inplace=True)

# Saving all these files to run predictions on
training_data_preprocessed.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/prediction-ready-data/training.csv')
testing_data_preprocessed.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/prediction-ready-data/testing.csv')
submission_data_preprocessed.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/prediction-ready-data/submission.csv')
