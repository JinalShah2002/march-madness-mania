"""

author: Jinal Shah

In this script, I am simply splitting
the processed data into training & testing sets.

For validation, I will use cross validation since 
the dataset isn't quite big enough for a separate validation set.

"""
# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Getting the datasets
mens_data = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/training-data/mens.csv',index_col=0)
womens_data = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/training-data/womens.csv',index_col=0)

"""

Splitting the data:

I will be splitting the data using Stratified Sampling by year.
The game changes overtime (ex: emergence of 3-pointers) hence,
I don't want the model to be solely trained on old data.


Will stratify both men and women datasets then combine to form 
the training & testing sets.

"""
# Separating data into features matrix and target vector
X_mens = mens_data.drop(['LowerWin?'],axis=1)
y_mens = mens_data['LowerWin?']
X_womens = womens_data.drop(['LowerWin?'],axis=1)
y_womens = womens_data['LowerWin?']

# Stratified sampling by Season for each bracket
X_train_mens, X_test_mens, y_train_mens, y_test_mens = train_test_split(X_mens,y_mens,test_size=0.2,random_state=42,shuffle=True,stratify=X_mens['Season'])
X_train_womens, X_test_womens, y_train_womens, y_test_womens = train_test_split(X_womens,y_womens,test_size=0.2,random_state=42,shuffle=True,stratify=X_womens['Season'])

# Stacking the mens and womens data
X_training = pd.concat([X_train_mens,X_train_womens])
y_training = pd.concat([y_train_mens,y_train_womens])
X_testing = pd.concat([X_test_mens,X_test_womens])
y_testing = pd.concat([y_test_mens,y_test_womens])

# Combining to form training & testing data
training = pd.concat((X_training,y_training),axis=1)
testing = pd.concat((X_testing,y_testing),axis=1)

# Saving the datasets
training.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/modeling-data/training.csv')
testing.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/modeling-data/testing.csv')
