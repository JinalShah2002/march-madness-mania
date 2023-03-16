"""

author: Jinal Shah

This script will be used to build the final model.
The model was chosen based on results in the 
comparing_model_performances notebook. 

"""
# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
import warnings
import pickle

# ignoring warnings
warnings.filterwarnings('ignore')

# Getting the data
training_data = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/prediction-ready-data/training.csv',index_col=0)

# Splitting data into features and target
features = training_data.drop(['189'],axis=1).values
target = training_data['189'].values

"""

Model without PCA

Difference in performance is very minimal
so I want to see if 1 model can take the 
lead once and for all.


"""
model_no_pca = VotingClassifier(estimators=[
    ('lr',LogisticRegression(penalty=None,C=1.0,random_state=42,max_iter=5000,n_jobs=-1)),
    ('rf',RandomForestClassifier(n_estimators=1000,criterion='gini',bootstrap=True,n_jobs=-1,random_state=42,max_samples=0.7))
],voting='soft',n_jobs=-1)

param_grid = {'lr__penalty':['elasticnet','l2'],'lr__C':[0.3,0.5,1,1.5,3,10],'lr__random_state':[42],'lr__max_iter':[1000],
              'lr__n_jobs':[-1],"lr__l1_ratio":[0.3,0.5,0.8],'rf__n_estimators':[30,50,100],'rf__max_depth':[5,10,20,50],
              'rf__min_samples_leaf':[1,3,5],'rf__max_leaf_nodes':[None,5,10,15],'rf__n_jobs':[-1],
              'rf__random_state':[42],'rf__ccp_alpha':[0,0.3,0.7,3],'rf__max_samples':[0.5,0.7,1],'lr__solver':['saga']}

# Performing Random Search on this Parameter Space
clf_no_pca = RandomizedSearchCV(estimator=model_no_pca,param_distributions=param_grid,n_iter=50,n_jobs=-1,refit='neg_brier_score',
                         cv=5,random_state=42,return_train_score=True,scoring=['neg_brier_score','neg_log_loss','accuracy'])

print('Starting Non-PCA Model...')
# Fitting it
clf_no_pca.fit(features,target)

# Saving in Dataframe
best_brier_no_pca = pd.DataFrame(clf_no_pca.cv_results_)[['mean_test_neg_brier_score','mean_test_accuracy','mean_test_neg_log_loss']].loc[clf_no_pca.best_index_,:]

# Saving the best model
pickle.dump(clf_no_pca.best_estimator_,open('/Users/jinalshah/Jinal/Projects/march-madness-mania/models/tuned_ensemble_no_pca.sav','wb'))

print('Non-PCA Done!')
print()
"""

Model with PCA

"""
model_with_pca = VotingClassifier(estimators=[
    ('lr',LogisticRegression(penalty=None,C=1.0,random_state=42,max_iter=1000,n_jobs=-1)),
    ('rf',RandomForestClassifier(n_estimators=1000,criterion='gini',bootstrap=True,n_jobs=-1,random_state=42,max_samples=0.7))
],voting='soft',n_jobs=-1)

# Performing PCA
pca = PCA(n_components= 0.97)
features_transformed = pca.fit_transform(features,target)

# Random Search
clf_pca = RandomizedSearchCV(estimator=model_with_pca,param_distributions=param_grid,n_iter=50,n_jobs=-1,refit='neg_brier_score',
                         cv=5,random_state=42,return_train_score=True,scoring=['neg_brier_score','neg_log_loss','accuracy'])

print('Starting PCA Model...')
# Fitting it
clf_pca.fit(features_transformed,target)

# Saving in Dataframe
best_brier_pca = pd.DataFrame(clf_pca.cv_results_)[['mean_test_neg_brier_score','mean_test_accuracy','mean_test_neg_log_loss']].loc[clf_pca.best_index_,:]

# Saving the best model
pickle.dump(clf_pca.best_estimator_,open('/Users/jinalshah/Jinal/Projects/march-madness-mania/models/tuned_ensemble_pca.sav','wb'))

print('PCA Done!')
print()

# Printing the results
print('Without PCA: ')
print(best_brier_no_pca)
print()
print('With PCA:')
print(best_brier_pca)
print()

# Without PCA was the best!