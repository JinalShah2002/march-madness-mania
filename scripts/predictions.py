"""

author: Jinal Shah

This file will be used to run predictions 
on the test set and the submission set.

"""
# Importing libraries 
import pandas as pd
import pickle
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss

# Getting the datasets
test_data = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/prediction-ready-data/testing.csv',index_col=0)
submission = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/prediction-ready-data/submission.csv',index_col=0)

# Separating test into features and target
features = test_data.drop(['189'],axis=1).values
target = test_data['189'].values

# Getting the best model
best_model = pickle.load(open('/Users/jinalshah/Jinal/Projects/march-madness-mania/models/tuned_ensemble_no_pca.sav','rb'))

# Running predictions on the test set
test_pred = best_model.predict(features)
test_proba = best_model.predict_proba(features)[:,1]
submission_pred = best_model.predict_proba(submission)[:,1]

# Checking to see how the model did on the test set
print(f'Test Brier Score: {brier_score_loss(target,test_proba)}')
print(f'Test Loss: {log_loss(target,test_proba)}')
print(f'Test Accuracy: {accuracy_score(target,test_pred)}')

# Concatenating the probabilities to the final submission
final_submission = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/raw-data/SampleSubmission2023.csv')

# Dropping predictions and replacing with mine
final_submission.drop(['Pred'],axis=1,inplace=True)
final_submission['Pred'] = submission_pred
final_submission.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/final_submission.csv',index=False)

# Mapping Final Submission to team names to build my bracket
final_submission['Season'] = final_submission['ID'].str.split("_").str[0]
final_submission['Team_1'] = final_submission['ID'].str.split("_").str[1]
final_submission['Team_2'] = final_submission['ID'].str.split("_").str[2]
final_submission.drop(['ID'],axis=1,inplace=True)
final_submission['lower_TeamID'] = final_submission[['Team_1','Team_2']].min(axis=1)
final_submission['higher_TeamID'] = final_submission[['Team_1','Team_2']].max(axis=1)

# Splitting data into mens and womens
mens = final_submission[final_submission['lower_TeamID'] < 3000]
womens = final_submission[final_submission['lower_TeamID'] > 3000]

# Splitting into lower and higher teams
team_mens_lower = mens.drop(['higher_TeamID'],axis=1)
team_mens_higher = mens.drop(['lower_TeamID'],axis=1)
team_womens_lower = womens.drop(['higher_TeamID'],axis=1)
team_womens_higher = womens.drop(['lower_TeamID'],axis=1)

# Renaming the columns so that we can merge the statistics in
team_mens_lower.rename(columns={'lower_TeamID':'TeamID'},inplace=True)
team_mens_higher.rename(columns={'higher_TeamID':'TeamID'},inplace=True)
team_womens_lower.rename(columns={'lower_TeamID':'TeamID'},inplace=True)
team_womens_higher.rename(columns={'higher_TeamID':'TeamID'},inplace=True)

# Mapping to teams
mens_teams = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/raw-data/MTeams.csv')
womens_teams = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/raw-data/WTeams.csv')

team_mens_lower = team_mens_lower.merge(mens_teams,how='left',on=['TeamID'])
team_mens_higher = team_mens_higher.merge(mens_teams,how='left',on=['TeamID'])
team_womens_lower = team_womens_lower.merge(womens_teams,how='left',on=['TeamID'])
team_womens_higher = team_womens_higher.merge(mens_teams,how='left',on=['TeamID'])

# Renaming columns prior to adding together
rename_cols_lower = {}
rename_cols_higher = {}

for column in team_mens_lower.columns:
    rename_cols_lower[column] = 'lower_'+column

for column in team_mens_higher.columns:
    rename_cols_higher[column] = 'higher_'+column

team_mens_lower.rename(columns=rename_cols_lower,inplace=True)
team_mens_higher.rename(columns=rename_cols_higher,inplace=True)
team_womens_lower.rename(columns=rename_cols_lower,inplace=True)
team_womens_higher.rename(columns=rename_cols_higher,inplace=True)

# Merging back into mens and womens
final_mens = pd.concat([team_mens_lower,team_mens_higher],axis=1)
final_womens = pd.concat([team_womens_lower,team_womens_higher],axis=1)

# print(final_mens[['lower_TeamName','higher_TeamName']])

# Save each for easy lookup
final_mens.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/final_submission_mens.csv')
final_womens.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/final_submission_womens.csv')