"""

author: Jinal Shah

This script will prepare the submission to be run through
the chosen model. Essentially, I just need to map the statistics
from 2023 to the appropriate data (mean, median, etc.).


"""
# Importing libraries
import pandas as pd

# Getting the final submission games
final_submission = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/raw-data/SampleSubmission2023.csv')

# Splitting up the submission id into columns
final_submission['Season'] = final_submission['ID'].str.split("_").str[0]
final_submission['Team_1'] = final_submission['ID'].str.split("_").str[1]
final_submission['Team_2'] = final_submission['ID'].str.split("_").str[2]
final_submission.drop(['ID'],axis=1,inplace=True)

# Changing Season to be a number instead of an object
final_submission['Season'] = final_submission['Season'].astype(int)

# Setting which team is the higher and which team is the lower
final_submission['lower_TeamID'] = final_submission[['Team_1','Team_2']].min(axis=1)
final_submission['higher_TeamID'] = final_submission[['Team_1','Team_2']].max(axis=1)

# Dropping Team_1, Team_2 and Pred
final_submission.drop(['Pred','Team_1','Team_2'],axis=1,inplace=True)

# Splitting data into higher and lower
team_lower = final_submission.drop(['higher_TeamID'],axis=1)
team_higher = final_submission.drop(['lower_TeamID'],axis=1)

# Renaming the columns so that we can merge the statistics in
team_lower.rename(columns={'lower_TeamID':'TeamID'},inplace=True)
team_higher.rename(columns={'higher_TeamID':'TeamID'},inplace=True)

# Splitting data into mens and womens because merging is breaking
team_lower_mens = team_lower[team_lower['TeamID'] < 3000]
team_lower_womens = team_lower[team_lower['TeamID'] > 3000]
team_higher_mens = team_higher[team_higher['TeamID'] < 3000]
team_higher_womens = team_higher[team_higher['TeamID'] > 3000]


"""

Performing the Mappings!

This section will perform all the mappings 
of the statistics to the teams.

"""
# Getting the datasets needed for mapping
mens_record_info = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/mens_reg_szn_records.csv',index_col=0)
mens_mean_reg_szn = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/mens_reg_szn_avgs.csv',index_col=0)
mens_median_reg_szn = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/mens_reg_szn_median.csv',index_col=0)
mens_std_reg_szn = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/mens_reg_szn_std.csv',index_col=0)
womens_record_info = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/womens_reg_szn_records.csv',index_col=0)
womens_mean_reg_szn = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/womens_reg_szn_avgs.csv',index_col=0)
womens_median_reg_szn = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/womens_reg_szn_median.csv',index_col=0)
womens_std_reg_szn = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/womens_reg_szn_std.csv',index_col=0)

# Doing column renaming for the mean, median, and std for easier differentiation
# Mean
rename_mean_cols = {}
for column in mens_mean_reg_szn.columns:
    if column != 'Season' and column != 'TeamID':
        rename_mean_cols[column] = column+"_mean"
mens_mean_reg_szn.rename(columns=rename_mean_cols,inplace=True)
womens_mean_reg_szn.rename(columns=rename_mean_cols,inplace=True)

# Median
rename_median_cols = {}
for column in mens_median_reg_szn.columns:
    if column != 'Season' and column != 'TeamID':
        rename_median_cols[column] = column+"_median"

mens_median_reg_szn.rename(columns=rename_median_cols,inplace=True)
womens_median_reg_szn.rename(columns=rename_median_cols,inplace=True)

# Standard Deviation
rename_std_cols = {}
for column in mens_std_reg_szn.columns:
    if column != 'Season' and column != 'TeamID':
        rename_std_cols[column] = column+"_std"
mens_std_reg_szn.rename(columns=rename_std_cols,inplace=True)
womens_std_reg_szn.rename(columns=rename_std_cols,inplace=True)

# Merging the record information
team_lower_mens = team_lower_mens.merge(mens_record_info,how='left',on=['Season','TeamID'])
team_higher_mens = team_higher_mens.merge(mens_record_info,how='left',on=['Season','TeamID'])
team_lower_womens = team_lower_womens.merge(womens_record_info,how='left',on=['Season','TeamID'])
team_higher_womens = team_higher_womens.merge(womens_record_info,how='left',on=['Season','TeamID'])

# Merging the means
team_lower_mens = team_lower_mens.merge(mens_mean_reg_szn,how='left',on=['Season','TeamID'])
team_higher_mens = team_higher_mens.merge(mens_mean_reg_szn,how='left',on=['Season','TeamID'])
team_lower_womens = team_lower_womens.merge(womens_mean_reg_szn,how='left',on=['Season','TeamID'])
team_higher_womens = team_higher_womens.merge(womens_mean_reg_szn,how='left',on=['Season','TeamID'])

# Merging the medians
team_lower_mens = team_lower_mens.merge(mens_median_reg_szn,how='left',on=['Season','TeamID'])
team_higher_mens = team_higher_mens.merge(mens_median_reg_szn,how='left',on=['Season','TeamID'])
team_lower_womens = team_lower_womens.merge(womens_median_reg_szn,how='left',on=['Season','TeamID'])
team_higher_womens = team_higher_womens.merge(womens_median_reg_szn,how='left',on=['Season','TeamID'])

# Merging the standard deviations
team_lower_mens = team_lower_mens.merge(mens_std_reg_szn,how='left',on=['Season','TeamID'])
team_higher_mens = team_higher_mens.merge(mens_std_reg_szn,how='left',on=['Season','TeamID'])
team_lower_womens = team_lower_womens.merge(womens_std_reg_szn,how='left',on=['Season','TeamID'])
team_higher_womens = team_higher_womens.merge(womens_std_reg_szn,how='left',on=['Season','TeamID'])


# Renaming columns to distinguish 
rename_cols_lower = {}
rename_cols_higher = {}

for column in team_lower_mens.columns:
    rename_cols_lower[column] = 'lower_'+column

for column in team_higher_mens.columns:
    rename_cols_higher[column] = 'higher_'+column

team_lower_mens.rename(columns=rename_cols_lower,inplace=True)
team_higher_mens.rename(columns=rename_cols_higher,inplace=True)
team_lower_womens.rename(columns=rename_cols_lower,inplace=True)
team_higher_womens.rename(columns=rename_cols_higher,inplace=True)

# Concatenating to a final dataset
final_dataset_mens = pd.concat([team_lower_mens,team_higher_mens],axis=1)
final_dataset_womens = pd.concat([team_lower_womens,team_higher_womens],axis=1)

# Some column renaming and adjustments to be done:
final_dataset_mens.drop(['higher_Season'],axis=1,inplace=True)
final_dataset_mens.rename(columns={'lower_Season':'Season'},inplace=True)
final_dataset_womens.drop(['higher_Season'],axis=1,inplace=True)
final_dataset_womens.rename(columns={'lower_Season':'Season'},inplace=True)

# Creating a feature indicating the type of games each game is 
final_dataset_mens['Bracket'] = 'M'
final_dataset_womens['Bracket'] = 'W'

# Concatenating to a final dataset
final_submission_mapped = pd.concat([final_dataset_mens,final_dataset_womens],axis=0)

# Saving this file for prediction
final_submission_mapped.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/final_submission_processed.csv')