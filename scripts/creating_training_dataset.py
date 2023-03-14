"""

author: Jinal Shah

This script will focus on creating the actual dataset
for EDA and modeling. From the problem description,
we know that we must predict whether or not the team
with the lower teamid wins (in reality, we spit out a 
probability but that is a modeling thing).

Hence, we need to take the tourney games and create the 
dataset that grabs each team's information and combines it 
into a big dataset for modeling and prediction.

"""
# Importing libraries
import pandas as pd
import warnings

warnings.filterwarnings('ignore') # Ignoring warning because they are getting in the way

# Loading the tournament datasets
mens_tourney = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/raw-data/MNCAATourneyDetailedResults.csv')
womens_tourney = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/raw-data/WNCAATourneyDetailedResults.csv')

# Creating the initial dataset that simply gives us the season, teams ,and whether the lower team id won
initial_mens_data = mens_tourney[['Season','WTeamID','LTeamID']]
initial_womens_data = womens_tourney[['Season','WTeamID','LTeamID']]

# Adding the target variable (Lower Team won)
initial_mens_data['LowerWin?'] = initial_mens_data['WTeamID'] < initial_mens_data['LTeamID']
initial_womens_data['LowerWin?'] = initial_womens_data['WTeamID'] < initial_womens_data['LTeamID']
initial_mens_data['LowerWin?']= initial_mens_data['LowerWin?'].astype(int)
initial_womens_data['LowerWin?']= initial_womens_data['LowerWin?'].astype(int)

# Renaming the columns to split up the teams in the games properly
initial_mens_data.rename(columns={'WTeamID':'TeamA','LTeamID':'TeamB'},inplace=True)
initial_womens_data.rename(columns={'WTeamID':'TeamA','LTeamID':'TeamB'},inplace=True)

# Splitting up the data into team A and team B for merging purposes of the features
team_a_mens = initial_mens_data.drop(['TeamB'],axis=1)
team_b_mens = initial_mens_data.drop(['TeamA','LowerWin?'],axis=1)
team_a_womens = initial_womens_data.drop(['TeamB'],axis=1)
team_b_womens = initial_womens_data.drop(['TeamA','LowerWin?'],axis=1)

# Renaming the columns so that we can merge the statistics in
team_a_mens.rename(columns={'TeamA':'TeamID'},inplace=True)
team_b_mens.rename(columns={'TeamB':'TeamID'},inplace=True)
team_a_womens.rename(columns={'TeamA':'TeamID'},inplace=True)
team_b_womens.rename(columns={'TeamB':'TeamID'},inplace=True)

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
team_a_mens = team_a_mens.merge(mens_record_info,how='left',on=['Season','TeamID'])
team_b_mens = team_b_mens.merge(mens_record_info,how='left',on=['Season','TeamID'])
team_a_womens = team_a_womens.merge(womens_record_info,how='left',on=['Season','TeamID'])
team_b_womens = team_b_womens.merge(womens_record_info,how='left',on=['Season','TeamID'])

# Merging the means
team_a_mens = team_a_mens.merge(mens_mean_reg_szn,how='left',on=['Season','TeamID'])
team_b_mens = team_b_mens.merge(mens_mean_reg_szn,how='left',on=['Season','TeamID'])
team_a_womens = team_a_womens.merge(womens_mean_reg_szn,how='left',on=['Season','TeamID'])
team_b_womens = team_b_womens.merge(womens_mean_reg_szn,how='left',on=['Season','TeamID'])

# Merging the medians
team_a_mens = team_a_mens.merge(mens_median_reg_szn,how='left',on=['Season','TeamID'])
team_b_mens = team_b_mens.merge(mens_median_reg_szn,how='left',on=['Season','TeamID'])
team_a_womens = team_a_womens.merge(womens_median_reg_szn,how='left',on=['Season','TeamID'])
team_b_womens = team_b_womens.merge(womens_median_reg_szn,how='left',on=['Season','TeamID'])

# Merging the standard deviations
team_a_mens = team_a_mens.merge(mens_std_reg_szn,how='left',on=['Season','TeamID'])
team_b_mens = team_b_mens.merge(mens_std_reg_szn,how='left',on=['Season','TeamID'])
team_a_womens = team_a_womens.merge(womens_std_reg_szn,how='left',on=['Season','TeamID'])
team_b_womens = team_b_womens.merge(womens_std_reg_szn,how='left',on=['Season','TeamID'])

# Renaming columns to distinguish 
rename_cols_a = {}
rename_cols_b = {}

for column in team_a_mens.columns:
    rename_cols_a[column] = 'A_'+column

for column in team_b_mens.columns:
    rename_cols_b[column] = 'B_'+column

team_a_mens.rename(columns=rename_cols_a,inplace=True)
team_b_mens.rename(columns=rename_cols_b,inplace=True)
team_a_womens.rename(columns=rename_cols_a,inplace=True)
team_b_womens.rename(columns=rename_cols_b,inplace=True)

# Concatenating to a final dataset
final_dataset_mens = pd.concat([team_a_mens,team_b_mens],axis=1)
final_dataset_womens = pd.concat([team_a_womens,team_b_womens],axis=1)

# Some column renaming and adjustments to be done:
final_dataset_mens.drop(['B_Season'],axis=1,inplace=True)
final_dataset_mens.rename(columns={'A_LowerWin?':'LowerWin?','A_Season':'Season'},inplace=True)
final_dataset_womens.drop(['B_Season'],axis=1,inplace=True)
final_dataset_womens.rename(columns={'A_LowerWin?':'LowerWin?','A_Season':'Season'},inplace=True)

# Saving the datasets
final_dataset_mens.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/training-data/mens.csv')
final_dataset_womens.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/training-data/womens.csv')
