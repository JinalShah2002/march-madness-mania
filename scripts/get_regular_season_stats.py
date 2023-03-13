"""

author: Jinal Shah

This script will get the regular season numbers
for each team for each season. Box score statistics
will be aggregated via mean, median, and standard deviation.

This is done to mitigate impact outliers have and to best 
forecast the winner.

"""
# Importing libraries
import pandas as pd

"""

Helper Functions

"""
def get_record_info(winning_data: pd.DataFrame, losing_data:pd.DataFrame) -> list:
    """
    get_record_info

    A function to get the record information:
    - Wins
    - Losses
    - Winning %

    inputs:
    - winning_data: Pandas Dataframe with box scores of wins
    - losing_data: Pandas Dataframe with box scores of losses

    output:
    - a list of length 3 with index 0 = wins, index 1 = losses, index 2 = winning percentage
    
    """
    # Creating variables
    wins = winning_data.shape[0]
    losses = losing_data.shape[0]
    winning_percentage = wins / (wins + losses)

    # returning the list
    return [wins,losses,winning_percentage]


def get_every_season_record(raw_data:pd.DataFrame) -> dict:
    """
    get_every_season_record

    A function to get the record of every team 
    for every season.

    inputs:
    - raw_data: a pandas dataframe with the raw data
    outputs:
    - a dictionary with the record information for every team
    for every season.
    
    """
    # Getting the names of the columns for the winners and losers
    winning_cols = []
    losing_cols = []
    for column in raw_data.columns:
        if column[0] == 'W':
            winning_cols.append(column)
        elif column[0] == 'L':
            losing_cols.append(column)

    # Getting the mens record information for each team for each season
    seasons = list(raw_data['Season'].unique())
    final_record_info = {'Season':[],'TeamID':[],'Wins':[],'Losses':[],'Winning Percentage':[]}
    for season in seasons:
        # Getting the data for the season
        season_data = raw_data[raw_data['Season'] == season]

        # Getting the winning data and losing data
        winning = season_data[winning_cols]
        losing = season_data[losing_cols]

        # Getting all the teams
        winners = list(season_data['WTeamID'].unique())
        losers = list(season_data['LTeamID'].unique())
        winners.extend(losers)
        total_teams = list(set(winners))

        for team in total_teams:
            # Getting the record info
            wins = winning[winning['WTeamID'] == team]
            losses = losing[losing['LTeamID'] == team]
            record_info = get_record_info(wins,losses)

            # Adding the information to the dictionary
            temp = final_record_info['Season']
            temp.append(season)
            final_record_info['Season'] = temp
            temp = final_record_info['TeamID']
            temp.append(team)
            final_record_info['TeamID'] = temp
            temp = final_record_info['Wins']
            temp.append(record_info[0])
            final_record_info['Wins'] = temp
            temp = final_record_info['Losses']
            temp.append(record_info[1])
            final_record_info['Losses'] = temp
            temp = final_record_info['Winning Percentage']
            temp.append(record_info[2])
            final_record_info['Winning Percentage'] = temp
    
    # returning the dictionary with the records
    return final_record_info

def aggregate(raw_data: pd.DataFrame,statistic:str) -> pd.DataFrame:
    """
    aggregate_by_mean

    A function to aggregate the box score by the specified statistic
    for every season for every team.

    inputs:
    - raw_data: the raw data with all the box scores
    - statistic: the statistic to aggregate by (options are mean, median, or standard deviation)
    outputs:
    - a dataframe with the aggregated box scores by mean 
    
    """
    # Getting the names of the columns for the winners and losers
    winning_cols = []
    losing_cols = []
    for column in raw_data.columns:
        if column[0] == 'W' and column != 'WLoc':
            winning_cols.append(column)
        elif column[0] == 'L':
            losing_cols.append(column)

    # Getting the mens record information for each team for each season
    seasons = list(raw_data['Season'].unique())

    # Creating the final dataframe to return
    final_df = None

    # Renaming the columns so we can line up
    # Stripping the first letter (W or L)
    winning_replace = {}
    loser_replace = {}
    for column in winning_cols:
        new_col_name = column[1:]
        winning_replace[column] = new_col_name
        loser_replace['L'+new_col_name] = new_col_name

    # Iterating through the seasons
    for season in seasons:
        # Getting the season data
        season_data = raw_data[raw_data['Season']==season]

        # Getting the winners and losers dataframes
        winning = season_data[winning_cols]
        losing = season_data[losing_cols]

        # Renaming the columns
        winning.rename(columns=winning_replace,inplace=True)
        losing.rename(columns=loser_replace,inplace=True)

        # Getting the statistic of the stats
        if statistic.lower() == 'mean':
            team_stats = pd.concat([winning,losing],axis=0).groupby(by='TeamID',axis='index').mean()
        elif statistic.lower() == 'median':
            team_stats = pd.concat([winning,losing],axis=0).groupby(by='TeamID',axis='index').median()
        else:
            team_stats = pd.concat([winning,losing],axis=0).groupby(by='TeamID',axis='index').std()

        team_stats.reset_index(inplace=True)

        # Adding the season
        team_stats['Season'] = season

        # Adding it to the final dataframe
        if final_df is None:
            final_df = team_stats
        else:
            final_df = pd.concat([final_df,team_stats],axis=0)
    
    # Returning the final df
    return final_df


# Main Method
if __name__ == '__main__':
    # Getting the datasets
    mens_reg = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/expanded-features/men_reg.csv',index_col=0)
    womens_reg = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/expanded-features/women_reg.csv',index_col=0)

    # Getting the records data and saving it
    mens_records = pd.DataFrame.from_dict(get_every_season_record(mens_reg))
    womens_records = pd.DataFrame.from_dict(get_every_season_record(womens_reg))
    mens_records.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/mens_reg_szn_records.csv')
    womens_records.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/womens_reg_szn_records.csv')
    
    # Getting the statistics and saving to a csv
    mens_avg = aggregate(mens_reg,statistic='mean')
    mens_median = aggregate(mens_reg,statistic='median')
    mens_std = aggregate(mens_reg,statistic='standard deviation')
    mens_avg.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/mens_reg_szn_avgs.csv')
    mens_median.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/mens_reg_szn_median.csv')
    mens_std.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/mens_reg_szn_std.csv')
    womens_avg = aggregate(womens_reg,statistic='mean')
    womens_median = aggregate(womens_reg,statistic='median')
    womens_std = aggregate(womens_reg,statistic='standard deviation')
    womens_avg.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/womens_reg_szn_avgs.csv')
    womens_median.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/womens_reg_szn_median.csv')
    womens_std.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/regular-season-statistics/womens_reg_szn_std.csv')












# # Getting the datasets
# mens_reg = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/data-preprocessed/men_reg.csv',index_col=0)

# # womens_reg = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/data-preprocessed/women_reg.csv',index_col=0)
# # womens_tourney = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/data-preprocessed/women_tourney.csv',index_col=0)


# twenty_three = mens_reg[mens_reg['Season'] == 2023]

# winning_cols = []
# losing_cols = []
# for column in twenty_three.columns:
#     if column[0] == 'W':
#         winning_cols.append(column)
#     elif column[0] == 'L':
#         losing_cols.append(column)

# winners = twenty_three[winning_cols]
# losers = twenty_three[losing_cols]

# rutgers_wins = winners[winners['WTeamID'] == 1353]
# rutgers_loss = losers[losers['LTeamID'] == 1353]


# rutgers_wins.drop(['WLoc'],axis=1,inplace=True)

# # Stripping the first letter (W or L)
# winning_replace = {}
# loser_replace = {}
# for column in rutgers_wins.columns:
#     new_col_name = column[1:]
#     winning_replace[column] = new_col_name
#     loser_replace['L'+new_col_name] = new_col_name

# rutgers_wins.rename(columns=winning_replace,inplace=True)
# rutgers_loss.rename(columns=loser_replace,inplace=True)


# # Concatenating
# rutgers = pd.concat([rutgers_wins,rutgers_loss],axis=0)
# rutgers_mean = rutgers.groupby(by='TeamID',axis='index').mean()
# rutgers_median = rutgers.groupby(by='TeamID',axis='index').median()
# rutgers_std = rutgers.groupby(by='TeamID',axis='index').std()
# rutgers_concat = pd.concat([rutgers_mean,rutgers_median,rutgers_std],axis=0)
# rutgers_concat.reset_index(inplace=True)
# print(rutgers_concat)

