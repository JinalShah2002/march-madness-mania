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

