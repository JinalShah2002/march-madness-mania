"""
author: Jinal Shah

This script is going host a class to 
perform all feature engineering for 
incoming data.

I assume that the data is the same format
as the regular season and tournament data
given to us via the Kaggle competition page.
This assumption assumes all feature names are the same
and the structure of the incoming data is the same.

"""
# Importing libraries
import pandas as pd

class AddFeatures():
    """
    AddFeatures

    A class that adds features to the original dataset
    with the box scores of each game.
    
    """
    def __init__(self) -> None:
        """
        Constructor

        Constructs a AddFeatures object.
        
        inputs:
        -None
        outputs:
        -None
        """
        pass

    """
    This section will contain functions 
    that will calculate the necessary statistics
    based off of the base dataset from Kaggle.
    """

    def field_goal_percentage(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        field_goal_percentage

        A method that gives us the field goal percentage 
        and 3 point field goal percentage for both the winners 
        and losers of each game. 
        
        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.
        """
        # Making a copy of the data
        data_copy = data

        # Adding the field goal percentages
        data_copy['WFieldGoal'] = data_copy['WFGM'] / data_copy['WFGA'] * 100
        data_copy['LFieldGoal'] = data_copy['LFGM'] / data_copy['LFGA'] * 100

        # Adding the 3PT field goal percentages
        data_copy['WFieldGoal3'] = data_copy['WFGM3'] / data_copy['WFGA3'] * 100
        data_copy['LFieldGoal3'] = data_copy['LFGM3'] / data_copy['LFGA3'] * 100

        # returning the new dataset
        return data_copy
    
    def free_throw_percentage(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        free_throw_percentage

        A function that gives us the free throw
        percentage of the each team in the game.

        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.
        """
        # Making a copy of the data
        data_copy = data.copy()
        data_copy['WFreeThrow'] = data_copy['WFTM'] / data_copy['WFTA'] * 100
        data_copy['LFreeThrow'] = data['LFTM'] / data_copy['LFTA'] * 100

        # returning the new data
        return data_copy

    def rebounds(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        rebounds

        A function that gives the following rebounding metrics:
        - Total Rebounds
        - % of total rebounds that were defensive rebounds
        - % of total rebounds that were offensive rebounds

        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.
        """
        # Creating data copy
        data_copy = data.copy()

        # Adding total rebounds
        data_copy['WTotalRebounds'] = data_copy['WOR'] + data_copy['WDR']
        data_copy['LTotalRebounds'] = data_copy['LOR'] + data_copy['LDR']

        # Adding % of Offensive rebounds
        data_copy['WORPercent'] = data_copy['WOR'] / data_copy['WTotalRebounds']
        data_copy['LORPercent'] = data_copy['LOR'] / data_copy['LTotalRebounds']

        # Adding % of Defensive rebounds
        data_copy['WDRPercent'] = data_copy['WDR'] / data_copy['WTotalRebounds']
        data_copy['LDRPercent'] = data_copy['LDR'] / data_copy['LTotalRebounds']

        # Returning the new data
        return data_copy
    
    def assist_to_turnover(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        assist_to_turnover

        A method that calculates the assist to turnover
        ratio (assist/turnover).

        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.
        """
        # Creating a data copy
        data_copy = data.copy()

        # Adding assist to turnover
        data_copy['WAssistToTurnoverRatio'] = data_copy['WAst'] / data_copy['WTO']
        data_copy['LAssistToTurnoverRatio'] = data_copy['LAst'] / data_copy['LTO']

        # Returning the new data
        return data_copy

    def ratings(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        ratings

        A method to add the offensive and defensive ratings 
        I came up with. Formulas are pretty rough but 
        I am interested to see how they impact decisions.

        Formulas:
            - Team Offensive Rating = (points * fg % * 3 PT fg %) + assists + offensive boards + free throw attempts + free throw percentage - 2.5*turnovers 
            - Team Defensive Rating = (steals * blocks * defensive boards) - Opp FG % - Opp 3 PT %  - # of free throws taken by opponent
        
        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.
        """        
        # Create a data copy
        data_copy = data.copy()

        # Adding Offensive Rating
        data_copy['WOffensiveRating'] = abs((data_copy['WScore'] * data_copy['WFieldGoal'] * data_copy['WFieldGoal3']) + data_copy['WAst'] + data_copy['WOR'] + data_copy['WFTA'] + data_copy['WFreeThrow'] - 2.5 * data_copy['WTO'])
        data_copy['LOffensiveRating'] = abs((data_copy['LScore'] * data_copy['LFieldGoal'] * data_copy['LFieldGoal3']) + data_copy['LAst'] + data_copy['LOR'] + data_copy['LFTA'] + data_copy['LFreeThrow'] - 2.5 * data_copy['LTO'])

        # Adding Defensive Rating
        data_copy['WDefensiveRating'] = abs((data_copy['WStl'] * data_copy['WBlk'] * data_copy['WDR']) - data_copy['WOpposingFG'] - data_copy['LFTA'])
        data_copy['LDefensiveRating'] = abs((data_copy['LStl'] * data_copy['LBlk'] * data_copy['LDR']) - data_copy['LOpposingFG'] - data_copy['WFTA'])  

        # Returning the data
        return data_copy
    
    def opposing_field_goal_percentage(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        opposing_field_goal_percentage

        A method to add the opposing team's fg percentage to each team.
        F.E if LTeam had a field goal percentage of 38%, WOpposingFG would be 38

        I am doing this for 3 PT as well.

        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.
        """
        # Creating a data copy
        data_copy = data.copy()

        # Adding the total Opposing FG
        data_copy['WOpposingFG'] = data_copy['LFieldGoal']
        data_copy['LOpposingFG'] = data_copy['WFieldGoal']

        # Adding the opposing 3 PT FG
        data_copy['WOpposingFG3'] = data_copy['LFieldGoal3']
        data_copy['LOpposingFG3'] = data_copy['WFieldGoal3']

        # returning the data
        return data_copy

    def possession(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        possession

        A method that calculates possesion:
        Possesion = 0.96 (FGA - OR - TO + (0.475 * FTA))
        Credit: Formula was provided by this paper: 
        https://www.researchgate.net/publication/257749099_Predicting_college_basketball_match_outcomes_using_machine_learning_techniques_some_results_and_lessons_learned
        
        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.

        """
        # Creating a copy of the data
        data_copy = data.copy()

        # Adding possesions
        data_copy['WPossessions'] = 0.96 * (data_copy['WFGA'] - data_copy['WOR'] - data_copy['WTO'] + (0.475 * data_copy['WFTA']))
        data_copy['LPossessions'] = 0.96 * (data_copy['LFGA'] - data_copy['LOR'] - data_copy['LTO'] + (0.475 * data_copy['LFTA']))

        # returning the data
        return data_copy
    
    def efficiencies(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        efficiencies

        A method to calculate the efficiencies:
        Offensive Efficiency = Points Scored * 100 / Possessions
        Defensive Efficiency = Points allowed * 100 / Possessions

        Formulas provided by this paper:
        https://www.researchgate.net/publication/257749099_Predicting_college_basketball_match_outcomes_using_machine_learning_techniques_some_results_and_lessons_learned

        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.
        
        """
        # Making a copy of the data
        data_copy = data.copy()

        # Adding the efficiencies
        data_copy['WOffEff'] = (data_copy['WScore'] * 100)/data_copy['WPossessions']
        data_copy['LOffEff'] = (data_copy['LScore'] * 100)/data_copy['LPossessions']
        data_copy['WDefEff'] = (data_copy['LScore'] * 100)/data_copy['WPossessions']
        data_copy['LDefEff'] = (data_copy['WScore'] * 100)/data_copy['LPossessions']

        # returning the data
        return data_copy
    
    def turnover_percentage(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        turnover_percentage

        A method to calculate turnover percentage.

        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed data.
        
        """
        # Making a copy of the data
        data_copy = data.copy()

        # Adding turnover percentage
        data_copy['WTO%'] = data_copy['WTO'] / data_copy['WPossessions']
        data_copy['LTO%'] = data_copy['LTO'] / data_copy['LPossessions']

        # Returning the data
        return data_copy
        

    def transform(self,data:pd.DataFrame) -> pd.DataFrame:
        """
        transform

        A method that adds all the features
        defined in the above methods.

        inputs:
        - data: a Pandas Dataframe containing the data.
        outputs:
        - a Pandas Dataframe with the transformed datas
        
        """
        # Getting the copy of the data
        final_data = data.copy()

        # Adding field goal percentages, free throw percentages, and opposing field goal percentages
        final_data = self.field_goal_percentage(final_data)
        final_data = self.free_throw_percentage(final_data)
        final_data = self.opposing_field_goal_percentage(final_data)

        # Adding the rebounds
        final_data = self.rebounds(final_data)

        # Adding assist to turnover ratio
        final_data = self.assist_to_turnover(final_data)

        # Adding possessions
        final_data = self.possession(final_data)

        # Adding the efficiencies
        final_data = self.efficiencies(final_data)

        # Adding turnover percentage
        final_data = self.turnover_percentage(final_data)

        # Adding Point Differential
        final_data['WPointDiff'] = final_data['WScore'] - final_data['LScore']
        final_data['LPointDiff'] = final_data['LScore'] - final_data['WScore']

        # Adding the ratings
        final_data = self.ratings(final_data)

        # returning the transformed data
        return final_data
    
# Main Method to transform all the data
if __name__ == '__main__':
    # Getting the data sets
    men_reg = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/Actual Data /MRegularSeasonDetailedResults.csv')
    men_tourney = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/Actual Data /MNCAATourneyDetailedResults.csv')
    women_reg = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/Actual Data /WRegularSeasonDetailedResults.csv')
    women_tourney = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/Actual Data /WNCAATourneyDetailedResults.csv')

    # Creating the pipeline object
    features = AddFeatures()

    # Putting each file through the pipeline
    final_data = features.transform(men_reg)
    final_data.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/data-preprocessed/men_reg.csv')
    final_data = features.transform(men_tourney)
    final_data.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/data-preprocessed/men_tourney.csv')
    final_data = features.transform(women_reg)
    final_data.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/data-preprocessed/women_reg.csv')
    final_data = features.transform(women_tourney)
    final_data.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/data/data-preprocessed/women_tourney.csv')
