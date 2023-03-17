# March Madness Mania
![alt text](https://upload.wikimedia.org/wikipedia/en/thumb/2/28/March_Madness_logo.svg/1200px-March_Madness_logo.svg.png)

## Project Purpose
To be able to develop a machine learning & data science backed solution that is able to accurately predict winners of games in the NCAA March Madness tournament (for both the Men's and Women's tournaments).

## Project Details
As an avid basketball fan and an upcoming data scientist, I decided to tackle the [2023 Kaggle March Machine Learning Mania Challenge](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/overview). Just to clarify, March Madness is a tournament where NCAA college basketball teams compete against each other. This tournament happens annually, and millions of people try to build the perfect bracket: you predict every game perfectly in terms of who wins and who loses. However, the probability of such a bracket being created is 1 in 120.2 billion. 

I wanted to see if I could use my machine learning and data science expertise to my advantage. Furthermore, I wanted to see if using machine learning & data science gives you an edge in terms of predicting who wins each game. Let's take a look at the project goals.

## Project Goals:
__Main Goal:__ Engineer a data science & machine learning backed solution to predict outcomes of games in the NCAA college basketball tournament (for both Men's and Women's).

__SubGoals:__
1. Leverage my solution to submit brackets to various tournaments (ESPN, NCAA, etc).
2. Be able to earn an award in the Kaggle competition (linked above) as it is the basis of this project.

## Problem Details
In this section, I would like to cover what exactly we are building and how we will use it to achieve the main goal. Since the basis of this project is based off of the Kaggle competition, we have the "what" we are building solved. We are building a machine learning system that returns the probability that the team with the lower TeamID (assigned via the data provided) wins. For example, if Rutgers has a TeamID of 1311 and Purdue has a TeamID of 1312, we would want our system to predict P(Rutgers Wins). Likewise, if Purdue has TeamID 1312 and Ohio State has TeamID 1310, we want to predict P(Ohio State Wins). 

Hence, this is a classification problem (supervised learning). The data is provided to us via the Kaggle Competition. We can use this problem to predict game outcomes. Our model would return P(lower TeamID wins) ,and from this we can derive P(higher TeamID wins) = 1 - P(lower TeamID wins). We simply choose the TeamID with the highest probability and get the team name based on provided mappers. We can then use the predictions to build brackets and to submit to the Kaggle Challenge.

Note, the Kaggle Challenge looks at all possible matchups in each tournament since the challenge was released prior to the tournament brackets being made. Also note, the Kaggle Challenge provides us data for both Men's and Women's tournaments. 

One final thing to note is how I am evaluating the model The model is being evaluated using [Brier Score](https://en.wikipedia.org/wiki/Brier_score). This was provided to us via the Kaggle Competition. I am also using [Log-Loss](https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a#:~:text=is%20dependent%20on.-,What%20does%20log%2Dloss%20conceptually%20mean%3F,is%20the%20log%2Dloss%20value.) to compare my model's results to last year's competition. I am also using accuracy because I want to see how accurate my model is. Note, I will not choose my model based off of this.

## Tools, Libraries, and Frameworks Used
- Python
- [Pandas](https://pandas.pydata.org/): library to help leverage and manipulate tables (DataFrames) 
- [Numpy](https://numpy.org/): library that provides lots of the fundamental scientific computing in Python.
- [Matplotlib](https://matplotlib.org/): library that provides tools to create nice visuals in Python.
- [Seaborn](https://seaborn.pydata.org/): library that gives us nice visuals in Python.
- [Scikit-Learn](https://scikit-learn.org/stable/): API that allows us to build machine learning models in Python.
- [Catboost](https://catboost.ai/): library that allows us to leverage the Gradient Boosting on Decision Trees Algorithm.
- [Keras](https://keras.io/): a framework that allows the user to leverage Deep Learning models for various problems. (Note, this version of Keras is using [Tensorflow](https://www.tensorflow.org/) backend).
- [Scikeras](https://www.adriangb.com/scikeras/stable/): A wrapper that allows us to leverage Scikit-Learn functions with Keras deep learning models.
- [Poetry](https://python-poetry.org/): A Python dependency management system.
- [Pickle](https://docs.python.org/3/library/pickle.html): Python Object Serialization, helps with saving models.

## References
- [Predicting college basketball match outcomes using machine learning techniques: some results and lessons learned](https://www.researchgate.net/publication/257749099_Predicting_college_basketball_match_outcomes_using_machine_learning_techniques_some_results_and_lessons_learned): A research paper published in 2013 that provided me with some ideas on features to create and models to try. Also gave me some insight on what some key predictors might be.

## Next Steps
- Build a proper preprocessing pipeline that can automate the preprocessing of the data (box scores to actual model ready data).
- Read more papers to improve our feature set so that we can elevate the model's performance (engineering better features).
- Build a UI/UX so that this can become an actual application with a frontend that a user can use.
- Build a smooth pipeline such that we can send in 2 team names and get a prediction.
