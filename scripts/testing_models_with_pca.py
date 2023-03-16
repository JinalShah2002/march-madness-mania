"""

author: Jinal Shah


In this file, I will test various models
with PCA. I suspect that the vast amount of 
features causes some errors. I experiment with how
PCA may improve or worsen the models. 
I did this in a notebook,
but it became very messy. Let's do it 
in a script for cleaniness.

Metrics:
- Brier Score => main one competition is monitoring
- Accuracy => Just to see how accurate I am
- Log Loss => See how the models stack up to last year's top scorers.

"""
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
from catboost import CatBoostClassifier, Pool, cv
import tensorflow as tf
from tensorflow import keras
from keras import layers
from scikeras.wrappers import KerasClassifier
from sklearn.decomposition import PCA
import pickle

# Importing the data
training_data = pd.read_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/preprocessed-data/prediction-ready-data/training.csv',index_col=0)

# Splitting data up into features and target
features = training_data.drop(['189'],axis=1).values
target = training_data['189'].values

# Running the data through PCA
pca = PCA(n_components=75,random_state=42)
features_reduced = pca.fit_transform(features,target)

# Creating a training pool for catboost
training_pool = Pool(data=features_reduced,label=target)

# Creating a dictionary that holds various metrics
metrics = {'Model':[],'Log Loss Train':[],'Log Loss CV':[],'Brier Score Training':[],
           'Mean Brier Score CV':[], 'Accuracy Training':[],'Mean Accuracy CV':[]}

# Creating a function for updating
def update_metrics(model, name:str, catboost_model:bool, catboost_params:list=None):
    # Fitting the model to the data
    model.fit(features_reduced,target)

    # Getting training scores
    if not catboost_model:
        train_pred = model.predict(features_reduced)
        train_probs = model.predict_proba(features_reduced)[:,1]
    else:
        train_pred = model.predict(features_reduced,prediction_type='Class',ntree_start=0,ntree_end=model.tree_count_ - 1)
        train_probs = model.predict_proba(features_reduced,ntree_start=0,ntree_end=model.tree_count_ - 1)[:,1]

    # Getting the metrics
    train_brier = brier_score_loss(target,train_probs)
    train_accuracy = accuracy_score(target,train_pred)
    train_loss = log_loss(target,train_probs)

    # Adding metrics to the dictionary
    model_list = metrics['Model']
    model_list.append(name)
    metrics['Model'] = model_list
    train_briers = metrics['Brier Score Training']
    train_briers.append(train_brier)
    metrics['Brier Score Training'] = train_briers
    train_accuracies = metrics['Accuracy Training']
    train_accuracies.append(train_accuracy)
    metrics['Accuracy Training'] = train_accuracies
    loss = metrics['Log Loss Train']
    loss.append(train_loss)
    metrics['Log Loss Train'] = loss

    # Getting the results and adding it to the dictionary
    if not catboost_model:
        results = cross_validate(model,features_reduced,target,scoring=['accuracy','neg_brier_score','neg_log_loss'],
                                cv=5,n_jobs=-1)

        cv_briers = metrics['Mean Brier Score CV']
        cv_briers.append((-1*results['test_neg_brier_score']).mean())
        metrics['Mean Brier Score CV'] = cv_briers
        cv_accuracies = metrics['Mean Accuracy CV']
        cv_accuracies.append((results['test_accuracy']).mean())
        metrics['Mean Accuracy CV'] = cv_accuracies
        cv_log_loss = metrics['Log Loss CV']
        cv_log_loss.append((-1*results['test_neg_log_loss']).mean())
        metrics['Log Loss CV'] = cv_log_loss

        # Saving model in a pickle file
        pickle.dump(model,open(f'/Users/jinalshah/Jinal/Projects/march-madness-mania/models/{name}.sav','wb'))
    else:
        scores = cv(training_pool,catboost_params,fold_count=5)
        cv_briers = metrics['Mean Brier Score CV']
        cv_briers.append(scores.loc[model.tree_count_-1,'test-BrierScore-mean'])
        metrics['Mean Brier Score CV'] = cv_briers
        cv_accuracies = metrics['Mean Accuracy CV']
        cv_accuracies.append(scores.loc[model.tree_count_-1,'test-Accuracy-mean'])
        metrics['Mean Accuracy CV'] = cv_accuracies
        cv_loss = metrics['Log Loss CV']
        cv_loss.append(scores.loc[model.tree_count_-1,'test-Logloss-mean'])
        metrics['Log Loss CV'] = cv_loss
        model.save_model(f'/Users/jinalshah/Jinal/Projects/march-madness-mania/models/{name}.cbm')

"""

Creating the Models


"""
dummy_classifier = DummyClassifier(strategy='uniform',random_state=42) # Dummy Classifier, will randomly predict this is the baseline
logistic_reg = LogisticRegression(penalty=None,C=1.0,random_state=42,max_iter=1000,n_jobs=-1) # vanilla logistic regression
decision_tree = DecisionTreeClassifier(criterion='gini',random_state=42) # vanilla decision tree
random_forest = RandomForestClassifier(n_estimators=1000,criterion='gini',bootstrap=True,n_jobs=-1,random_state=42,max_samples=0.7)
adaboost_logistic = AdaBoostClassifier(estimator=logistic_reg,n_estimators=500,learning_rate=1.5,random_state=42)
adaboost_decision_tree = AdaBoostClassifier(estimator=decision_tree,n_estimators=500,learning_rate=1.5,random_state=42)
elastic_net = LogisticRegression(penalty='elasticnet',solver='saga',random_state=42,max_iter=5000,n_jobs=-1,l1_ratio=0.7)
log_forest = VotingClassifier(estimators=[('lr',LogisticRegression(penalty=None,C=1.0,random_state=42,max_iter=1000,n_jobs=-1)),
                                         ('forest',RandomForestClassifier(n_estimators=1000,criterion='gini',bootstrap=True,n_jobs=-1,random_state=42,max_samples=0.7))],
                                         voting='soft',n_jobs=-1)
catboost_clf = CatBoostClassifier(iterations=1000,learning_rate=0.05,loss_function='Logloss',random_seed=42,verbose=False,
                                  early_stopping_rounds=10)
catboost_params = {'iterations':1000,'learning_rate':0.05,'loss_function':'Logloss','random_seed':42,'verbose':False,
                   'custom_metric':['BrierScore','Accuracy']}

# Neural Network
# Building the neural network
neural_net = keras.Sequential([
    keras.Input(shape=(features_reduced.shape[1])), # input layer
    layers.Dense(100,activation='relu'),
    layers.Dense(50,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(25,activation='relu'),
    layers.Dense(1,activation='softmax'),
])
neural_net_wrapped = KerasClassifier(model=neural_net,optimizer='adam',loss='binary_crossentropy',random_state=42,batch_size=32,
                                     optimizer__learning_rate=0.03,epochs=500,shuffle=True)

# Sending each classifier into the function
update_metrics(dummy_classifier,'Dummy-Classifier-PCA',False)
print('Finished Dummy Classifier-PCA')
update_metrics(logistic_reg,'Vanilla-Logistic-Regression-PCA',False)
print('Finished Vanilla-Logistic-Regression-PCA')
update_metrics(decision_tree,'Decision-Tree-PCA',False)
print('Finished Decision-Tree-PCA')
update_metrics(random_forest,'Random-Forest-PCA',False)
print('Finished Random-Forest-PCA')
update_metrics(adaboost_logistic,'AdaBoost-Logistic-PCA',False)
print('Finished AdaBoost-Logistic-PCA')
update_metrics(adaboost_decision_tree,'AdaBoost-Decision-Tree-PCA',False)
print('Finished AdaBoost-Decision-Tree-PCA')
update_metrics(elastic_net,'Logistic-Regression-Elastic-PCA',False)
print('Finished Logistic-Regression-Elastic-PCA')
update_metrics(log_forest,'Logistic-and-RF-PCA',False)
print('Finished Logistic-and-RF-PCA')
update_metrics(catboost_clf,'Vanilla-Catboost-PCA',True,catboost_params)
print('Finished Vanilla-Catboost-PCA')
update_metrics(neural_net_wrapped,'Neural Network-PCA',False)
print('Finished Neural Network-PCA')

# Converting metrics to a dataframe
metrics_df = pd.DataFrame(metrics)

# Saving the dataframe as a csv for later analysis
metrics_df.to_csv('/Users/jinalshah/Jinal/Projects/march-madness-mania/model-performances/metrics_pca.csv')

print(metrics_df)