#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:15:26 2018

@author: rjotsin
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from xgboost import XGBRegressor


np.random.seed(1)

views = pd.read_csv(r'training_set.csv')
views = views.pivot(index='Game_ID', columns='Country', values='Rounded Viewers')
views = views
views = views.fillna(0)
#views.to_csv('views.csv')

player = pd.read_csv(r'player_data.csv')

#player['AllStar_Count'] = dfgame.groupby(['Game_ID', 'Team'])['Final_Score']\
#.transform(lambda x: (x!='None')

game = pd.read_csv(r'game_data.csv')
streaks = pd.read_csv(r'streaks.csv')
streaks['result'] = np.where(streaks['result']=='f', 0,1)

game_streaks=pd.merge(game,streaks,left_on=['Game_ID', 'Location'], right_on=['game_id', 'location'] , how='left')

game1 = pd.pivot_table(game_streaks,index=['Game_ID'], columns=['Location'], values=['Game_Date', 'Season', 'Team'], aggfunc=sum)

#game1['Game_ID'] = game1.index
game1.columns = ['Game_Date','Game_Date_2','Season','Season_2','Team_A','Team_H']
game1 = game1.drop('Game_Date_2',1)
game1 = game1.drop('Season_2',1)


game2 = pd.pivot_table(game_streaks,index=['Game_ID'], columns=['Location'], values=['Wins_Entering_Gm', \
                       'Losses_Entering_Gm', 'Team_Minutes', 'Final_Score', 'Lead_Changes', \
                       'Ties', 'Largest_Lead', 'Full_Timeouts','Short_Timeouts', 'Qtr_4_Score', \
                       'L2M_Score','streak_win_before', 'streak_loss_before','result'], aggfunc=np.mean)
#game2['Game_ID'] = game2.index
game2.columns = ['Final_Score_A', 'Final_Score_H', 'Full_Timeouts_A', 'Full_Timeouts_H', 'L2M_Score_A', \
                 'L2M_Score_H', 'Largest_Lead_A', 'Largest_Lead_H', 'Lead_Changes_A', 'Lead_Changes_H', \
                  'Losses_Entering_Gm_A', 'Losses_Entering_Gm_H', 'Qtr_4_Score_A', 'Qtr_4_Score_H', \
                  'Short_Timeouts_A', 'Short_Timeouts_H', 'Team_Minutes_A', 'Team_Minutes_H', 'Ties_A', \
                  'Ties_H', 'Wins_Entering_Gm_A', 'Wins_Entering_Gm_H','result_A','result_H','streak_loss_before_A',\
                 'streak_loss_before_H', 'streak_win_before_A','streak_win_before_H']

game2['max_win'] = game2[['streak_win_before_A','streak_win_before_H']].max(axis=1).fillna(0)
game2['max_loss'] = game2[['streak_loss_before_A','streak_loss_before_H']].max(axis=1).fillna(0)
game2 = game2.drop(['streak_loss_before_A','streak_loss_before_H', 'streak_win_before_A','streak_win_before_H'],1)
        
dfgame=pd.merge(game1,game2,on='Game_ID', how='inner')

dfgame['Game_Date'] = pd.to_datetime(dfgame['Game_Date'])
dfyear = pd.get_dummies('y' + dfgame['Game_Date'].dt.strftime('%y'))
dfmonth = pd.get_dummies('m' + dfgame['Game_Date'].dt.strftime('%m'))
dfday = pd.get_dummies('d' + dfgame['Game_Date'].dt.dayofweek.apply(str))
#dfwin = pd.get_dummies('w' + dfgame['max_win'].apply(str))
#dfloss = pd.get_dummies('l' + dfgame['max_loss'].apply(str))
dfgame['Season_Num'] = np.where(dfgame['Season']=='2016-17', 0,1)

dfgame['Team_A'] = dfgame['Team_A'] + '_A'
dfgame['Team_H'] = dfgame['Team_H'] + '_H'


dfTeamA = pd.get_dummies(dfgame['Team_A'])
dfTeamH = pd.get_dummies(dfgame['Team_H'])
dfgame = dfgame.join(dfTeamA)
dfgame = dfgame.join(dfTeamH)
dfgame = dfgame.drop('Team_A',1)
dfgame = dfgame.drop('Team_H',1)
#dfgame = dfgame.join(dfwin)
#dfgame = dfgame.join(dfloss)
#dfgame = dfgame.drop('max_win',1)
#dfgame = dfgame.drop('max_loss',1)
dfgame = dfgame.join(dfyear)
dfgame = dfgame.join(dfmonth)
dfgame = dfgame.join(dfday)
dfgame.to_csv('gametst.csv')
gp =dfgame.columns[29:89].tolist()
gp_A =dfgame.columns[28:59].tolist() + dfgame.columns[89:106].tolist()
gp_H =dfgame.columns[28:29].tolist() + dfgame.columns[59:106].tolist()
gp_A2 =dfgame.columns[28:59].tolist() + dfgame.columns[89:99].tolist()
gp_H2 =dfgame.columns[28:29].tolist() + dfgame.columns[59:99].tolist() #+ dfgame.columns[127:137].tolist()
gp_for = game.columns[7:].tolist()

for i in gp_for:    
    dfgame[i + '_A'] = dfgame.groupby(gp)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp)[i + '_H'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_A'] = dfgame.groupby(gp_A)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp_H)[i + '_H'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_A'] = dfgame.groupby(gp_A2)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp_H2)[i + '_H'].transform(lambda x: x.fillna(x.mean()))

dfgame.to_csv('gametst.csv')


dfTeam = dfgame.iloc[:,29:89].T.groupby([s.split('_')[0] for s in dfgame.iloc[:,29:89].T.index.values]).sum().T

df = pd.merge(dfgame.iloc[:,:29], dfTeam, on='Game_ID', how='inner')
df2 = pd.merge(df, dfgame.iloc[:,89:], on='Game_ID', how='inner')
df3 = pd.merge(df2, views, on='Game_ID', how='inner')
df3.rename(columns={'Team_Minutes_A':'Team_Minutes'}, inplace=True)
df3 = df3.drop('Team_Minutes_H',1)

#df3.to_csv('output.csv')

train_cols = df3.columns[2:75] #99-31 = 68 + 7 = 75
views_for = views.columns.tolist()

#cols = {}
#
#from sklearn import datasets
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression()
#rfe = RFE(logreg, 20)
#for k in views_for:
#    cols['rfe' + k] = rfe.fit(df3[train_cols], df3[k])
#
#
#
#train_cols2 = df3.columns[2:27].union(df3.columns[28:68])
#train_cols3 = df3.columns[2:58]
#train_cols4 = df3.columns[12:58]

views_test = pd.read_csv(r'test_set.csv')
test2 = pd.merge(df2, views_test, on='Game_ID', how='inner')
test2.rename(columns={'Team_Minutes_A':'Team_Minutes'}, inplace=True)
test2 = test2.drop('Team_Minutes_H',1)

test_country = pd.DataFrame()
test_country['Game_ID'] = views_test['Game_ID']

#dtrain = xg.DMatrix(X_train, label=y_train)
#dtest = xg.DMatrix(X_test, label=y_test)

model = XGBRegressor()
result = {}

for i in views_for:
    model.fit(df3[train_cols],df3[i])
    result[i] = model.predict(test2[train_cols])

test_country2= pd.DataFrame(result)
test_country2['Game_ID'] = views_test['Game_ID']
test_country2.to_csv('test_country2.csv')


#for i in views_for:
#    dtrain = xg.DMatrix(df3[train_cols],label = df3[i])
#for j in views_for:
#    dtest =xg.DMatrix(test2[train_cols],label = df3[j])
    
#param = {
#    'max_depth': 3,  # the maximum depth of each tree
#    'eta': 0.3,  # the training step for each iteration
#    'silent': 1,  # logging mode - quiet
#    'objective': 'multi:softprob',  # error evaluation for multiclass training
#    'num_class': 3}  # the number of classes that exist in this datset
#num_round = 100  # the number of training iterations

#    try:
#        df4 = df3.loc[df3[i] != 0]
#        train_for = df4[train_cols].apply(np.log)
#        result['r_' + i] = train_for.columns[:]
#        logit['l_' + i] = sm.Logit(df3[i], df3[train_for.columns[:]]).fit()
#    except:
#        try:
#            df4 = df3.loc[df3[i] != 0]
#            df4 = df4[train_cols2]
#            train_for = df4.loc[:, (df4 != 0).any(axis=0)]
#            result['r_' + i] = train_for.columns[:]
#            logit['l_' + i] = sm.Logit(df3[i], df3[train_for.columns[:]]).fit()
#        except:
#            try:
#                df4 = df3.loc[df3[i] != 0]
#                df4 = df4[train_cols3]
#                train_for = df4.loc[:, (df4 != 0).any(axis=0)]
#                result['r_' + i] = train_for.columns[:]
#                logit['l_' + i] = sm.Logit(df3[i], df3[train_for.columns[:]]).fit()
#            except:
#                df4 = df3.loc[df3[i] != 0]
#                df4 = df4[train_cols4]
#                train_for = df4.loc[:, (df4 != 0).any(axis=0)]
#                result['r_' + i] = train_for.columns[:]
#                logit['l_' + i] = sm.Logit(df3[i], df3[train_for.columns[:]]).fit()
#    
#result = {}
#bst = xg.train(param, dtrain, num_round)
#
#test_country['Total_Viewers'] = test_country.sum(axis=1)
#test_country['Total_Viewers'] = test_country['Total_Viewers'] - test_country['Game_ID']
#
#test2.to_csv('test2.csv')
#test_country.to_csv('test_country.csv')