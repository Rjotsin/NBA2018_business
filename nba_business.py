#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:15:26 2018

@author: rjotsin
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np

np.random.seed(1)

views = pd.read_csv(r'training_set.csv')
views = views.pivot(index='Game_ID', columns='Country', values='Rounded Viewers')
views = views/100000
views = views.fillna(0)
#views.to_csv('views.csv')

player = pd.read_csv(r'player_data.csv')

#player['AllStar_Count'] = dfgame.groupby(['Game_ID', 'Team'])['Final_Score']\
#.transform(lambda x: (x!='None')

game = pd.read_csv(r'game_data.csv')

game1 = pd.pivot_table(game,index=['Game_ID'], columns=['Location'], values=['Game_Date', 'Season', 'Team'], aggfunc=sum)

#game1['Game_ID'] = game1.index
game1.columns = ['Game_Date','Game_Date_2','Season','Season_2','Team_A','Team_H']
game1 = game1.drop('Game_Date_2',1)
game1 = game1.drop('Season_2',1)
#,'Game_ID'

game2 = pd.pivot_table(game,index=['Game_ID'], columns=['Location'], values=['Wins_Entering_Gm', \
                       'Losses_Entering_Gm', 'Team_Minutes', 'Final_Score', 'Lead_Changes', \
                       'Ties', 'Largest_Lead', 'Full_Timeouts','Short_Timeouts', 'Qtr_4_Score', \
                       'L2M_Score'], aggfunc=np.mean)
#game2['Game_ID'] = game2.index
game2.columns = ['Final_Score_A', 'Final_Score_H', 'Full_Timeouts_A', 'Full_Timeouts_H', 'L2M_Score_A', \
                 'L2M_Score_H', 'Largest_Lead_A', 'Largest_Lead_H', 'Lead_Changes_A', 'Lead_Changes_H', \
                  'Losses_Entering_Gm_A', 'Losses_Entering_Gm_H', 'Qtr_4_Score_A', 'Qtr_4_Score_H', \
                  'Short_Timeouts_A', 'Short_Timeouts_H', 'Team_Minutes_A', 'Team_Minutes_H', 'Ties_A', \
                  'Ties_H', 'Wins_Entering_Gm_A', 'Wins_Entering_Gm_H']

dfgame=pd.merge(game1,game2,on='Game_ID', how='inner')

dfgame['Game_Date'] = pd.to_datetime(dfgame['Game_Date'])
dfyear = pd.get_dummies('y' + dfgame['Game_Date'].dt.strftime('%y'))
dfmonth = pd.get_dummies('m' + dfgame['Game_Date'].dt.strftime('%m'))
dfday = pd.get_dummies('d' + dfgame['Game_Date'].dt.strftime('%d'))
dfgame['Season_Num'] = np.where(dfgame['Season']=='2016-17', 0,1)

dfgame['Team_A'] = dfgame['Team_A'] + '_A'
dfgame['Team_H'] = dfgame['Team_H'] + '_H'


dfTeamA = pd.get_dummies(dfgame['Team_A'])
dfTeamH = pd.get_dummies(dfgame['Team_H'])
#dfLocation = pd.get_dummies(game['Location'])
dfgame = dfgame.join(dfTeamA)
dfgame = dfgame.join(dfTeamH)
#dfgame = dfgame.join(dfLocation)
dfgame = dfgame.drop('Team_A',1)
dfgame = dfgame.drop('Team_H',1)
#dfgame = dfgame.drop('Location',1)
dfgame = dfgame.join(dfyear)
dfgame = dfgame.join(dfmonth)
dfgame = dfgame.join(dfday)
dfgame.to_csv('gametst.csv')
gp =dfgame.columns[25:85].tolist()
gp_A =dfgame.columns[24:55].tolist() + dfgame.columns[85:95].tolist()
gp_H =dfgame.columns[24:25].tolist() + dfgame.columns[55:95].tolist()
gp_for = game.columns[7:].tolist()

for i in gp_for:    
    dfgame[i + '_A'] = dfgame.groupby(gp)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp)[i + '_H'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_A'] = dfgame.groupby(gp_A)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp_H)[i + '_H'].transform(lambda x: x.fillna(x.mean()))

dfgame.to_csv('gametst.csv')


#dfTeam = pd.get_dummies(game['Team'])
#dfLocation = pd.get_dummies(game['Location'])
#
#dfgame = game.join(dfTeam)
#dfgame = dfgame.join(dfLocation)
#dfgame = dfgame.drop('Team',1)
#dfgame = dfgame.drop('Location',1)



dfTeam = dfgame.iloc[:,25:85].T.groupby([s.split('_')[0] for s in dfgame.iloc[:,25:85].T.index.values]).sum().T

df = pd.merge(dfgame.iloc[:,:25], dfTeam, on='Game_ID', how='inner')
df2 = pd.merge(df, dfgame.iloc[:,85:], on='Game_ID', how='inner')
df3 = pd.merge(df2, views, on='Game_ID', how='inner')
df3.rename(columns={'Team_Minutes_A':'Team_Minutes'}, inplace=True)
df3 = df3.drop('Team_Minutes_H',1)

#df3.to_csv('output.csv')

train_cols = df3.columns[2:95] #64
views_for = views.columns.tolist()

train_cols2 = df3.columns[2:23].union(df3.columns[24:64])
train_cols3 = df3.columns[2:54]
train_cols4 = df3.columns[12:54]
train_cols5 = df3.columns[24:64]

#df4 = df3.loc[df3['C1'] != 0]
#df4 = df4[train_cols2]
#train_for = df4.loc[:, (df4 != 0).any(axis=0)]
#logit = sm.Logit(df3['C1'], df3[train_for.columns[:]])
#result = logit.fit()

logit={}
result = {}
for i in views_for:
    try:
        df4 = df3.loc[df3[i] != 0]
        df4 = df4[train_cols]
        train_for = df4.loc[:, (df4 != 0).any(axis=0)]
        result['r_' + i] = train_for.columns[:]
        logit['l_' + i] = sm.OLS(df3[i], df3[train_for.columns[:]]).fit()
    except:
        pass
#        try:
#            df4 = df3.loc[df3[i] != 0]
#            df4 = df4[train_cols2]
#            train_for = df4.loc[:, (df4 != 0).any(axis=0)]
#            result['r_' + i] = train_for.columns[:]
#            logit['l_' + i] = sm.OLS(df3[i], df3[train_for.columns[:]]).fit()
#        except:
#            try:
#                df4 = df3.loc[df3[i] != 0]
#                df4 = df4[train_cols3]
#                train_for = df4.loc[:, (df4 != 0).any(axis=0)]
#                result['r_' + i] = train_for.columns[:]
#                logit['l_' + i] = sm.OLS(df3[i], df3[train_for.columns[:]]).fit()
#            except:
#                df4 = df3.loc[df3[i] != 0]
#                df4 = df4[train_cols4]
#                train_for = df4.loc[:, (df4 != 0).any(axis=0)]
#                result['r_' + i] = train_for.columns[:]
#                logit['l_' + i] = sm.OLS(df3[i], df3[train_for.columns[:]]).fit()
    
#res = logit['l_C1'].fit()
    
views_test = pd.read_csv(r'test_set.csv')
test2 = pd.merge(df2, views_test, on='Game_ID', how='inner')
test2.rename(columns={'Team_Minutes_A':'Team_Minutes'}, inplace=True)
test2 = test2.drop('Team_Minutes_H',1)

test_country = pd.DataFrame()
test_country['Game_ID'] = views_test['Game_ID']
for j in views_for:
    test_country[j] =logit['l_' + j].predict(test2[result['r_' + j]])
test_country['Total_Viewers'] = test_country.sum(axis=1)
test_country['Total_Viewers'] = test_country['Total_Viewers'] - test_country['Game_ID']

test2.to_csv('test2.csv')
test_country.to_csv('test_country.csv')

    


#result.summary()
#result.conf_int()

#game = game.groupby(['Game_ID', 'Game_Date'])
#game.to_csv('game.csv')



