#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:15:26 2018

@author: rjotsin
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
    
views = pd.read_csv(r'training_set.csv')
views = views.pivot(index='Game_ID', columns='Country', values='Rounded Viewers')
views['C176'] = views['C176']/100000
#views.to_csv('views.csv')

game = pd.read_csv(r'game_data.csv')
game['Game_Date'] = pd.to_datetime(game['Game_Date'])
dfyear = pd.get_dummies('y' + game['Game_Date'].dt.strftime('%y'))
dfmonth = pd.get_dummies('m' + game['Game_Date'].dt.strftime('%m'))
dfday = pd.get_dummies('d' + game['Game_Date'].dt.strftime('%d'))
game['Season_Num'] = np.where(game['Season']=='2016-17', 0,1)
dfTeam = pd.get_dummies(game['Team'])
dfLocation = pd.get_dummies(game['Location'])
dfgame = game.join(dfTeam)
dfgame = dfgame.join(dfLocation)
dfgame = dfgame.drop('Team',1)
dfgame = dfgame.drop('Location',1)
dfgame = dfgame.join(dfyear)
dfgame = dfgame.join(dfmonth)
dfgame = dfgame.join(dfday)
gp =dfgame.columns[14:57].tolist()


dfgame['Final_Score'] = dfgame.groupby(gp)['Final_Score'].transform(lambda x: x.fillna(x.mean()))
dfgame.to_csv('gametst.csv')


#dfTeam = pd.get_dummies(game['Team'])
#dfLocation = pd.get_dummies(game['Location'])
#
#dfgame = game.join(dfTeam)
#dfgame = dfgame.join(dfLocation)
#dfgame = dfgame.drop('Team',1)
#dfgame = dfgame.drop('Location',1)


df = pd.merge(dfgame, views, on='Game_ID', how='inner')
#df.to_csv('output.csv')

train_cols = df.columns[3:88]

logit = sm.Logit(df['C176'], df[train_cols])

# fit the model
result = logit.fit()

#result.summary()
#result.conf_int()

#game = game.groupby(['Game_ID', 'Game_Date'])
#game.to_csv('game.csv')



