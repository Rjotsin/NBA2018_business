import pandas as pd
import pandasql as ps
import numpy as np
from xgboost import XGBRegressor

np.random.seed(1)

#Import and Pivot Training Data
views = pd.read_csv(r'training_set.csv')
views = views.pivot(index='Game_ID', columns='Country', values='Rounded Viewers')
views = views.fillna(0)

# Import Game, Streaks, and Pace Data
game = pd.read_csv(r'game_data.csv')
streaks = pd.read_csv(r'streaks.csv')
streaks['result'] = np.where(streaks['result']=='f', 0,1)
pacesql = """select Team, avg(q4pace) as q4pace, avg(l2mpace) as l2mpace
from (
select *, Final_Score - Qtr_4_Score as q4pace, Final_Score - L2M_Score as l2mpace
from game
  )
group by 1"""
pace = ps.sqldf(pacesql, locals())

#Merge Game and Streaks
game_streaks=pd.merge(game,streaks,left_on=['Game_ID', 'Location'], right_on=['game_id', 'location'] , how='left')

#Merge rows with Home and Away Status
game1 = pd.pivot_table(game_streaks,index=['Game_ID'], columns=['Location'], values=['Game_Date', 'Season', 'Team'], aggfunc=sum)
game1.columns = ['Game_Date','Game_Date_2','Season','Season_2','Team_A','Team_H']
game1 = game1.drop('Game_Date_2',1)
game1 = game1.drop('Season_2',1)

#Import and Merge Timezones and All Stars per game
timezone = pd.read_csv(r'timezones.csv') 
allstar = pd.read_csv(r'asg.csv')
game1['Game_ID'] = game1.index
game1=pd.merge(game1,timezone,left_on='Team_H', right_on='Team', how='left')
game1 = game1.drop('Team',1)
game1=pd.merge(game1,allstar,left_on='Game_ID', right_on='game_id', how='left')
game1 = game1.drop('game_id',1)

#Merge Pace
game_streaks=pd.merge(game_streaks,pace,on='Team', how='left')
points = pd.read_csv(r'pts.csv')
game_streaks=pd.merge(game_streaks,points, on='game_id', how='left')

#Pivot and assign home and away stats
game2 = pd.pivot_table(game_streaks,index=['Game_ID'], columns=['Location'], values=['Wins_Entering_Gm', \
                       'Losses_Entering_Gm', 'Team_Minutes', 'Final_Score', 'Lead_Changes', \
                       'Ties', 'Largest_Lead', 'Full_Timeouts','Short_Timeouts', 'Qtr_4_Score', \
                       'L2M_Score','streak_win_before', 'streak_loss_before','result', 'q4pace', \
                       'l2mpace', 'points'], aggfunc=np.mean)
game2.columns = ['Final_Score_A', 'Final_Score_H', 'Full_Timeouts_A', 'Full_Timeouts_H', 'L2M_Score_A',\
                 'L2M_Score_H', 'Largest_Lead_A', 'Largest_Lead_H', 'Lead_Changes_A', 'Lead_Changes_H',\
                  'Losses_Entering_Gm_A', 'Losses_Entering_Gm_H', 'Qtr_4_Score_A', 'Qtr_4_Score_H',\
                  'Short_Timeouts_A', 'Short_Timeouts_H', 'Team_Minutes_A', 'Team_Minutes_H', 'Ties_A',\
                  'Ties_H', 'Wins_Entering_Gm_A', 'Wins_Entering_Gm_H', 'l2mpace_A', 'l2mpace_H', 'points_A', \
                  'points_H', 'q4pace_A', 'q4pace_H', 'result_A','result_H','streak_loss_before_A', \
                  'streak_loss_before_H', 'streak_win_before_A','streak_win_before_H']

#Replace streaks with Max and Min Streak
game2['max_win'] = game2[['streak_win_before_A','streak_win_before_H']].max(axis=1).fillna(0)
game2['max_loss'] = game2[['streak_loss_before_A','streak_loss_before_H']].max(axis=1).fillna(0)
game2 = game2.drop(['streak_loss_before_A','streak_loss_before_H', 'streak_win_before_A','streak_win_before_H'],1)

#Merge all data        
dfgame=pd.merge(game1,game2,on='Game_ID', how='inner')

#Make dummy variables with Dates, Season Number, Teams, and Timezone
#Drop normal columns
dfgame['Game_Date'] = pd.to_datetime(dfgame['Game_Date'])
dfyear = pd.get_dummies('y' + dfgame['Game_Date'].dt.strftime('%y'))
dfmonth = pd.get_dummies('m' + dfgame['Game_Date'].dt.strftime('%m'))
dfday = pd.get_dummies('d' + dfgame['Game_Date'].dt.dayofweek.apply(str))
dfgame['Season_Num'] = np.where(dfgame['Season']=='2016-17', 0,1)
dfgame['Team_A'] = dfgame['Team_A'] + '_A'
dfgame['Team_H'] = dfgame['Team_H'] + '_H'
dfTeamA = pd.get_dummies(dfgame['Team_A'])
dfTeamH = pd.get_dummies(dfgame['Team_H'])
dftimezones = pd.get_dummies(dfgame['Timezone'])
dfallstar = pd.get_dummies('as' + dfgame['sum'].apply(str))
dfgame = dfgame.join(dfTeamA)
dfgame = dfgame.join(dfTeamH)
dfgame = dfgame.join(dftimezones)
dfgame = dfgame.join(dfallstar)
dfgame = dfgame.drop('Team_A',1)
dfgame = dfgame.drop('Team_H',1)
dfgame = dfgame.drop('Timezone',1)
dfgame = dfgame.drop('sum',1)
dfgame = dfgame.join(dfyear)
dfgame = dfgame.join(dfmonth)
dfgame = dfgame.join(dfday)

#Create Impute Data
gp =dfgame.columns[36:96].tolist()
gp_A =dfgame.columns[35:66].tolist() + dfgame.columns[96:125].tolist()
gp_H =dfgame.columns[35:36].tolist() + dfgame.columns[66:125].tolist()
gp_A2 =dfgame.columns[35:66].tolist() + dfgame.columns[96:118].tolist()
gp_H2 =dfgame.columns[35:36].tolist() + dfgame.columns[66:118].tolist()
gp_A3 =dfgame.columns[35:66].tolist() + dfgame.columns[100:118].tolist()
gp_H3 =dfgame.columns[35:36].tolist() + dfgame.columns[66:96].tolist() + dfgame.columns[108:118].tolist()
gp_for = game.columns[7:].tolist()

#Impute test data loop
for i in gp_for:    
    dfgame[i + '_A'] = dfgame.groupby(gp)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp)[i + '_H'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_A'] = dfgame.groupby(gp_A)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp_H)[i + '_H'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_A'] = dfgame.groupby(gp_A2)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp_H2)[i + '_H'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_A'] = dfgame.groupby(gp_A3)[i + '_A'].transform(lambda x: x.fillna(x.mean()))
    dfgame[i + '_H'] = dfgame.groupby(gp_H3)[i + '_H'].transform(lambda x: x.fillna(x.mean()))

#Test check
dfgame.to_csv('gametst.csv')

#Remove H and A labels for teams
dfTeam = dfgame.iloc[:,36:96].T.groupby([s.split('_')[0] for s in dfgame.iloc[:,36:96].T.index.values]).sum().T
dfTeam['Game_ID'] = dfgame['Game_ID']
dates = dfgame.iloc[:,96:]
dates['Game_ID'] = dfgame['Game_ID']
df = pd.merge(dfgame.iloc[:,:36], dfTeam, on='Game_ID', how='inner')
#Merge Date Stats
df2 = pd.merge(df, dates, on='Game_ID', how='inner')
#Merge viewership
df3 = pd.merge(df2, views, on='Game_ID', how='inner')
#Clean up on Team Minutes
df3.rename(columns={'Team_Minutes_A':'Team_Minutes'}, inplace=True)
df3 = df3.drop('Team_Minutes_H',1)
df3.rename(columns={'points_A':'points'}, inplace=True)
df3 = df3.drop('points_H',1)


#Assign Columns to Train on
train_cols = df3.columns[3:93] #99-31 = 77 + 7 = 84
#Assign Columns to Predict
views_for = views.columns.tolist()

#Import Test Data and Merge imported data
views_test = pd.read_csv(r'test_set.csv')
test2 = pd.merge(df2, views_test, on='Game_ID', how='inner')
test2.rename(columns={'Team_Minutes_A':'Team_Minutes'}, inplace=True)
test2 = test2.drop('Team_Minutes_H',1)
test2.rename(columns={'points_A':'points'}, inplace=True)
test2 = test2.drop('points_H',1)

#Create XG model
model = XGBRegressor()
result = {}
#Training loop
for i in views_for:
    model.fit(df3[train_cols],df3[i])
    result[i] = model.predict(test2[train_cols])

#Assign predictions to csv
test_country= pd.DataFrame(result)
test_country['Total']= test_country.sum(axis=1)
test_country['Game_ID'] = views_test['Game_ID']
test_country.to_csv('test_country.csv')