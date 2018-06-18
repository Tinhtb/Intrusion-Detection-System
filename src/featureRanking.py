'''
@author: TinhTB

Perform feature ranking process which combine three algorithms: ANOVA F-value, ExtraTrees and LightGBM

'''

import util
import classifier
from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif



#Loading the training data including both benigh and attack
#Benign data of each day
benignMon=util.pklReader('Mon-Benign',path=util.getResourcePath() +'/Pickle Files/57 Features/Benign/')
benignTues=util.pklReader('Tues-Benign',path=util.getResourcePath() +'/Pickle Files/57 Features/Benign/')
benignWed=util.pklReader('Wed-Benign',path=util.getResourcePath() +'/Pickle Files/57 Features/Benign/')
benignThursMor=util.pklReader('ThursMor-Benign',path=util.getResourcePath() +'/Pickle Files/57 Features/Benign/')
benignThursAft=util.pklReader('ThursAft-Benign',path=util.getResourcePath() +'/Pickle Files/57 Features/Benign/')
benignFriMor=util.pklReader('FriMor-Benign',path=util.getResourcePath() +'/Pickle Files/57 Features/Benign/')
benignFriDos=util.pklReader('FriDos-Benign',path=util.getResourcePath() +'/Pickle Files/57 Features/Benign/')
benignFriPort=util.pklReader('FriPort-Benign',path=util.getResourcePath() +'/Pickle Files/57 Features/Benign/')
benign={'mon': benignMon, 'tues':benignTues, 'wed':benignWed, 'thursMor': benignThursMor, 'thursAft':benignThursAft,
         'friMor':benignFriMor, 'friPort': benignFriPort, 'friDos': benignFriDos}


##Loading attack data
#Tuesday attack
bruteForce=util.pklReader('Brute Force',path=util.getResourcePath() +'/Pickle Files/57 Features/Attacks/')

#Wednesday attack
dos=util.pklReader('DoS',path=util.getResourcePath() +'/Pickle Files/57 Features/Attacks/')

#Thursday morning attack
web=util.pklReader('Web attack',path=util.getResourcePath() +'/Pickle Files/57 Features/Attacks/')

#Friday attack
bot=util.pklReader('Bot',path=util.getResourcePath() +'/Pickle Files/57 Features/Attacks/')
portScan=util.pklReader('PortScan',path=util.getResourcePath() +'/Pickle Files/57 Features/Attacks/')
ddos=util.pklReader('DDoS',path=util.getResourcePath() +'/Pickle Files/57 Features/Attacks/')
#attack=[ftp]+[ssh]+[dosHulk]+[dosGold]+[dosSlowLoris]+[dosSlowHttp]+[web]+[bot]+[portScan]+[ddos]
allAttack={'bruteForce': bruteForce, 'dos':dos, 'web': web, 'bot': bot, 'portScan': portScan, 'ddos': ddos}


def getRanking(train_data, filename):
    # Examining feature importance by excluding three common features
    train_data=train_data.drop([' Source Port', ' Destination Port', ' Protocol'],axis=1)
    
    #Seperating observation and label
    xData=train_data.drop([' Label'],axis=1)
    y=train_data[' Label'].values
    
    #Standardise the value of observations
    X_std = StandardScaler().fit_transform(xData)
    # feature extraction
    filterModel = SelectKBest(score_func=f_classif)
    model = filterModel.fit(X_std, y)
    
    # Construct Extra tree
    extraTree=classifier.extraTrees(train_data)
    # Construct gradient boosting
    gradientModel=classifier.lightGBM_model(train_data,customEval=True)
    
    
    #anova_imp= MinMaxScaler().fit_transform(np.array([model.scores_]).T).T
    # et_imp = MinMaxScaler().fit_transform(np.array([extraTree2.feature_importances_]).T).T
    # gd_imp = MinMaxScaler().fit_transform(np.array([gradientModel2.feature_importance()]).T).T
    
    
    # Save the results of threes algorithm to a data frame
    order=pd.DataFrame({'ANOVA F-value':model.scores_, 'Extra Tree': extraTree.feature_importances_, 'Gradient Boosting' :gradientModel.feature_importance()},index=xData.columns)
    order=order.fillna(0)
    
    #Rescale the value of each algorithm in the range [0, 1]
    orderScaled=MinMaxScaler().fit_transform(order)
    orderScaled=pd.DataFrame(orderScaled,index=order.index,columns=order.columns)
    
    #Adding one more column to sum the value of three algorithms
    orderScaled['Total']=orderScaled.sum(axis=1)
    orderScaled=orderScaled.round(3)
    
    #Save the result to a pickle file and sort the sum value from the highest to the lowest
    util.pklSaver(orderScaled,filename,path=util.getResourcePath() +'/Pickle Files/Feature Importance/')
    print(orderScaled.sort_values('Total',ascending=False))
    
#Rank the features of bot attack
att=bot
benign=benignTues.sample(frac=0.1)

# Construct the training data
att[' Label']=1
benign[' Label']=0
train_data=pd.concat([att,benign],axis=0)
train_data=shuffle(train_data)

getRanking(train_data, 'Bot_importance')