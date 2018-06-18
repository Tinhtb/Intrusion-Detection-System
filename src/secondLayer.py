'''
@author: TinhTB

The final model for our second layer, which sum the probability of all 15 lightGBM base models
and take the highest probability as the output class

'''

import util
import numpy as np
import pandas as pd

#List of six attack types
attNames=['BruteForce','DoS','Web','Bot','PortScan','DDoS']
#The number of attack
n_attack=len(attNames)
#clfKey is a list represent 15 pairs of 6 attack types
clfKeys=[]
for i in range(n_attack-1):
    for j in range (i+1,n_attack):
        clfKeys+=[attNames[i]+'-'+attNames[j]]

#Construct the second layer of the IDS system using one vs one approach
 
#Using ExtraTrees classifier to construct 45 models
et_clfModels={}
lgbm_clfModels={}
for clfkey in clfKeys:
    lgbm_clfModels[clfkey]=util.pklReader(clfkey,path=util.getResourcePath()+'/Pickle Files/Models/Second Layer/LightGBM/')


#Using constructed models to predict the data
def getClassification(X_test):
    initialData=np.zeros((len(X_test),n_attack))
    predictResult=pd.DataFrame(initialData,columns=attNames)

    #Sum the probability of each classifier
    for clfkey in clfKeys:   
        key=clfkey.split('-')
        lgbm_clfModel=lgbm_clfModels[clfkey]
        y_predict=lgbm_clfModel.predict_proba(X_test)
        predictResult[key[0]]+=y_predict[:,0]
        predictResult[key[1]]+=y_predict[:,1]

    #Convert string name to numeric number
    predictResult.columns=list(range(1,7))
    #The prediction is the class with highest probability
    y_predict=predictResult.idxmax(axis=1)
    return y_predict


    