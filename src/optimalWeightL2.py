'''
@author: TinhTB

Finding the optimal weight for lightGBM model and ExtraTrees model to 
construct an ensemble model for the second layer

'''

import util
import time
import numpy as np
import pandas as pd

##Finding the optimal weight for each base class in the second layer
#List of six attack types
attNames=['BruteForce','DoS','Web','Bot','PortScan','DDoS']
#The number of attack
n_attack=len(attNames)
#clfKey is a list represent 15 pairs of 6 attack types
clfKeys=[]
for i in range(n_attack-1):
    for j in range (i+1,n_attack):
        clfKeys+=[attNames[i]+'-'+attNames[j]]
        
#Loading the model of each algorithm
et_clfModels={}
lgbm_clfModels={}
for clfkey in clfKeys:
    et_clfModels[clfkey]=util.pklReader(clfkey,path=util.getResourcePath()+'/Pickle Files/Models/Second Layer/ExtraTrees/')
    lgbm_clfModels[clfkey]=util.pklReader(clfkey,path=util.getResourcePath()+'/Pickle Files/Models/Second Layer/LightGBM/')

        
#Using constructed models to predict the data
def getPrediction(X_test,a):
    initialData=np.zeros((len(X_test),n_attack))
    predictResult=pd.DataFrame(initialData,columns=attNames)

    #Sum the probability of each classifier
    for clfkey in clfKeys:   
        key=clfkey.split('-')
        
        #Loading the model of each algorithm
        et_clfModel=et_clfModels[clfkey]
        lgbm_clfModel=lgbm_clfModels[clfkey]
        
        #Get the prediction of each algorithm
        y_predictET=et_clfModel.predict_proba(X_test)
        y_predictLGBM=lgbm_clfModel.predict_proba(X_test)

        #Compute the final prediction
        y_predict=a*y_predictLGBM+(1-a)*y_predictET

        #Prediction result for each attack
        predictResult[key[0]]+=y_predict[:,0]
        predictResult[key[1]]+=y_predict[:,1]

    #Convert string name to numeric number
    predictResult.columns=list(range(1,7))
    #The prediction is the class with highest probability
    y_predict=predictResult.idxmax(axis=1)
    return y_predict
   
#Find the optimal weight for each base class in the second layer
def secondLayerWeights():
    #Loading the test set
    allAttack20=util.pklReader('AllAttack20',path=util.getResourcePath() +'/Pickle Files/Data for model construction/Second layer/')
   
    # Convert the tring lable to numeric number
    count=1
    for att in allAttack20:
        allAttack20[att][' Label']=count
        count=count+1
    
    #A dictionary to store the number of misclassification for each value of lightGBM weight
    allErr={}
    start = time.time()
    
    #Calculate the prediction accuracy for each attack
    for i in range(50):
        #randomly select value for lightGBM
        a=np.random.uniform(low=0.7,high=1)
        totalErr=0
        for attack in allAttack20:    
            X_test=allAttack20[attack].drop([' Source Port', ' Destination Port', ' Protocol',' Label'],axis=1)
            y_test=allAttack20[attack][' Label'].values
            y_predict=getPrediction(X_test,a)
    
            error=(y_predict!=y_test).sum()
            totalErr=totalErr+error            
            
        allErr[a]=totalErr
        print('The weight of lightGBM: ',a,', Number of errors: ',totalErr)
        
    #Calculate the time
    end = time.time()
    runningTime=(end-start)/60
    
    #lightGBM weight is the one that minimise the number of misclassification
    lightGBM_weight=min(allErr, key=lambda key: allErr[key])
    
    print('Running time: ', runningTime)    
    print('Max AUC: ',lightGBM_weight,allErr[lightGBM_weight] )
    
    return lightGBM_weight

#Find the optimal weight of the second layer
lgbm_L2=secondLayerWeights()
    
