'''
@author: TinhTB

Finding the optimal weight for lightGBM model and ExtraTrees model to 
construct an ensemble model for the first layer

'''

import util
import time
import numpy as np
from sklearn.metrics import roc_auc_score

#Find the optimal weight for each base class in the first layer
def firstLayerWeights():
    #Loading test data and two subsets of test data
    subset1=util.pklReader('subset1',path=util.getResourcePath()+'/Pickle Files/Data for model construction/Subset/')
    subset2=util.pklReader('subset2',path=util.getResourcePath()+'/Pickle Files/Data for model construction/Subset/')
    
    #Loading the best model of each algorithms
    et_model=util.pklReader('Exp1',path=util.getResourcePath()+'/Pickle Files/Models/First Layer/ExtraTrees/')
    lgbm_model=util.pklReader('Exp2',path=util.getResourcePath()+'/Pickle Files/Models/First Layer/LightGBM/')
    
    #Uses subset1 as training data to search for weight
    impList=util.pklReader('ImportanceList',path=util.getResourcePath()+'/Pickle Files/Data for model construction/')
    colET=list(subset1.drop([' Source Port', ' Destination Port', ' Protocol',' Label'],axis=1).columns)
    x_subset1ET=subset1[colET]
    x_subset1LGBM=subset1[impList[:35]]
    y_subset1=subset1[' Label'].values    
    
    #used subset2 as test data
    #x_subset2=subset2.drop(' Label',axis=1)
    x_subset2ET=subset2[colET]
    x_subset2LGBM=subset2[impList[:35]]
    y_subset2=subset2[' Label'].values
    
    #A dictionary to store AUC score of each value of lightGBM weights
    auc_score={}
    start = time.time()
    
    #Used only lightGBM and ExtraTrees
    for i in range(100):
        #randomly select value for lightGBM
        a=np.random.uniform(low=0.5,high=1)
        b=1-a
        
        predict =a*lgbm_model.predict_proba(x_subset1LGBM)+b*et_model.predict_proba(x_subset1ET)
        auc= roc_auc_score(y_subset1,predict[:,1])
        auc_score[a]=auc
        print('The weight of lightGBM: ',a,', AUC score: ',auc)
        
    #Calculate the time
    end = time.time()
    runningTime=end-start

    #Get the weight of lightGBM which maximise AUC score
    lightGBM_weight=max(auc_score, key=lambda key: auc_score[key])
    
    #Display results
    print('Running time: ', runningTime)
    print('Max AUC: ',lightGBM_weight,auc_score[lightGBM_weight] )
    
    #Compare the result with individual algorithm
    print('LightGBM', roc_auc_score(y_subset1,lgbm_model.predict_proba(x_subset1LGBM)[:,1])) 
    print('ExtraTrees', roc_auc_score(y_subset1,et_model.predict_proba(x_subset1ET)[:,1])) 
    
    #For the second subset
    predict =lightGBM_weight*lgbm_model.predict_proba(x_subset2LGBM)+(1-lightGBM_weight)*et_model.predict_proba(x_subset2ET)
    auc= roc_auc_score(y_subset2,predict[:,1])
    print('\n The performance on the second subset')
    print('The ensemble model: ', auc)
    print('LightGBM', roc_auc_score(y_subset2,lgbm_model.predict_proba(x_subset2LGBM)[:,1])) 
    print('ExtraTrees', roc_auc_score(y_subset2,et_model.predict_proba(x_subset2ET)[:,1])) 
    
    return lightGBM_weight

#Find the optimal weight of the first layer
lgbm_L1=firstLayerWeights()