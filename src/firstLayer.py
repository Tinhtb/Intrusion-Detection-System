'''
@author: TinhTB

The final model for our first layer, which is an ensemble of LightGBM model and
ExtraTrees model. The weight of LightGBM model is 0.9985

'''
import util

#Loading the ExtraTrees model
et_model=util.pklReader('Exp1',path=util.getResourcePath() +'/Pickle Files/Models/First Layer/ExtraTrees/')

#Loading the LightGBM model
lgbm_model=util.pklReader('Exp2',path=util.getResourcePath() +'/Pickle Files/Models/First Layer/LightGBM/')

#Loading the list of features' importance 
impList=util.pklReader('ImportanceList',path='E:/Workspace/res/Pickle Files/First layer datasets/')


def getPred(x_test):   
    colET=list(x_test.drop([' Source Port', ' Destination Port', ' Protocol'],axis=1).columns)
         
    #Assign the optimal weight for lightGBM model
    lgbm_weight=0.9985
    x_testET=x_test[colET]
    x_testLGBM=x_test[impList[:35]]

    #Compute the final prediction value by multiply the prediction of each model with its weight
    predict =lgbm_weight*lgbm_model.predict_proba(x_testLGBM)+(1-lgbm_weight)*et_model.predict_proba(x_testET)
    pred=predict[:,1]    
        
    return pred
