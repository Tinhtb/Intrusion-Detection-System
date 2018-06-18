'''
@author: TinhTB

Define LightGBM and ExtraTress classifier which are used to rank the importance of each feature

'''
import lightgbm as lgbm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import time


#Use AUC for evaluation
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'AUC', roc_auc_score(labels, preds), True

#Get parameter for lightGBM
def getParam():
    #Parameters for training the model
    learning_rate = 0.1
    num_leaves = 15
    feature_fraction=0.8
    params = {"objective": "binary",
              "boosting_type": "gbdt",
              "learning_rate": learning_rate,
              "num_leaves": num_leaves,
              "feature_fraction":feature_fraction,
               "max_bin": 256,
              "verbosity": 0,
              "drop_rate": 0.1,
              "is_unbalance": False,
              "max_drop": 50,
              "min_child_samples": 10,
              "min_child_weight": 150,
              "min_split_gain": 0,
              "subsample": 0.9
              }
    return params

#Get num_boost_round
def getBoostRoundN():
    num_boost_round=300
    return num_boost_round

#Build a lightGBM model
def lightGBM_model(train_data,customEval=False):
    start = time.time()
    X=train_data.drop([' Label'],axis=1)
    y=train_data[' Label'].values
     
    #Create train dataset for lightGBM
    dtrain = lgbm.Dataset(X, y)    
    
    if customEval:
        #Perform cross-validation to optimise the parameter
        lgbm_cv=lgbm.cv(getParam(),dtrain,num_boost_round=getBoostRoundN(),feval=evalerror,early_stopping_rounds=100,verbose_eval=100)

        #Retrieve the best parameter
        bestCVScore=lgbm_cv['AUC-mean'][-1]
        bestBoostRound=len(lgbm_cv['AUC-mean'])
    else:
        #Perform cross-validation to optimise the parameter
        lgbm_cv=lgbm.cv(getParam(),dtrain,num_boost_round=getBoostRoundN(),early_stopping_rounds=100,verbose_eval=100)
        #print(lgbm_cv)
        #Retrieve the best parameter
        bestCVScore=lgbm_cv['binary_logloss-mean'][-1]
        bestBoostRound=len(lgbm_cv['binary_logloss-mean'])

    #Construct the model
    model = lgbm.train(getParam(), dtrain, num_boost_round=bestBoostRound)
   
    #Calculate time
    end = time.time()
    runningTime=(end-start)/60

    #Display the result
    print('Best round: ',bestBoostRound,',Best CV score: ',bestCVScore)
    print("Running time: ",runningTime)
    return model

#Build a model using Extremely Randomised Trees
def extraTrees(data): 
    X=data.drop([' Label'],axis=1)
    y=data[' Label'].values
    
    clf = ExtraTreesClassifier(n_estimators=120,random_state=42)
      
    model=clf.fit(X,y)
    return model