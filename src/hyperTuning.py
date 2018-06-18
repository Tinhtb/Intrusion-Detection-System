'''
@author: TinhTB

Perform hyper parameters tuning process to find the best model for each algorithm

'''
import util
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgbm
from sklearn.grid_search import RandomizedSearchCV
import time


#Perform hyperparameters tuning process to find the set of optimal hyperparameters
#for each algorithm
def hyperparaTuning(data, testSet,expName, mode=2, storedPath=util.getResourcePath()+'/Pickle Files/Models/First Layer/'):
    # Construct the set of hyperparameters for each algorithm
    etTree_params = {"n_estimators": [150, 250, 350],
                     "max_features" : [None, 'sqrt', 'log2'],
                     "min_samples_leaf": [64, 128, 256]}

    lightGBM_params = {"learning_rate": [0.06, 0.08, 0.1],
                       "num_leaves" : [15, 31, 63],
                       "max_bin": [63, 127, 255],
                      "feature_fraction": [0.6, 0.8, 0.9]}

    knn_params = {"n_neighbors": np.arange(5, 47, 2),
                  "weights" : ["uniform","distance"],
                  "metric": ["euclidean", "manhattan", "chebyshev"]}

    #Construct a model for each algorithm
    et_model=ExtraTreesClassifier()
    lgbm_model=lgbm.LGBMClassifier(objective = 'binary')
    knn_model=KNeighborsClassifier()
    
    #Construct the training and test data
    trainData=data.drop(' Label',axis=1)
    y_train=data[' Label'].values

    testData=testSet.drop(' Label',axis=1)
    y_test=testSet[' Label'].values
    
    #Perform Extremely Randomized ALgorithm
    if(mode==1):
        modelName='ExtraTrees'
        params=etTree_params
        model = et_model
    
    #Performing LightGBM
    elif(mode==2):
        modelName='LightGBM'
        params=lightGBM_params
        model = lgbm_model
     
    #Performing KNN algorithm   
    else:
        modelName='KNN'
        params=knn_params
        model = knn_model
        
        #Standardise the data in the case of KNN
        scaling=StandardScaler()
        trainData= scaling.fit_transform(trainData)
        testData= scaling.transform(testData)
        
    
    # tune the hyperparameters via a cross-validated Randomized search
    grid = RandomizedSearchCV(model, params,verbose=1, cv=5, n_jobs=1)
    start = time.time()
    grid.fit(trainData, y_train)

    #Calculate the time
    end = time.time()
    runningTime=(end-start)/60

    # evaluate the best grid searched model on the testing data
    preds = grid.predict_proba(testData)
    auc= roc_auc_score(y_test,preds[:,1])

    print("Experiment: ", expName)
    print("Randomized search best parameters: {}".format(grid.best_params_))
    print("AUC of the best model: ", auc)
    print("Running time: ",runningTime)

    #Save the model
    util.pklSaver(grid,expName,path=storedPath+modelName+'/')