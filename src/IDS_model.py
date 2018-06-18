'''
@author: TinhTB

Perform the evaluation on the model of the first and second layer

'''
import util
import firstLayer
import secondLayer
import time
import numpy as np

##Evaluate the first model
def firstLayerEval(testDataSet):
   
    #Exclude the label from observations
    x_test=testDataSet.drop(' Label',axis=1)
    y_test=testDataSet[' Label'].values
    #Evaluate the performance of the first layer
    start = time.time()
    pred=firstLayer.getPred(x_test)    
    #Calculate the running time
    end = time.time()
    runningTime=end-start
    
    print('Prediction  time: ',runningTime)
    
    #Finding optimal threshold
    optTh=util.optThreshold(y_test,pred)
    print('The optimal threshold: ',optTh)        
    
    #Calculate the confusion matrix of the classification result
    clfResult=np.where(pred>optTh,1,0)
    conf=util.confusionMat(y_test,clfResult)
  
    #Display the evaluation of the first Layer
    util.displayEval(conf)

##Evaluate the second layer
def secondLayerEval(testSet):    
    #Label each attack with corresponding number : 
    #'BruteForce':1 ,'DoS':2 ,'Web':3 ,'Bot':4 ,'PortScan':5 ,'DDoS':6
    count=1
    for att in testSet:
        testSet[att][' Label']=count
        count=count+1
    
    totalErr=0
    totalLen=0
    #Evaluate the accuracy of each attack
    for attack in testSet:    
            #Separate obsevation and label
            X_test=testSet[attack].drop([' Source Port', ' Destination Port', ' Protocol',' Label'],axis=1)
            y_test=testSet[attack][' Label'].values
            
            #Get the prediction for the observation
            y_predict=secondLayer.getClassification(X_test)
    
            #Calculate the number of misclassification
            error=(y_predict!=y_test).sum()
            totalErr+=error
            totalLen+=len(testSet[attack])
            #Display attack name, total observation, number of misclassification and the accuracy
            print(attack,', misclassification rate: ',error,'/',len(testSet[attack]),', Accuracy: ',100-error*100/len(y_predict))
            
    print('Total number of misclassification: ',totalErr)
    print('Overall accuracy: ',1-totalErr/totalLen)

#Display the evaluation result
#Loading the test set for the first layer
test_set=util.pklReader('Testset',path=util.getResourcePath() +'/Pickle Files/Data for model construction/First Layer/')
print('Evaluating the performance of the first layer ...')
firstLayerEval(test_set)

#Loading the test set for the second layer
allAttack20=util.pklReader('AllAttack20',path=util.getResourcePath() +'/Pickle Files/Data for model construction/Second layer/')
print('\nEvaluating the performance of the second layer ...')
secondLayerEval(allAttack20)