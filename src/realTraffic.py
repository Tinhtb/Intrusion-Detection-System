'''
@author: TinhTB

Perform the evaluation on the real traffic of our own network

'''

import util
import firstLayer
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

#Loading the real traffic
real_traffic=util.dataConvRealTraff('HulkWithTime.pcap_ISCX')
#Label the DoS as the attack target port 8080
real_traffic.ix[real_traffic[' Destination Port']==8080, ' Label'] = 1

#Load the test set of the evaluation dataset
testDataSet=util.pklReader('Testset',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First Layer/')

def getRocInfo(data):    
    #Exclude the label from observations
    x_test=data.drop(' Label',axis=1)
    y_test=data[' Label'].values
    
    #Get the prediction for the data
    pred=firstLayer.getPred(x_test)
    
    #Compute fpr, tpr 
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
    roc_auc=metrics.auc(fpr,tpr)
    
    return {'pred': pred, 'fpr': fpr, 'tpr': tpr, 'auc':roc_auc}

#Get the rocInfo for the real traffic
rocInfoR=getRocInfo(real_traffic)

#Get the prediction for the evaluation dataset
rocInfo=getRocInfo(testDataSet)

#Plot both ROC curve in a same figure
plt.title('Receiver Operating Characteristic')
plt.plot(rocInfo['fpr'],rocInfo['tpr'], 'b', label = 'Evaluation Dataset AUC = %0.4f' % rocInfo['auc'])
plt.plot(rocInfoR['fpr'],rocInfoR['tpr'], 'g', label = 'Our Traffic AUC = %0.4f' % rocInfoR['auc'])
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Finding optimal threshold
pred=rocInfoR['pred']
y_test=real_traffic[' Label'].values
optTh=util.optThreshold(y_test,pred)
print('\nThe performance of the model against new traffic data.')
print('The optimal threshold: ',optTh)        

#Calculate the confusion matrix of the classification result
clfResult=np.where(pred>optTh,1,0)
conf=util.confusionMat(y_test,clfResult)

#Display the evaluation of the first Layer
util.displayEval(conf)