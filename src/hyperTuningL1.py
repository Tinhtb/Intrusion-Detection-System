'''
@author: TinhTB

Perform hyperparameters tuning to find the best model in each experiment for the construction of the first layer

'''
import util
import hyperTuning

#Create a new dataset from the original dataset with n is the number of features 
#in the new dataset
def getNewDataSet(train_data, test_data,n,pca=False):
    #If the new dataset is created from the original data
    if(pca==False):
        featureList=impList[:n]+[' Label']
        data=train_data[featureList]
        testSet=test_data[featureList]
        
    #If the dataset is created from PCA
    else:
        col=list(train_data.columns)
        featureList=col[:n]+[' Label']
    
        data=train_data[featureList]
        testSet=test_data[featureList]
    
    return {'train': data,'test': testSet}   

## Perform hyperparameters for the first layer
# Loading train and test set
originalData=util.pklReader('Original',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')
testDataSet=util.pklReader('Testset',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')
impList=util.pklReader('ImportanceList',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')
pca_trainData=util.pklReader('PCA_data',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')
pca_testData=util.pklReader('PCA_testData',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')

#Using LightGBM for hyperparameters tuning
# #Exp1
# train1=originalData.drop([' Source Port', ' Destination Port', ' Protocol'],axis=1)
# test1=testDataSet.drop([' Source Port', ' Destination Port', ' Protocol'],axis=1)
# hyperTuning.hyperparaTuning(train1,test1,'Exp1')
# 
# #Exp2
# newData=getNewDataSet(originalData, testDataSet, 35)
# hyperTuning.hyperparaTuning(newData['train'],newData['test'],'Exp2')
# 
# #Exp3
newData=getNewDataSet(originalData, testDataSet, 25)
hyperTuning.hyperparaTuning(newData['train'],newData['test'],'Exp3')
# 
# #Exp4
# newData=getNewDataSet(originalData, testDataSet, 15)
# hyperTuning.hyperparaTuning(newData['train'],newData['test'],'Exp4')
# 
# #Exp5
# newData=getNewDataSet(pca_trainData, pca_testData, 34,pca=True)
# hyperTuning.hyperparaTuning(newData['train'],newData['test'],'Exp5')
# 
# #Exp6
# newData=getNewDataSet(pca_trainData, pca_testData, 26,pca=True)
# hyperTuning.hyperparaTuning(newData['train'],newData['test'],'Exp6')
# 
# #Exp7
# newData=getNewDataSet(pca_trainData, pca_testData, 22,pca=True)
# hyperTuning.hyperparaTuning(newData['train'],newData['test'],'Exp7')
