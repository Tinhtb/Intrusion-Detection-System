'''
@author: TinhTB

Perform data analysis to examine the correlation between features, 
Perform PCA to reduce the dimension of the data

'''

import  util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

#Load all data (benign and attack)
allData=util.pklReader('All Data',path=util.getResourcePath()+'/Pickle Files/Original Data/')

#Loading the train and test data 
train_data=util.pklReader('Original',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')
test_data=util.pklReader('Testset',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')


#Remove the duplicate 'Fwd Header Length' column
allData=allData.drop([' Fwd Header Length.1'],axis=1)
print('Shape of the data', allData.shape)

#Plot the number of normal vs number of attack
def normalVsAttack():
    #Plot number of normal/attack
    n_attack=sum(allData[' Label']==1)*100/len(allData)
    n_normal=100-n_attack
    
    label=['Normal', 'Attack']
    y=[n_normal,n_attack]
    plt.bar(np.arange(len(label)), y,align='center')
    plt.xticks(np.arange(len(label)),label)
    plt.ylabel('Percentage')
    plt.title('The number of Normal/Attack')
    plt.show()


#Pairs of high correlated features
pair1=[' Flow Duration', 'Fwd IAT Total']
pair2=[' Total Fwd Packets', ' Total Backward Packets',' Total Length of Bwd Packets', 
       'Subflow Fwd Packets', ' Subflow Bwd Packets', ' Subflow Bwd Bytes']
pair3=['Total Length of Fwd Packets', ' Subflow Fwd Bytes']
pair4=[' Fwd Packet Length Max', ' Fwd Packet Length Std']
pair5=[' Fwd Packet Length Mean', ' Avg Fwd Segment Size']
pair6=['Bwd Packet Length Max', ' Bwd Packet Length Mean',
       ' Bwd Packet Length Std', ' Avg Bwd Segment Size']
pair7=[' Flow Packets/s', 'Fwd Packets/s']
pair8=[' Flow IAT Max', ' Fwd IAT Max', 'Idle Mean', ' Idle Max', ' Idle Min']
pair9=['Fwd PSH Flags', ' SYN Flag Count']
pair10=[' Fwd URG Flags', ' CWE Flag Count']
pair11= [' Fwd Header Length', ' Fwd Header Length.1']
pair12= [' Max Packet Length', ' Packet Length Std']
pair13= [' Packet Length Mean', ' Average Packet Size']
pair14= [' RST Flag Count', ' ECE Flag Count']

#Display the correlation between features of a particular group
def pairCorr(allData,pair):
    correlations = allData[pair].corr()
    print(correlations)
    
#Construct PCA from train data
def getPCA(train_data,test_data):
    originalData=train_data
    #Extract the observation
    xData=originalData.drop(' Label',axis=1)
    #Define a StandardScaler
    scaling=StandardScaler()
    #Standardise the data
    X_std = scaling.fit_transform(xData)
    
    #Construct PCA data
    pca_std=PCA().fit(X_std)
    
    #Transform to PCA components
    pcaData=pca_std.transform(X_std)
    
    #Create the column for pcaData
    pcaCol=[]
    for i in range(pcaData.shape[1]):
        col='Component'+str(i+1)
        pcaCol+=[col]
        
    #Convernt numpy array to data frame
    pcaDf=pd.DataFrame(data=pcaData,columns=pcaCol)
    
    #Add Label column to pcaDf
    pcaDf[' Label']=originalData[' Label'].values
    #Save the result in a pickle file
    util.pklSaver(pcaDf,'PCA_data',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')
    
    
    #Convert test Data to PCA
    testDataSet=test_data
    testData=testDataSet.drop(' Label',axis=1)
    
    #Using the same scaling in the train data to transform the test data
    testDataStd = scaling.transform(testData)
    test_pcaData=pca_std.transform(testDataStd)
        
    #Convernt numpy array to data frame
    test_pcaDf=pd.DataFrame(data=test_pcaData,columns=pcaCol)
    
    #Add Label column to pcaDf
    test_pcaDf[' Label']=testDataSet[' Label'].values
    #Save the result in a pickle file
    util.pklSaver(test_pcaDf,'PCA_testData',path=util.getResourcePath()+'/Pickle Files/Data for model construction/First layer/')
    
#Plot normal vs attack
normalVsAttack()

#Display correlation of pair2
pairCorr(allData, pair2)

#Perform  Principle Component analysis
getPCA(train_data,test_data)