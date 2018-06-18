'''
@author: TinhTB

Provide common functions used in the project

'''
#Import necessary library
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_curve



#Get the path of the resource folder
def getResourcePath():   
    projectPath=os.path.dirname(os.path.abspath(__file__+ "/../"))
    return projectPath+'/res'

#Used to read UNB dataset
def csvReader(fileName):
    return pd.read_csv(getResourcePath()+'/UNB data/'+fileName+'.csv')

#Used to save variable to pickle file
def pklSaver(data, fileName,path=getResourcePath()+'/Pickle Files/'):
    with open(path+fileName+'.pkl','wb') as file:
        pickle.dump(data,file)

#Used to read pickle file
def pklReader(fileName,path=getResourcePath()+'/Pickle Files/'):
    if '.pkl' in fileName:
        fileName=fileName.split('.pkl')[0]
    with open(path+fileName+'.pkl','rb') as file:
        data=pickle.load(file)
    return data

#Define the columns need to be removed
def removeCol():   
    rmvCol1=['Flow ID',' Source IP',' Destination IP',' Timestamp']
    rmvCol2=[' Flow Duration','Total Length of Fwd Packets',' Fwd Packet Length Std',' Avg Fwd Segment Size','Fwd Packets/s','Fwd PSH Flags',' Fwd URG Flags',
    ' Packet Length Std',' Fwd Header Length.1',' Average Packet Size',' RST Flag Count',' Flow IAT Max', ' Fwd IAT Max', ' Idle Max', ' Idle Min',
    'Bwd Packet Length Max', ' Bwd Packet Length Mean', ' Avg Bwd Segment Size',' Total Fwd Packets',' Total Length of Bwd Packets','Subflow Fwd Packets', 
     ' Subflow Bwd Packets', ' Subflow Bwd Bytes']
    return rmvCol1+rmvCol2

#Used to reconstruct data for each day
def dataConv(fileName,savedName):
    data=csvReader(fileName) 
    
    #In the case of Friday DDos, ' External IP' should be also removed
    data=data.drop('External IP',axis=1)
    
    #Check NaN and Infinity value in two columns
    objCol=['Flow Bytes/s',' Flow Packets/s']

    #Convert 'Flow Bytes/s' and 'Flow Packets/s' to float (NaN and Infinity value will become nan and inf respectively)
    data[objCol]=data[objCol].astype(float)
    data[objCol]= data[objCol].replace(np.inf,np.nan)
    data=data.dropna()
    
    data=data.drop(removeCol(),axis=1)
    
    data.loc[data[' Label']!='BENIGN',' Label']=1
    data.loc[data[' Label']=='BENIGN',' Label']=0   
    pklSaver(data,savedName,path=getResourcePath()+'/Pickle Files/Each day/')
    
#Used to process and reconstruct data for the real traffic
def dataConvRealTraff(fileName):
    data=csvReader(fileName) 
    
    colNames=['Flow ID', ' Source IP', ' Source Port', ' Destination IP',
       ' Destination Port', ' Protocol', ' Timestamp', ' Flow Duration',
       ' Total Fwd Packets', ' Total Backward Packets',
       'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
       ' Fwd Packet Length Max', ' Fwd Packet Length Min',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
       ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
       ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
       ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
       ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
       ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
       ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
       ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
       ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
       ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
       ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
       ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
       ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
       ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
       'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
       ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
       ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
       ' Idle Max', ' Idle Min', ' Label']
    
    data.columns=colNames 
    #print(data.head())
    
    #Check NaN and Infinity value in two columns
    objCol=['Flow Bytes/s',' Flow Packets/s']

    #Convert 'Flow Bytes/s' and 'Flow Packets/s' to float (NaN and Infinity value will become nan and inf respectively)
    data[objCol]=data[objCol].astype(float)
    data[objCol]= data[objCol].replace(np.inf,np.nan)
    data=data.dropna()
    
    rmvCol=removeCol()
    rmvCol.remove(' Fwd Header Length.1')
    data=data.drop(rmvCol,axis=1)
    data.loc[data[' Label']=='No Label',' Label']=0 
    
    return data


#Find the optimal threshold using roc curve
def optThreshold(y,pred):
    fpr, tpr, thresolds= roc_curve(y,pred)
    optInx=np.argmax(abs(tpr-fpr))    
    optThreshold=thresolds[optInx]
    return optThreshold

#Calculate confusion matrix
def confusionMat(Ydata, y_pred):
    compare=(Ydata != y_pred)
    #Indices of wrong classification
    diff=np.where(compare)[0]
    #Indices of attack
    attackIndices=np.where(Ydata)[0]
    #Indices of attack which was classified as normal
    FNindices=np.intersect1d(attackIndices , diff)
    
    #Caculate the performance matrix
    FN=len(FNindices)
    TP=len(attackIndices)-FN
    FP=len(diff)-FN
    TN=len(Ydata)-len(attackIndices)-FP
    matrix={'TP':TP,'FN':FN,'FP':FP,'TN': TN,'N':len(Ydata)}
    return matrix

#Display evaluation parameters
def displayEval(evalMatrix):
#     print("False negative: ",evalMatrix['FN'])
#     print("False positive: ",evalMatrix['FP'])
#     print("True positive: ",evalMatrix['TP'])
#     print("True negative: ",evalMatrix['TN'])
    Acc=(evalMatrix['TP']+evalMatrix['TN'])/evalMatrix['N']
    Pr=evalMatrix['TP']/(evalMatrix['TP']+evalMatrix['FP'])
    Rc=evalMatrix['TP']/(evalMatrix['TP']+evalMatrix['FN'])
    F1=2*(Pr*Rc)/(Pr+Rc)
    print("Accuracy: ",Acc)
    print("Precision: ",Pr)
    print("Sensibility: ",Rc)
    print("F1 Score: ",F1)
    
