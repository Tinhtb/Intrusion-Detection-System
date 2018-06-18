'''
@author: TinhTB

Apply the one-vs-one approach to build 15 classifier for 15 pairs of 6 attacks,
Each classifier was constructed by performing hyperparameters tuning

'''
import util
import pandas as pd
from sklearn.utils import shuffle
import hyperTuning

## Perform hyperparameters for the second layer
# Loading train and test set
allAttack80=util.pklReader('AllAttack80',path=util.getResourcePath()+'/Pickle Files/Data for model construction/Second layer/')
allAttack20=util.pklReader('AllAttack20',path=util.getResourcePath()+'/Pickle Files/Data for model construction/Second layer/')
#Name of six types of attack
attName=['BruteForce','DoS','Web','Bot','PortScan','DDoS']
   
#Total number of attack 
n_attack=len(attName)

#A list to store pairs of attack
clfKeys=[]
for i in range(n_attack-1):
    for j in range (i+1,n_attack):
        clfKeys+=[attName[i]+'-'+attName[j]]
 
#Tuning each one of 15 classifiers for each pair of attacks using lightGBM
clfModels={}
for clfkey in clfKeys:
    #Getting the name of attack from each pair   
    key=clfkey.split('-')
    
    #Construct train data by setting the first attack as 0 and the second one as 1
    allAttack80[key[0]][' Label']=0
    allAttack80[key[1]][' Label']=1
    
    #Concatinate the data of both attacks
    train_dataClf=pd.concat([allAttack80[key[0]],allAttack80[key[1]]],axis=0)
    train_dataClf=shuffle(train_dataClf)
    
    #Construct test data
    allAttack20[key[0]][' Label']=0
    allAttack20[key[1]][' Label']=1
    test_dataClf=pd.concat([allAttack20[key[0]],allAttack20[key[1]]],axis=0)
    
    #train_dataClf=train_dataClf.sample(frac=0.1)
    #test_dataClf=test_dataClf.sample(frac=0.1)
    
    #Perform hyperparameters tuning
    hyperTuning.hyperparaTuning(train_dataClf,test_dataClf,expName=clfkey, storedPath=util.getResourcePath()+'/Pickle Files/Models/Second Layer/')
