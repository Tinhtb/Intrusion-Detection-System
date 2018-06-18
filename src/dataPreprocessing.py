'''
@author: TinhTB

Perform data preprocessing process to improve the quality of the original data
The results were saved in corresponding pickle files

'''
import util
import numpy as np
from os import walk

#Load the data of each day
monday=util.csvReader('Monday-WorkingHours.pcap_ISCX')
tues=util.csvReader('Tuesday-WorkingHours.pcap_ISCX')
wed=util.csvReader('Wednesday-workingHours.pcap_ISCX')
thursMor=util.csvReader('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX')
thursAft=util.csvReader('Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX')
friMor=util.csvReader('Friday-WorkingHours-Morning.pcap_ISCX')
friPort=util.csvReader('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX')
friDos=util.csvReader('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX')


def dataProcessor(day,inputData):
    #Perform some analysis
    print('Number of column: ',len(inputData.columns))
    
    #Display distinct value in Lable column
    labelSet=set(inputData[' Label'])
    print('The distinct value in Label column: ',labelSet)
    
    #Original shape
    origShape=inputData.shape
    print('Shape of original data: ',origShape )
    
    
    #Remove unnecessary columns
    rmvCol=['Flow ID',' Source IP',' Destination IP',' Timestamp']
    inputData=inputData.drop(rmvCol,axis=1)
    
    #Check NaN and Infinity value in two columns
    objCol=['Flow Bytes/s',' Flow Packets/s']
    
    #Convert 'Flow Bytes/s' and 'Flow Packets/s' to float (NaN and Infinity value will become nan and inf respectively)
    inputData[objCol]=inputData[objCol].astype(float)
    
    #Replace inf by nan
    inputData[objCol]= inputData[objCol].replace(np.inf,np.nan)
    print('Total number of nan: ',inputData[objCol].isna().sum())
    print('before: ',inputData.shape)
    # #Drop nan rows
    inputData=inputData.dropna()
    print('The final shape of the data: ',inputData.shape)
    
    #Check duplicate data after removing column and nan, inf
    origShape=inputData.shape
    inputData.drop_duplicates()
    rmvShape=inputData.shape
    print('Shape of the dataset after removing duplicates: ', rmvShape)
    print('Number of duplicate rows: ',origShape[0]-rmvShape[0])
    
    #Store data of each attack in pkl file
    labelList=list(labelSet)
    count=0
    for label in labelList:
        dataLabel=inputData[inputData[' Label']==label]
        print('Number of '+label+' :',dataLabel.shape)
        count+=dataLabel.shape[0]
        #pklSaver(dataLabel,label)
        
        if(label=='BENIGN'):
            util.pklSaver(dataLabel, day+'-Benign', path=util.getResourcePath()+'/Pickle Files/Original Data/Benigns/')
        
        else:
            util.pklSaver(dataLabel, label, path=util.getResourcePath()+'/Pickle Files/Original Data/Attacks/')        
        
    print('Total: ',count)
    
    return {'label': labelSet, 'data': inputData}

#Combine the data of each file in a folder to make a single dataset
def getAllData(folderPath=util.getResourcePath()+'/Pickle Files/Original Data/Benigns/'):
    #Get the list of files in the folder
    files=[]
    for (dirPath,dirNames,fileName) in walk(folderPath):
        files.extend(fileName)
    
    #Initial the allBenign
    allData=util.pklReader(files[0],path=folderPath)
    
    #Append each benign to allBenign
    for i in range(1,len(files)):
        pklData=util.pklReader(files[i],path=folderPath)
        allData=allData.append(pklData)
    print('Total length of allData: ',len(allData))
    
    return allData
   
#Perform dataPreprocessing
dataProcessor('Tues', tues)

#Get all Benign
allBenign=getAllData()
util.pklSaver(allBenign,'All Benign',path=util.getResourcePath()+'/Pickle Files/Original Data/')

#Get all attack
allAttacks=getAllData(folderPath=util.getResourcePath()+'/Pickle Files/Original Data/Attacks/')
util.pklSaver(allAttacks,'All Attacks',path=util.getResourcePath()+'/Pickle Files/Original Data/')

# Convert the Label of allAttack to 1 and allBenign to 0
allAttacks[' Label']=1
allBenign[' Label']=0
allData=allAttacks.append(allBenign)

#Display the length of all data and save it to 'All Data' pkl file
print('Length of all data: ', len(allData))
util.pklSaver(allData,'All Data',path=util.getResourcePath()+'/Pickle Files/Original Data/')
