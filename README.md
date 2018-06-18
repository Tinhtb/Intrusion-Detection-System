# Intrusion-Detection-System
This project aims to develop an IDS model using Machine Learning and Data Mining Techniques

In this project, we examined the performance of three common ML algorithms: 
Random Forest (ExtraTrees), Gradient Boosting (LightGBM) and K-nearest Neighbours 
against both evaluation dataset (UNB 2017) and the real traffic in our network. 
We also applied several Data Mining techniques to improve the quality of the original data 
such as Data Cleaning, Data Rebalancing, Principal Component Analysis, Feature Selection,…

1)	res: this folder contains all necessary files to run the code
-	UNB data: this folder contains all .cvs files of five days traffic 
captured by the University of New Brunswick together with a .csv file captured in our own network

-	Pickle Files: this folder contains all pickle files which were 
generated during the development of our project. The Models subfolder contains 
all models constructed during the evaluation process as well as the final model for our first and second layer. 

2)	src: this folder contains all python code of our project 
(please see the comments on each file for more information)

-	To run the evaluation on the UNB dataset:
			python IDS_model.py

-	To run the evaluation on the real traffic of our network:
			python realTraffic.py
