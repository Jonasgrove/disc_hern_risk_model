#!/usr/bin/env python



#Problem 4
import pandas as pd
import numpy as np

def atRisk(rows,k,path):
  import pandas as pd
  vert = pd.read_csv(my_path, sep=' ',header=None)

  vertX = vert.iloc[:,0:6]
  from sklearn.preprocessing import LabelEncoder  #encode the strings to integers
  le = LabelEncoder()
  vert.iloc[:,6] = le.fit_transform(vert.iloc[:,6])
  vertY = vert.iloc[:,6]

  #scale x data
  scaler = StandardScaler()
  columns_X = [list(vertX.columns)]
  for feature in columns_X:
      vertX[feature] = scaler.fit_transform(vertX[feature])

  #break into test and train data
  Xtest,Ytest,Xtrain,Ytrain = getTestSet(rows,vertX,vertY)
  print("Y test", Ytest)
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = k)
  knn.fit(Xtrain, Ytrain)
  riskProbs = knn.predict_proba(Xtest)

  #createNewMatrix
  dsMatrix = pd.DataFrame(columns=['DH','SL'])
  i = 1
  for result in riskProbs:
    instance = [0,0]
    #check DH
    if result[0] >= .5:
      instance[0] = 1
    elif result[0] >= 30:
      instance[0] = 2
    #check SL      
    if result[1] >= .5:
      instance[1] = 1
    elif result[1] >= 30:
      instance[1] = 2

    addInst = pd.DataFrame(data=[instance],index=[i],columns=['DH','SL'])
    dsMatrix = pd.concat([dsMatrix,addInst])
    i+=1
  
  return dsMatrix


def getTestSet(rows,XvertData,YvertData):
  import random

  Xtest = pd.DataFrame()
  Ytest = pd.DataFrame()
  randI = random.sample(range(len(XvertData)),rows)

  for i in randI:
    Xtest = pd.concat([Xtest,XvertData[i:i+1][:]])
    XvertData = XvertData.drop([i])
    Ytest = pd.concat([Ytest,YvertData[i:i+1][:]])
    YvertData = YvertData.drop([i])

  return Xtest,Ytest,XvertData,YvertData

my_path = '/content/drive/My Drive/Spring 20/ECS 171 ML/column_3C.dat'
rows = 15
print()
atRisk(rows,5,my_path)