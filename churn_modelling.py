import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

print(tf.__version__)

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:,4:-1]
y = df.iloc[:,-1]
print(df.isnull().sum())

le = LabelEncoder()
X.iloc[:,1]  =  le.fit_transform(X.iloc[:,1])
columntrans = ColumnTransformer([('encoder',OneHotEncoder(),  [1])],remainder = 'passthrough')
geo = pd.get_dummies(X['Geography'],drop_first = True)
X = pd.concat([X,geo],axis = 1)
X = X.drop('Geography',axis = 1)
print(X)

X_train,X_test,y_train,y_test = train_test_split(X,y)
print(X.loc[2])

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann  =  tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1,activation = 'sigmoid'))
ann.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
ann.fit(X_train,y_train,batch_size = 32,epochs = 100)

print(ann.predict(sc.transform([[0,42,8,159660.80,3,1,0,113931.57,0,0]])))

y_pred = ann.predict(X_test)
y_pred  =  y_pred > 0.5

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.values.reshape(len(y_pred),1)),1))
print(confusion_matrix(y_pred,y_test),accuracy_score(y_pred,y_test))






