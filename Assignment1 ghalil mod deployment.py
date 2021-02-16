import pandas as pd
import numpy as np 
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
LE=LabelEncoder()
data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
data.replace('?',np.nan,inplace=True)
data_boat=data['boat']
data_cabin=data['cabin']
data_embarked=data[['embarked']]
data_age=data[['age']]
data_sex=data[['sex']]
data_sex=LE.fit_transform(data_sex)
y=data[['survived']]

data_age=data[['age']].astype(float)
data_age.fillna(np.mean(data_age),inplace=True)
data_embarked=data_embarked.fillna('S')
data_embarked=LE.fit_transform(data_embarked)

k=[]
for i in data_cabin:
    #print(type(i))
    if type(i)!=type(2.3):
        if len(i)>3:
            i=i[0]+i[1]
    k.append(i)
k=pd.DataFrame(k)
k.fillna(0,inplace=True)
k=k.astype(str)
k=LE.fit_transform(k)       


z=[]
for i in data_boat:
   # print(i)
   # print(type(i))
   if i=='A' or i=='B' or i=='C' or i=='D' or i=='5 9'or  i=='13 15'or i=='15 16' or i=='C D' or i=='5 7' or i=='8 10' or i=='13 15 B':
      i='nan'
   #if type(i)=='str':
     # i=i[0]+i[1]
   z.append(i)
z_cleaned=[]
z=pd.DataFrame(z)
z=z.astype(float)
z.fillna(0,inplace=True)

#Array concatenations
final=pd.concat([z,pd.DataFrame(k),pd.DataFrame(data_sex),data_age,pd.DataFrame(data_embarked)],axis=1)

#model traininng and splitting
x_train,x_test,y_train,y_test=train_test_split(final,y,random_state=0,test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
LR=LogisticRegression()
LR.fit(x_train,y_train)
predict=LR.predict(x_test)
accuracy_score(predict,y_test)
f1_score(predict,y_test)
