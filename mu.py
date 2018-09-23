# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 11:07:10 2018

@author: sn06
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
#READ DATA#####################################################################
df = pd.read_csv('mushrooms.csv')
df['class'][df['class']=='e'] = 0
df['class'][df['class']=='p'] = 1
df['class'] = df['class'].map(int)
#CONVERT DATA TO NUMBERS#######################################################
le = LabelEncoder()

def encode(i):
    df[i] = le.fit_transform(df[i])
    z = dict(zip(le.classes_,le.transform(le.classes_)))
    return z

key = []
for i in list(df.iloc[:,1:]):
    temp = encode(i)
    key.append(i)
    key.append(temp)
#VIEW CORRELATION MATRIX#######################################################
sns.heatmap(df.corr(),cmap='plasma')
plt.show()
plt.close()
#SPLIT TRAIN/TEST DATA AND FIT MODEL###########################################
X = df.iloc[:,1:].values
y = df['class'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))
#PLOT FEATURES IN ORDER OF SIGNIFICANCE########################################
coefs = abs(model.coef_).reshape(-1)
labels = list(df.iloc[:,1:])
output = pd.DataFrame({'coefs':coefs,'labels':labels})
output = output.sort_values(by='coefs')
plt.bar(range(len(output)),output['coefs'])
plt.xticks(range(len(output)),output['labels'],rotation=90)
plt.show()