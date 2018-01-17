import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import lunardate as ld 

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import tensorflow.contrib.learn as learn

df = pd.read_csv('2006_2017_features.csv')
solardt = pd.to_datetime(df['date'])

lunaryr = []
lunarmth = []
lunarday = []

for i in range(df.shape[0]):

	lunardt = ld.LunarDate.fromSolarDate(solardt[i].year, solardt[i].month, solardt[i].day)

	lunaryr.append(lunardt.year)

	lunarmth.append(lunardt.month)

	lunarday.append(lunardt.day)

df['lunar_year'] = lunaryr
df['lunar_month'] = lunarmth
df['lunar_day'] = lunarday

blue = []
red = []
green = []

for i in range(1,1828):
    blue.append(df['color'].iloc[:i].value_counts()['blue'])
    
for i in range(2,1828):
    red.append(df['color'].iloc[:i].value_counts()['red'])
    
for i in range(3,1828):
    green.append(df['color'].iloc[:i].value_counts()['green'])

blue = [0] + blue
red = [0,0] + red
green = [0,0,0] + green

df['red'] = red[0:1827]
df['green'] = green[0:1827]
df['blue'] = blue[0:1827]

dict_color = {'red':0, 'green':1, 'blue':2}
color = df['color'].map(dict_color)
df.loc[:,'color'] = color

df_clean = df[['year', 'lunar_day', 'red', 'green', 'blue', 'color']]
holdout = df_clean[df_clean['year']==2017]
df_clean = df_clean[df_clean['year']!=2017]
df_clean = df_clean.drop('year', axis=1)

X = df_clean.drop('color', axis=1)
y = df_clean['color']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Logistic Regression
clf_log = LogisticRegression()
clf_log.fit(X_train, y_train)
score_log = cross_val_score(clf_log, X_train, y_train, cv=5)
pred_log = clf_log.predict(X_test)

# SVC
clf_svc = SVC()
clf_svc.fit(X_train, y_train)
score_svc = cross_val_score(clf_svc, X_train, y_train, cv=5)
pred_svc = clf_svc.predict(X_test)

# Decision tree
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)
score_dt = cross_val_score(clf_dt, X_train, y_train, cv=5)
pred_dt = clf_dt.predict(X_test)

# DNN
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
clf_dnn = learn.DNNClassifier(hidden_units=[10,10,5], n_classes=3, feature_columns=feature_columns)
clf_dnn.fit(X_train, y_train, steps=10, batch_size=20)
predgen_dnn = clf_dnn.predict(X_test)
pred_dnn = []
for i in predgen_dnn:
	pred_dnn.append(i)

print('Logistic Regression')
print(score_log)
print(np.mean(score_log))
print(confusion_matrix(y_test, pred_log))
print('\n')
print(classification_report(y_test, pred_log))

print('SVC')
print(score_svc)
print(np.mean(score_svc))
print(confusion_matrix(y_test, pred_svc))
print('\n')
print(classification_report(y_test, pred_svc))

print('Decision Tree')
print(score_dt)
print(np.mean(score_dt))
print(confusion_matrix(y_test, pred_dt))
print('\n')
print(classification_report(y_test, pred_dt))

print('DNN')
print(confusion_matrix(y_test, pred_dnn))
print('\n')
print(classification_report(y_test, pred_dnn))







