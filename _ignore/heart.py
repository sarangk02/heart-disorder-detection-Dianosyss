import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

#import eli5
#from eli5.sklearn import PermutationImportance

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from tensorflow import keras


df = pd.read_csv('Heart_Disease_Prediction.csv')

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-2] ,df.iloc[:,-1], random_state = 42, test_size = 0.0000000000001, shuffle = True)

encoder =LabelEncoder()
df['Heart Disease'] = encoder.fit_transform(df['Heart Disease'])
y = encoder.fit_transform(y_test)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train)
x_test = sc_x.transform(X_test)


y_train_dum = pd.get_dummies(y_train)

classifier=keras.Sequential([
    keras.layers.Dense(32,activation='relu',input_dim=x_train.shape[1],kernel_initializer='glorot_normal',bias_initializer='zeros'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(22,activation='relu',kernel_initializer='glorot_normal',bias_initializer='zeros'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2,activation='relu',kernel_initializer='glorot_normal',bias_initializer='zeros'),
    keras.layers.Dense(y_train_dum.shape[1],activation='sigmoid')
])

classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
classifier.fit(x_train, y_train_dum,epochs = 100, batch_size = 20)


# prediction = classifier.predict(x_test)
# prediction = [np.argmax(i) for i in prediction]
# og_prediction = encoder.inverse_transform(prediction)
# ann_score = accuracy_score(y,prediction) * 100

# print(ann_score)

# classifier.save("heart_disease_model.h5")
pickle.dump(classifier, open('heart_disease_model.pkl', 'wb'))
pickle.dump(sc_x, open('hd_scaler.pkl', 'wb'))