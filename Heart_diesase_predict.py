# import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import keras
from sklearn import model_selection
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chlo', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
         'slope', 'ca', 'thal', 'class']
df = pd.read_csv(url, names=names)

# print('shape of DataFrame: {}'.format(df.shape))
# print(df.loc[1])
# print(df.loc[200:])
# this
df = df.replace('?', np.nan)
df = df.dropna()
# print(df.isnull().sum())

# or this
#
# df = cleveland(~cleveland.isin(['?']))
# df = df.dropna(axis=0)

# print('shape of DataFrame: {}'.format(df.shape))
df = df.apply(pd.to_numeric)
# print(df.dtypes)
# print(df.describe())
df.hist(figsize=(12, 12))
# plt.show()


x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
# print(Y_train.shape)
# print(Y_train[:10])

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


## defind function to build the keras model

def creat_model():
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # complie model
    adam = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def creat_model_binary():
    model1 = Sequential()
    model1.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model1.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model1.add(Dense(1, activation='sigmoid'))# binary
    # complie model
    adam = Adam(lr=0.001)

    model1.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) # binary

    return model1

model = creat_model()
print(model.summary())

# model.fit(X_train,Y_train,epochs=100,batch_size=10,verbose=1)

# conveirt into binary classifcation heart disease of not

Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])
# print(Y_test_binary[:20])
binary_model = creat_model_binary()

binary_model.fit(X_train,Y_train_binary,epochs=100,batch_size=10,verbose=1)


from sklearn.metrics import classification_report,accuracy_score

# categ_pd = np.argmax(model.predict(X_test),axis=1)
# print('result')
# print(accuracy_score(y_test,categ_pd))
# print(classification_report(y_test,categ_pd))

binary11 = np.round(binary_model.predict(X_test)).astype(int)
print('result')
print(accuracy_score(y_test,binary11))
print(classification_report(y_test,binary11))





