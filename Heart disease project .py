import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
         'slop', 'ca', 'hal', 'HeartDisease']

df = pd.read_excel('4.1 Ch3.ClevelandData.xlsx', names=names)
# print(df.head(20))
# print(df.info())
# summary = df.describe()
# print(summary)
Dataview = df.replace('?', np.nan)
# print(Dataview.info())
# print(Dataview.describe())
# print(Dataview.isnull().sum())

# delete all rows
Dataview = Dataview.dropna()
# print(Dataview.isnull().sum())
# print(Dataview.info())

# x(scaled) = (x-mean)/sd
# mean = 0 ,  standard deviation = 1
# value  > mean will have +z score
# value  < mean will have -z score

inputnames = names
inputnames.pop()
Input = pd.DataFrame(Dataview.iloc[:, 0:13], columns=inputnames)
Target = pd.DataFrame(Dataview.iloc[:, 13], columns=['HeartDisease'])
# print(Target)
scaler = StandardScaler()
# print(scaler.fit(Input))
InputScaled = scaler.fit_transform(Input)
InputScaled = pd.DataFrame(InputScaled, columns=inputnames)
summary = InputScaled.describe()
summary = summary.transpose()
# print(summary)
# print(InputScaled)
#
boxblot = InputScaled.boxplot(column=inputnames, showmeans=True)
pd.plotting.scatter_matrix(InputScaled, figsize=(6, 6))
# plt.show()
CorData = InputScaled.corr(method='pearson')
with pd.option_context('display.max_rows', None, 'display.max_columns', CorData.shape[1]):
    print(CorData)

plt.matshow(CorData)
plt.xticks(range(len(CorData.columns)), CorData.columns)
plt.yticks(range(len(CorData.columns)), CorData.columns)
plt.colorbar()
# plt.show()

Input_train, Input_test,Target_train,Target_test = train_test_split(InputScaled, Target, test_size=0.3,
                                                                      random_state=5)
# print(Input_train.shape)
# print(Input_test.shape)
# print(Target_test.shape)
# print(Target_train.shape)

# test size = 0.3 mean that 30% of the data is divided up as data set
# random_State parameter is used to set the seed used by the random number generator.
# inputscaled and target parameter are inputs and target Dataframes.
# 207 rows (input train) and 89 (input test)
#
# create keras sequential model
# import sequential class from keras.models
# stack the layer using the .add() method
# configure the learning process and using the .compile() method
# train the model on the train data set using the .fit() method
#
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim=13, activation='tanh'))  # input
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

''' * The adam optimaizer : an algorithom for the first-order gradient-base oprimizer 
of stochoastic objective function , based on adaptive estimates of lower-order moment 

* binary_crossentropy loss function: we will use logarithmic loss , which
for a binary classsification problem is definde in Keras as binary_crossentropy

*  the accuracy metrics : is a function this is used to evaluate the 
proformance of your model during training and testing 
'''

model.fit(Input_train,Target_train,epochs=1000,verbose=1)

# input tainn :  array for input training data
# target train  : array of target(label) data
# epochs = 1000 number of epochs to train the model.an epoch is an iteration over the entire x and y data provided
# verbose = 1 : an integer , either 0,1,2 verbose mode 0 = silent ,1=progress bar , 2 = one line per epoch

model.summary()
score = model.evaluate(Input_test,Target_test,verbose=0)
print('Keras model accuracy = ',score[1])
Tatget_classification = model.predict(Input_test)
Tatget_classification = (Tatget_classification >0.5)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Target_test,Tatget_classification))
#35 + 36 = 71
# 71 observation of 89 correctly classifacted by making 18 error with and accuracy equal 0.7977
