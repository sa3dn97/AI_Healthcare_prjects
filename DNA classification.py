# import sys
import numpy as np
# import sklearn
import pandas as pd
#######
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import model_selection

########


# print('python:{}'.format(sys.version))
# print('numpy:{}'.format(np.__version__))
# print('sklearn:{}'.format(sklearn.__version__))
# print('pandas:{}'.format(pd.__version__))


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters' \
      '.data '
names = ['Class', 'id', 'Sequence']
data = pd.read_csv(url, names=names)
# print(data.iloc[0])

classes = data.loc[:, 'Class']
# print(classes[:5])
sequance = list(data.loc[:, 'Sequence'])
dataset = {}

for i, seq in enumerate(sequance):
    # split and remove tab characters
    nucleotides = list(seq)
    nucleotides = [x for x in nucleotides if x != '\t']
    # add class
    nucleotides.append(classes[i])
    # add to dataset
    dataset[i] = nucleotides
# print(dataset[0])

dframe = pd.DataFrame(dataset)
# print(dframe)
# switch row and columns using transpose() function

df = dframe.transpose()
# print(df.iloc[:5])
# rename the last colume as Class
df.rename(columns={57: 'Class'}, inplace=True)
# print(df.iloc[:5])
# print(df.describe())

series = []
for name in df.columns:
    series.append(df[name].value_counts())

info = pd.DataFrame(series)
details = info.transpose()
# print(details)

numerrical = pd.get_dummies(df)
# print(numerrical.iloc[:5])
df1 = numerrical.drop(columns=['Class_-'])
df1.rename(columns={'Class_+': 'Class'}, inplace=True)
# print(df1.iloc[:5])
# print(df1.iloc[60])


##
# CREAT x and y datasets for training

X = np.array(df1.drop(['Class'], 1))
print(X)
y = np.array(df1['Class'])
print(y)
# definde seed of prodactivety
seed = 1
# split data into training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=seed)

# defind scoring method
scoring = 'accuracy'
# defind the model to train
names = ['Nearest Neighbors', 'Gaussian process', 'Decision Tree', 'Random Forest',
         'Neural Net', 'AdaBoost', 'Naive Bayes', 'SVM Linear', 'SVM RBF', 'SVM Sigmoid']

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(kernel='linear'),
    SVC(kernel='rbf'),
    SVC(kernel='sigmoid')
]
models = zip(names, classifiers)
# evaluate each model in turn
result = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_result = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    result.append(cv_result)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_result.mean(), cv_result.std())
    print(msg)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(classification_report(y_test, prediction))


# E.coli bacteria DNA