import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 names=['sepal_length', 'sepal width', 'petal length', 'petal width', 'class'])

# print(df.info())
# print(df.describe())

marker_shape = ['.', '^', '*']
ax = plt.axes()
for i, spacie in enumerate(df['class'].unique()):
    spacie_data = df[df['class'] == spacie]
    # print(spacie_data)
    spacie_data.plot.scatter(x='sepal_length', y='sepal width', marker=marker_shape[i],
                             s=100, title='sepal width vs length by speacies',
                             label=spacie, figsize=(10, 7), ax=ax)
# plt.show()

df['petal length'].plot.hist(title='histogram of petal length')
# plt.show()
df.plot.box(title='Boxplot of Sepal length & width and petal length and width  ')
# plt.show()


df2 = pd.DataFrame({'Day': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', ' sunday']})
# print(df2)
# print(pd.get_dummies(df2))

random_index = np.random.choice(df.index, replace=False, size=10)
df.loc[random_index, 'sepal_length'] = None
print(df.isnull().any())

print('number of rows before deleting : %d' %(df.shape[0]))
df3 = df.dropna()
print('number of rows before deleting : %d' %(df3.shape[0]))
df.sepal_length = df.sepal_length.fillna(df.sepal_length.mean())
print(df.isnull().any())
















