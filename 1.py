from enum import Enum

import numpy
import pandas


# dataset = pandas.read_csv('datasets/iris/bezdekIris.data',index_col=None,header=None)
#
# print(dataset)


# index = pandas.read_csv('datasets/iris/index',index_col=None,header=None)
#
# print(index.head())
#
#
iris = pandas.read_csv('datasets/iris/iris.data',index_col=None,header=None)

print(iris)
unique_text_features = iris.iloc[:, 4].unique()
print(unique_text_features)

class IrisClass(Enum):
    Iris_setosa = 1
    Iris_versicolor = 2
    Iris_virginica = 3

print(IrisClass.Iris_setosa.value)

text_array = ['Iris_setosa', 'Iris_versicolor', 'Iris_virginica', 'Iris_setosa']

print(IrisClass[text_array[1]].value)

x=iris.iloc[:,0:3].values
print(x)

y=iris.iloc[:,4].values
print(y)