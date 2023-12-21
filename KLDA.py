import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 加载Iris数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# 数据标准化
scaler = StandardScaler()
X_iris_standardized = scaler.fit_transform(X_iris)

# KLDA
klda = LinearDiscriminantAnalysis(n_components=2)
X_klda = klda.fit_transform(X_iris_standardized, y_iris)

# SVM分类器
svm = SVC(kernel='linear')

# 五重交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 使用KLDA进行交叉验证
scores_klda = cross_val_score(svm, X_klda, y_iris, cv=cv, scoring='accuracy')

# 打印结果
print("Accuracy with KLDA (5-fold cross-validation):")
print(scores_klda)
print("Mean Accuracy: {:.2f}".format(np.mean(scores_klda)))

# 绘制KLDA的决策边界
h = .02
x_min, x_max = X_klda[:, 0].min() - 1, X_klda[:, 0].max() + 1
y_min, y_max = X_klda[:, 1].min() - 1, X_klda[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm.fit(X_klda, y_iris).predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_klda[:, 0], X_klda[:, 1], c=y_iris, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title('KLDA Decision Boundaries')
plt.show()
