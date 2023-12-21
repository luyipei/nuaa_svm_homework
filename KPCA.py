import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载Iris数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# 数据标准化
scaler = StandardScaler()
X_iris_standardized = scaler.fit_transform(X_iris)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_iris_standardized, y_iris, test_size=0.2, random_state=42)

# PCA
pca = PCA(n_components=2)
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.transform(X_test)

# KPCA
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca_train = kpca.fit_transform(X_train)
X_kpca_test = kpca.transform(X_test)

# SVM分类器
svm = SVC(kernel='linear')
svm.fit(X_pca_train, y_train)
y_pred_pca = svm.predict(X_pca_test)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

svm.fit(X_kpca_train, y_train)
y_pred_kpca = svm.predict(X_kpca_test)
accuracy_kpca = accuracy_score(y_test, y_pred_kpca)

# 打印结果
print("Accuracy with PCA:", accuracy_pca)
print("Accuracy with KPCA:", accuracy_kpca)

# 绘制PCA的决策边界
h = .02
x_min, x_max = X_pca_train[:, 0].min() - 1, X_pca_train[:, 0].max() + 1
y_min, y_max = X_pca_train[:, 1].min() - 1, X_pca_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title('PCA Decision Boundaries')
plt.show()

# 绘制KPCA的决策边界
x_min, x_max = X_kpca_train[:, 0].min() - 1, X_kpca_train[:, 0].max() + 1
y_min, y_max = X_kpca_train[:, 1].min() - 1, X_kpca_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_kpca_train[:, 0], X_kpca_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title('KPCA Decision Boundaries')
plt.show()
