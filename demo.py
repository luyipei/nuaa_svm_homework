from enum import Enum

import numpy as np
import pandas

class IrisClass(Enum):
    Iris_setosa = 1
    Iris_versicolor = 2
    Iris_virginica = 3

iris = pandas.read_csv('datasets/iris/iris.data',index_col=None,header=None)

# unique_text_features = iris.iloc[:, 4].unique()
# print(unique_text_features)

X = iris.iloc[:,0:3].values
y_txt = iris.iloc[:,4].values
# 转换为字符串类型数组
y_txt = y_txt.astype(str)
# 将文本数组中的 '-' 替换为 '_'，因为-枚举用不了
y_txt=np.char.replace(y_txt,'-','_')


# print(y_txt[0])
# print(IrisClass[y_txt[0]].value)
y = []
for i in  range(len(y_txt)):
    label=IrisClass[y_txt[i]].value
    y.append(label)

print(X)
print(y)

class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_iter=100):
        self.C = C  # 正则化参数
        self.tol = tol  # 容忍度
        self.max_iter = max_iter  # 最大迭代次数
        self.b = 0
        self.alpha = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.alpha = np.zeros(len(X))
        self.b = 0

        for _ in range(self.max_iter):
            alpha_changed = 0
            for i in range(len(X)):
                error_i = self._predict(X[i]) - y[i]
                if (y[i] * error_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * error_i > self.tol and self.alpha[i] > 0):
                    j = self._select_random_index(i, len(X))
                    error_j = self._predict(X[j]) - y[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)

                    if L == H:
                        continue

                    eta = 2.0 * X[i] @ X[j] - X[i] @ X[i] - X[j] @ X[j]
                    if eta >= 0:
                        continue

                    alpha_j_new = alpha_j_old - y[j] * (error_i - error_j) / eta
                    alpha_j_new = self._clip_alpha(alpha_j_new, L, H)

                    if abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue

                    alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)

                    b1 = self.b - error_i - y[i] * (alpha_i_new - alpha_i_old) * X[i] @ X[i] - \
                         y[j] * (alpha_j_new - alpha_j_old) * X[i] @ X[j]

                    b2 = self.b - error_j - y[i] * (alpha_i_new - alpha_i_old) * X[i] @ X[j] - \
                         y[j] * (alpha_j_new - alpha_j_old) * X[j] @ X[j]

                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_new < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    self.alpha[i] = alpha_i_new
                    self.alpha[j] = alpha_j_new
                    alpha_changed += 1

            if alpha_changed == 0:
                break

    def _predict(self, x):
        return np.sum(self.alpha * self.y * self._kernel(self.X, x)) + self.b

    def _kernel(self, x1, x2):
        # 简单的线性核函数
        return np.dot(x1, x2)

    def _select_random_index(self, i, m):
        j = i
        while j == i:
            j = np.random.randint(0, m)
        return j

    def _clip_alpha(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        return alpha

# 示例用法
# 注意：这只是一个简化的实现，实际的 SMO 算法更复杂
# 在真实的情况下，你可能需要进一步的优化和调整

# 生成一些示例数据
# X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y = np.array([-1, -1, 1, 1])

# 创建并训练 SVM 模型
svm_model = SVM(C=1.0)
svm_model.fit(X, y)

# 进行预测
predictions = [svm_model._predict(x) for x in X]
print("Predictions:", predictions)
