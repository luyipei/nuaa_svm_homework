from enum import Enum

import numpy
import pandas

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn

#根据数据集得到的鸢尾花数据的类型，使用枚举对应上其编号
class IrisClass(Enum):
    Iris_setosa = 1
    Iris_versicolor = 1
    Iris_virginica = 3

def min_max_normalize(x):
    min_val = numpy.min(x, axis=0)
    max_val = numpy.max(x, axis=0)
    min_max_normalized = (x - min_val) / (max_val - min_val)
    return min_max_normalized



#划分数据集

from sklearn.model_selection import train_test_split


iris = pandas.read_csv('datasets/iris/iris.data',index_col=None,header=None)

# unique_text_features = iris.iloc[:, 4].unique()
# print(unique_text_features)

X = iris.iloc[:,0:3].values
x = min_max_normalize(X)
y_txt = iris.iloc[:,4].values
# 转换为字符串类型数组
y_txt = y_txt.astype(str)
# 将文本数组中的 '-' 替换为 '_'，因为-枚举用不了
y_txt=numpy.char.replace(y_txt,'-','_')


# print(y_txt[0])
# print(IrisClass[y_txt[0]].value)
y = []
for i in  range(len(y_txt)):
    label=IrisClass[y_txt[i]].value
    y.append(label)

# print(X)
# print(y)



# 数据集是 X（特征）和 y（标签）

# 划分数据集为训练集、验证集和测试集
# test_size 参数可以指定测试集的大小，可以是一个百分比（如0.2）或样本数量
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# 再将剩余的部分划分为验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 输出各个集合的大小
print("训练集大小:", len(X_train))
print("验证集大小:", len(X_val))
print("测试集大小:", len(X_test))


#定义smo算法
class  SMO:
    def __init__(self, C=3.0, tol=1e-3, max_iter=100):
        self.C = C  # 正则化参数
        self.tol = tol  # 容忍度
        self.max_iter = max_iter  # 最大迭代次数
        self.b = 0
        self.alpha = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        #初始化参数列表
        self.X = X
        self.y = y
        self.alpha = numpy.zeros(len(X))
        self.b = 0

        #开始进行迭代
        for iter_i in range(self.max_iter):
            num_changed_alphas = 0
            for i in range(len(X)):
                error_i = self._predict(X[i]) - y[i]
                if((y[i]*error_i<self.tol and self.alpha[i]<self.C)or(y[i]*error_i>self.tol and self.alpha[i]>0)):
                    #选择一个与当前样本 i 不同的、随机的样本 j
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

                    #计算eta，@为计算二者的内积，这里第一次遇到这种表示
                    eta = 2.0 * X[i] @ X[j] - X[i] @ X[i] - X[j] @ X[j]
                    if eta >= 0:
                        continue

                    alpha_j_new = alpha_j_old - y[j] * (error_i - error_j) / eta
                    alpha_j_new = self._clip_alpha(alpha_j_new, L, H)

                    if abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue

                    #计算alpha_i的值
                    alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)
                    #计算b1和b2的值
                    b1 = self.b - error_i - y[i] * (alpha_i_new - alpha_i_old) * X[i] @ X[i] - y[j] * (alpha_j_new - alpha_j_old) * X[i] @ X[j]

                    b2 = self.b - error_j - y[i] * (alpha_i_new - alpha_i_old) * X[i] @ X[j] - y[j] * (alpha_j_new - alpha_j_old) * X[j] @ X[j]

                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_new < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    self.alpha[i] = alpha_i_new
                    self.alpha[j] = alpha_j_new
                    num_changed_alphas += 1



    def _predict(self, x):
        return numpy.sum(self.alpha * self.y * self._kernel(self.X, x)) + self.b

    def _kernel(self, x1, x2):
        # 简单的线性核函数
        return numpy.dot(x1, x2)

    def _select_random_index(self, i, m):
        j = i
        while j == i:
            j = numpy.random.randint(0, m)
        return j

    def _clip_alpha(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        return alpha

    def score(self, X, y):
        pass


# 创建并训练 SVM 模型
svm_model = SMO(C=1.0)
svm_model.fit(X, y)

# 进行预测
predictions = [svm_model._predict(x) for x in X]
print("Predictions:", predictions)







