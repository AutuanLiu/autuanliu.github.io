# 模型选择与评估

 * 为了避免过拟合，在进行（监督）机器学习实验时，通常取出部分可利用数据作为 test set（测试数据集） X_test, y_test

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)
```

* 当评价估计器的不同设置（”hyperparameters(超参数)”）时，例如手动为 SVM 设置的 C 参数， 由于在训练集上，通过调整参数设置使估计器的性能达到了最佳状态；但 在测试集上 可能会出现过拟合的情况。 此时，测试集上的信息反馈足以颠覆训练好的模型，评估的指标不再有效反映出模型的泛化性能。 为了解决此类问题，还应该准备另一部分被称为 “validation set(验证集)” 的数据集，**模型训练完成以后在验证集上对模型进行评估。 当验证集上的评估实验比较成功时，在测试集上进行最后的评估**

* 然而，通过将原始数据分为3个数据集合，我们就大大减少了可用于模型学习的样本数量， 并且得到的结果依赖于集合对（训练，验证）的随机选择
* 这个问题可以通过 交叉验证（CV 缩写） 来解决。 交叉验证仍需要测试集做最后的模型评估，但不再需要验证集
* k-fold
    * 将 k-1 份训练集子集作为 training data （训练集）训练模型，
    * 将剩余的 1 份训练集子集作为验证集用于模型验证（也就是利用该数据集计算模型的性能指标，例如准确率）
    * k-折交叉验证得出的性能指标是循环计算中每个值的平均值
    * 该方法虽然计算代价很高，但是它不会浪费太多的数据（如固定任意测试集的情况一样）， 在处理样本数据集较少的问题（例如，逆向推理）时比较有优势
* 使用交叉验证最简单的方法是在估计器和数据集上调用 cross_val_score 辅助函数
* [Model evaluation: quantifying the quality of predictions — scikit-learn 0.19.1 documentation](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, iris.data, iris.target, cv=5) # 会得到 5 个分数
# 评分估计的平均得分和 95% 置信区间
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

* 其他方式

```python
from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, iris.data, iris.target, cv=cv)
```

* 保留数据的数据转换
* 不可以在一开始就对 所有的数据进行转换

```python
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)
# 应当保存
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
# 应用到测试集上
X_test_transformed = scaler.transform(X_test)
clf.score(X_test_transformed, y_test)  
```

* 其他方式

```python
from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, iris.data, iris.target, cv=cv)
```

* cross_validate 函数与 cross_val_score 在下面的两个方面有些不同

* 它允许指定多个指标进行评估.
* 除了测试得分之外，它还会返回一个包含训练得分，拟合次数， score-times （得分次数）的一个字典

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
cv=5, return_train_score=False)

# other
from sklearn.metrics.scorer import make_scorer
scoring = {'prec_macro': 'precision_macro',
'rec_micro': make_scorer(recall_score, average='macro')}

# 单一指标
scores = cross_validate(clf, iris.data, iris.target,
scoring='precision_macro')
```

* 除了返回结果不同，函数 cross_val_predict 具有和 cross_val_score 相同的接口， 对于每一个输入的元素，如果其在测试集合中，将会得到预测结果

## 交叉验证迭代器
* K-fold
* RepeatedKFold 当需要运行时可以使用它 KFold n 次，在每次重复中产生不同的分割
* LeaveOneOut 每个学习集都是通过除了一个样本以外的所有样本创建的，测试集是被留下的样本， 不会浪费太多数据
* LOO 经常导致较高的方差作为测试误差的估计器
* 5-fold 或者 10-fold 交叉验证应该优于 LOO
* LeavePOut 与 LeaveOneOut 非常相似，因为它通过从整个集合中删除 p 个样本来创建所有可能的 训练/测试集, p>1 时， 测试集会重叠
* ShuffleSplit 迭代器 将会生成一个用户给定数量的独立的训练/测试数据划分。样例首先被打散然后划分为一对训练测试集合
* ShuffleSplit 可以替代 KFold 交叉验证，因为其提供了细致的训练 / 测试划分的 数量和样例所占的比例等的控制
* 一些分类问题在目标类别的分布上可能表现出很大的不平衡性：例如，可能会出现比正样本多数倍的负样本。在这种情况下，建议采用如 StratifiedKFold 和 StratifiedShuffleSplit 中实现的分层抽样方法，确保相对的类别频率在每个训练和验证 fold 中大致相同
* StratifiedKFold 是 k-fold 的变种，会返回 stratified（分层） 的折叠：每个小集合中， 各个类别的样例比例大致和完整数据集中相同
* StratifiedShuffleSplit 是 ShuffleSplit 的一个变种，会返回直接的划分
* GroupKFold 是 k-fold 的变体，它确保同一个 group 在测试和训练集中都不被表示
```python
from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))
```

* LeaveOneGroupOut
* LeavePGroupsOut
* GroupShuffleSplit 迭代器是 ShuffleSplit 和 LeavePGroupsOut 的组合，它生成一个随机划分分区的序列，其中为每个分组提供了一个组子集
* **TimeSeriesSplit** 是 k-fold 的一个变体，它首先返回 k 折作为训练数据集，并且 (k+1) 折作为测试数据集

```python
from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)  

for train, test in tscv.split(X):
    print("%s %s" % (train, test))
```

* 如果数据的顺序不是任意的（比如说，相同标签的样例连续出现），为了获得有意义的交叉验证结果，首先对其进行 打散是很有必要的. 然而，当样例不是独立同分布时打散则是不可行的

## 超参数调整

