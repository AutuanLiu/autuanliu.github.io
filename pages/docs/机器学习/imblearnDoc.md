# 不均衡学习

* sampler 采样器

## base

* 拟合
```python
estimator = obj.fit(data, targets)
```
* 采样
```python
data_resampled, targets_resampled = obj.sample(data, targets)
```
* 同时采样 拟合
```python
data_resampled, targets_resampled = obj.fit_sample(data, targets)
```

* 在不平衡比率较大的情况下，决策功能倾向于采用更多样本的类别，通常称为多数类别

## Over-sampling

1. RandomOverSampler
2. SMOTE
3. ADASYN

* 在样本量少的类别中生成新的样本(随机采样替换当前可用样本)
* Apart from the random sampling with replacement, there is two popular methods to over-sample minority classes:
    * Synthetic Minority Oversampling Technique (SMOTE) 
    * Adaptive Synthetic (ADASYN) sampling method. These algorithm can be used in the same manner
* RandomOverSampler 通过复制少数类别的一些原始样本进行上抽样
* SMOTE, ADASYN 通过 **插值** 产生新的样本进行上采样
* ADASYN专注于在 **使用k-最近邻分类器错误分类** 的原始样本的旁边生成样本
* SMOTE might connect inliers and outliers while ADASYN might focus solely on outliers which, in both cases, might lead to a sub-optimal decision function
* SMOTE 基于最佳决策函数边界附近的样本提供3个额外的选项生成样本，在最近邻类别的相反方向生成样本 'regular'，'borderline1'，'borderline2'，'svm'
* ADASYN is working similarly to the regular SMOTE
* RandomOverSampler样本生成过程中不需要任何类间信息。因此，每个目标类都是独立重新采样的
* 无论是ADASYN和 SMOTE需要有关用于生成样本的每个样本的相邻信息

## under-sampling

1. Prototype generation
2. Prototype selection
3. RandomUnderSampler

* Prototype generation 原型生成技术将减少目标类中的样本数量，但剩余的样本是 **从原始集合中生成的，而不是被选中的**
* ClusterCentroids 使用 k-means 来减少样本数量， 每个类将与K-means方法的质心合成，而不是原始样本
* ClusterCentroids 提供了一种有效的方式来用少量的样本表示数据集群， 这种方法要求数据分组到集群中。另外，应该设置质心的数量，使得欠采样的簇代表原始的簇
* ClusterCentroids支持稀疏矩阵。但是，新生成的样本并不是特别稀疏的。因此，即使得到的矩阵是稀疏的，在这方面算法将是低效的

* 与原型生成算法相反，原型选择 Prototype selection 算法将从原始集合 S 中选择样本
    * the controlled under-sampling techniques
    * the cleaning under-sampling techniques
* RandomUnderSampler 是一种通过随机选择目标类的数据子集来平衡数据的快速而简单的方法
* RandomUnderSampler允许通过设置replacement = True。多个类的重采样通过独立考虑每个目标类来执行

* NearMiss添加一些启发式规则来选择样本。NearMiss实现3种不同类型的启发式，可以用参数选择version


## collections 模块

* Python拥有一些内置的数据类型，比如str, int, list, tuple, dict等， collections模块在这些内置数据类型的基础上，提供了几个额外的数据类型：
    * namedtuple(): 生成可以使用名字来访问元素内容的tuple子类
    * deque: 双端队列，可以快速的从另外一侧追加和推出对象
    * Counter: 计数器，主要用来计数
    * OrderedDict: 有序字典
    * defaultdict: 带有默认值的字典


### 参考文献

1. [imbalanced-learn documentation](http://contrib.scikit-learn.org/imbalanced-learn/stable/install.html)
2. [不可不知的Python模块: collections](http://www.zlovezl.cn/articles/collections-in-python/)