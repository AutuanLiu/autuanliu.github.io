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
* RandomOverSampler样本生成过程中不需要任何类间信息.因此，每个目标类都是独立重新采样的
* 无论是ADASYN和 SMOTE需要有关用于生成样本的每个样本的相邻信息

## under-sampling

1. Prototype generation
2. Prototype selection
3. Cleaning under-sampling techniques

* Prototype generation 原型生成技术将减少目标类中的样本数量，但剩余的样本是 **从原始集合中生成的，而不是被选中的**
* ClusterCentroids 使用 k-means 来减少样本数量, 每个类将与K-means方法的质心合成，而不是原始样本
* ClusterCentroids 提供了一种有效的方式来用少量的样本表示数据集群, 这种方法要求数据分组到集群中.另外，应该设置质心的数量，使得欠采样的簇代表原始的簇
* ClusterCentroids支持稀疏矩阵.但是，新生成的样本并不是特别稀疏的.因此，即使得到的矩阵是稀疏的，在这方面算法将是低效的

* 与原型生成算法相反，原型选择 Prototype selection 算法将从原始集合 S 中选择样本
    * the controlled under-sampling techniques
    * the cleaning under-sampling techniques
* RandomUnderSampler 是一种通过随机选择目标类的数据子集来平衡数据的快速而简单的方法
* RandomUnderSampler允许通过设置replacement = True.多个类的重采样通过独立考虑每个目标类来执行

* NearMiss添加一些启发式规则来选择样本.NearMiss实现3种不同类型的启发式，可以用参数 version 选择
* NearMiss 基于最近邻
* NearMiss-1选择与负类的N个最接近样本的平均距离最小的正样本
* NearMiss-2选择与负类别的N个最远样本的平均距离最小的正样本
* NearMiss-3是一个两步算法.首先，对于每个负样本，他们的 M个最近邻将被保留.然后，选择的正样本是到N个最近邻的平均距离最大的样本
* NearMiss-1, NearMiss-2 都很容易受到噪声的影响

* Cleaning under-sampling techniques 不允许指定每个类别中的样品数量
* A Tomek’s link between two samples of different class x and y is defined such that there is no example z such that(where d(.) is the distance between the two samples)
$$
d(x, y) < d(x, z) \text{ or } d(y, z) < d(x, y)
$$
* 如果两个样本是彼此近邻，则存在 Tomek's link
* 参数ratio控制哪个link的样本将被删除, 默认会从 多数类别中 移除link，设置为 all 可以移除所有的 link
* EditedNearestNeighbours 用最近邻居算法并通过去除与他们的邻居不够“一致”的样本来“编辑”数据集.对于该类中的每个样本进行下采样，计算最近邻居，如果不满足选择标准，则移除样本
* RepeatedEditedNearestNeighbours通过多次重复该算法来扩展 .通常，重复该算法将删除更多的数据
* AllKNN differs from the previous RepeatedEditedNearestNeighbours since the number of neighbors of the internal nearest neighbors algorithm is increased at each iteration
* RepeatedEditedNearestNeighbours, EditedNearestNeighbours, AllKNN 会清除位于决策边界附近的 样本
* CondensedNearestNeighbour使用1个最近邻居规则迭代决定是否应该删除一个样本
* CondensedNearestNeighbour 对噪声敏感，会增加噪音样本
* OneSidedSelection将使用TomekLinks去除噪声样本
* NeighbourhoodCleaningRule will focus on cleaning the data than condensing them. Therefore, it will used the union of samples to be rejected between the EditedNearestNeighbours and the output a 3 nearest neighbors classifier
* InstanceHardnessThreshold is a specific algorithm in which a classifier is trained on the data and the samples with lower probabilities are removed
* SMOTE并表明，这种方法可以通过在边缘离群点和内点之间插入新点来生成噪声样本
* Tomek’s link and edited nearest-neighbours are the two cleaning methods which have been added pipeline after SMOTE over-sampling to obtain a cleaner space
*  SMOTETomek and  SMOTEENN 是既包含上采样又包含下采样的 方法, 同样不能保证各个类别的数目相同
* TomekLinks, EditedNearestNeighbours, CondensedNearestNeighbour, RepeatedEditedNearestNeighbours, AllKNN, NeighbourhoodCleaningRule, InstanceHardnessThreshold 均不指定类别的样本个数，不保证每个类别的样本数是相同的
* 即 Prototype generation(ClusterCentroids) 方法能够保证各个类别的样本数目是一样的而 Prototype selection 基本上不能保证各个类别的样本数目是一样的(仍然是不均衡的)
* ClusterCentroids，RandomUnderSampler 和 NearMiss 可以保证每个类别的样本数目相等
*  SMOTEENN tends to clean more noisy samples than SMOTETomek

### Ensemble of samplers

* 一个不平衡的数据集可以通过创建几个平衡的子集来平衡
* EasyEnsemble creates an ensemble of data set by randomly under-sampling the original set
* EasyEnsemble has two important parameters: (i) n_subsets will be used to return number of subset and (ii) replacement to randomly sample with or without replacement
* `BalanceCascade` differs from the previous method by using a classifier (using the parameter estimator) to ensure that misclassified samples can again be selected for the next subset
* In ensemble classifiers, bagging methods build several estimators on different randomly selected subset of data, However, this classifier does not allow to balance each subset of data. Therefore, when training on imbalanced data set, this classifier will favor the majority classes
* BalancedBaggingClassifier allows to resample each subset of data before to train each estimator of the ensemble. In short, it combines the output of an EasyEnsemble sampler with an ensemble of classifiers 

### 总结
* under sampling
    * ClusterCentroids 每类数目相同
    * RandomUnderSampler 每类数目相同
    * InstanceHardnessThreshold
    * NearMiss 每类数目相同
    * TomekLinks
    * EditedNearestNeighbours
    * RepeatedEditedNearestNeighbours
    * AllKNN
    * OneSidedSelection
    * CondensedNearestNeighbour
    * NeighbourhoodCleaningRule
* over samoling
    * ADASYN 每类数目基本相同
    * RandomOverSampler 每类数目相同
    * SMOTE 每类数目基本相同
* combine
    * SMOTEENN
    * SMOTETomek
* ensemble
    * EasyEnsemble 从原始数据集中随机下采样
    * BalancedBaggingClassifier
    * BalanceCascade
* metric
    * sensitivity_specificity_support
    * sensitivity_score
    * specificity_score
    * geometric_mean_score
    * make_index_balanced_accuracy
    * classification_report_imbalanced


## collections 模块

* Python拥有一些内置的数据类型，比如str, int, list, tuple, dict等, collections模块在这些内置数据类型的基础上，提供了几个额外的数据类型：
    * namedtuple(): 生成可以使用名字来访问元素内容的tuple子类
    * deque: 双端队列，可以快速的从另外一侧追加和推出对象
    * Counter: 计数器，主要用来计数
    * OrderedDict: 有序字典
    * defaultdict: 带有默认值的字典


### 参考文献

1. [imbalanced-learn documentation](http://contrib.scikit-learn.org/imbalanced-learn/stable/install.html)
2. [不可不知的Python模块: collections](http://www.zlovezl.cn/articles/collections-in-python/)