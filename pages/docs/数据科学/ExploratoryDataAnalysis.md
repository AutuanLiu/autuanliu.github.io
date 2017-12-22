# 探索性数据分析

```
author: autuanliu
email: autuanliu@163.com
date: 2017/11/30
```

## 要点

### 整体观察

* How many observations do I have?
* How many features?
* What are the data types of my features? Are they numeric? Categorical?
* Do I have a target variable?

## 数值特征的分布

* histogram 往往是最有效的
* 潜在的**离群值**是没有意义的
* 边界没有意义
* 潜在的测量错误
* 记下想要做的修复

## 分类特征的分布

* bar plot 是有效的
* 可以清楚看到**稀疏类别**
* 这里的**稀疏类别** 是关键，它少则不影响我们的模型，重则导致过拟合，这个一定要记下来（重新组合或者重新分配）

## segmentation 分割

 segmentation 是观察**分类特征** 与 **数值特征** 的最有效的方式

* Box plot 是合适的
* 这里的分析对 **评估模型的泛化能力**很重要

## 相关性分析

* 相关性分析 可以很容易的看到两个数值特征之间的关系
* 相关性 **heatmaps** 可以做到这一点（所有的数值都要放大 100 倍）
* 关键要分析，哪一个特征和target有很强烈的相关关系
* 是否有两个特征存在着意外的或者强烈的相关关系

## 数据清洗

* Better data beats fancier algorithms
* 在这块儿花费更多时间是合理的
* 不同类型的数据需要不同类型的清洗方法

1. 删除重复的或者无关的数据
    2. 重复数据：
        1. 合并来自多个地方的数据集
        2. 从客户或者其他地方接受的数据
        3. 拼凑的数据
    3. 无关数据：
        1. 探索性分析的内容
2. 结构错误
    1. 数据测量，数据传输中出现的错误
    2. 拼写错误或者不一致的大小写
    3. 这可以从**分类数据的条形图**中很容易看出来
3. 标记错误
    1. 如 IT 和 Information Technology 应该是一个意思
4. 离群值
    1. 线性回归模型对离群值很敏感，而决策树相对好一点
    2. 如果有**合理的理由**去除异常值，将是非常好的
    3. 不能因为是**离群值**就盲目的删除，或许，它可以提供很多有用的信息
    4. 删除数据必须要有理由，例如，**非真实数据的可能性**
5. 缺失值
    1. **不能简单地忽略数据集中的缺失值**
    2. 常见做法（丢弃或者插值）：**事实上这是很不明智的**
    3. 丢失缺失值比插值有时候更有效
    4. 尽可能的告诉算法：缺失了数据，因为数据缺失本身就是一种信息
6. 缺失分类数据
    1. 直接标记为“缺失”，即新添加一个类别
    2. 同时也达到了没有缺失值的要求
7. 缺失数值数据
    1. 标记并且填充值
    2. 将观察结果标记为缺失的指示变量
    3. 用 0 填充原来的缺失值（满足没有缺失值的要求）
    4. 这样的标记和填充手法可以让算法估计缺失的最佳常量，而不仅仅是使用平均值进行填充
    5. **直接丢弃或者插值**都是次优的方案（缺失同样是信息，所以最好的方法是使用标记的方法，并让算法自己去给出缺失值）

## 特征工程

特征工程是从已知的数据中**创建新的特征**作为输入

1. 数据清洗是一个 减 的过程而特征工程是一个 加 的过程

### 交互项

1. 是否可以创建一个**具有相互作用的特征**，他们是两个或者多个特征的组合（即是一个 **交互项**，可以是两个变量的 乘积，和或者差）
2. 应该询问自己“我可以把这两个变量组合起来以期望获得更好的结果吗？”
3. 例如，学校数量和学校质量两个变量（如果我们可以利用这两个变量做一个评分，那么结果是否会更加有效呢？）

### 稀疏类别

1. 每个类别至少有 50 个数据，当然这也取决于数据集的大小和其它特征数量
2. 将**相似的类别归为一个类**
3. 将**其它的数据量比较少的类别归属为一个 other 类**
4. ++**这里的 missing 是一个 单独的类（分类数据）**++

### 虚拟变量

1. 很多机器学习算法 **不能直接处理分类数据，尤其是文本值**， ++所以，我们需要对分类特征创建 虚拟变量++
2. 虚拟变量（0,1）等，每个值代表单一的 分类变量值
3. 尽可能多的引入 指示变量，这将会简化难度（分类变量以有缺失值的情况下）
 

### 删除没有使用或者 冗余的 特征
1. 未使用的特征
    1. ID 列
    2. 与预测无关
    3. 其它文本信息
2. 冗余特征
    1. 可以有其它的特征代替 

## 算法选择

* 大多数情况下，我们可以直接跳过 线性模型（不可以表达非线性关系；对离群值敏感；没有学到底层模型指示学到了训练集的噪声）
* 正则弧是一种**通过人为惩罚模型系数**来防止过度拟合
    * 可以防止系数过大
    * 也可以完全移除系数（正则项为0）
    * 惩罚的强度是可以 调整的

### 正则化回归

1. Lasso 回归
    1. 惩罚系数的 绝对大小
    2. 可能导致系数为 0
    3. 具有**自动特征选择**的功能
    4. 惩罚强度应该是可调节的
    5. 更强的惩罚必然导致大量系数为 0
2. Ridgo 回归
    1. 1. 惩罚系数的 平方大小
    2. 使系数变小，但不会为 0
    3. 岭回归 具有使 **特征收缩**的功能
3. Elastic-Net
    1. Lasso 和 Ridgo 的一种折中 
    2. 惩罚系数的 绝对大小和平方大小 的组合 mix
    3. 两种惩罚之间的系数 是可以调节的
    4. 总强度也是可以调节的

### 决策树

1. 由于 决策树 的分支结构，决策树可以很容易的构建 非线性关系
2. 不受约束 的决策树 很容易 过拟合

## 集成学习

1. bagging 试图**降低 复杂模型过拟合的风险**
    2. It trains a large number of "strong" learners in parallel.
    3. A strong learner is a model that's relatively unconstrained.
    4. Bagging then combines all the strong learners together in order to "smooth out" their predictions.
 
2. Boosting 方法**试图增加简单模型的预测灵活性**
    1. It trains a large number of "weak" learners in sequence.
    2. A weak learner is a constrained model (i.e. you could limit the max depth of each decision tree).
    3. Each one in the sequence focuses on learning from the mistakes of the one before it.
    4. Boosting then combines all the weak learners into a single strong learner.

3. Bagging 和 Boosting 都是集成学习的方法，但他们却从相反的方向来解决问题。Bagging 使用复杂的基本模型，并试图“平滑”他们的预测，而 Boosting 使用简单的基本模型，并试图“提高”总体的复杂性
4. 当 基本的模型是 **决策树** 的时候，Bagging 集成和 Boosting 集成 分别叫做 **random forests and boosted trees**

### 随机森林

随机森林有两个“随机性”来源：

1. 每棵树只允许从一个随机的特征子集中进行分割（导致特征选择）
2. 每棵树只在一个随机的观察子集上进行训练（一个称为重采样的过程）

在实践中，随机森林往往表现非常好，开箱即用

1. 他们经常击败需要几个星期才能建立起来的许多其他模型
2. 他们是完美的“瑞士军刀”算法，几乎总能获得好的结果
3. 他们没有很多复杂的参数来调整

### Boosted tree

1. 每棵树的最大深度可以被调整。
2. 序列中的每一棵树都会尝试纠正之前那棵树的预测错误

在实践中，提升的树木往往具有最高的性能上限

1. 经过适当的调整，他们经常击败许多其他类型的模型。
2. 调参比随机森林要复杂得多

## 参看文献

1. [Day 2: Exploratory Analysis - Machine Learning Crash Course](https://elitedatascience.com/exploratory-analysis)
2. [Day 3: Data Cleaning - Machine Learning Crash Course](https://elitedatascience.com/data-cleaning)
3. [How to Handle Imbalanced Classes in Machine Learning](https://elitedatascience.com/imbalanced-classes)
4. [Day 4: Feature Engineering - Machine Learning Crash Course](https://elitedatascience.com/feature-engineering)
5. [Day 5: Algorithm Selection - Machine Learning Crash Course](https://elitedatascience.com/algorithm-selection)
