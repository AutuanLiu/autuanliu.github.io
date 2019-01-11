# 数据科学基本路径

```
author: autuanliu
email: autuanliu@163.com
date: 2017/11/27
```

## 主要元素

* 人为的监督：设置路径
* 数据质量
* 过拟合

## 基本路径

![](http://ozesj315m.bkt.clouddn.com/notesImgs/blueprint.png)
![](http://ozesj315m.bkt.clouddn.com/notesImgs/blueprint1.png)

## 需要避免的错误

* **Spending too much time on theory**
    * Balance your studies with projects that provide you hands-on practice.
    * Learn to be comfortable with partial knowledge. You'll naturally fill in the gaps as you progress.
    * Learn how each piece fits into the big picture 
* **Coding too many algorithms from scratch**
    * Pick up general-purpose machine learning libraries, such as Scikit-Learn (Python) or Caret (R).
    * If you do code an algorithm from scratch, do so with the intention of learning instead of perfecting your implementation.
    * Understand the landscape of modern machine learning algorithms and their strengths and weaknesses. 
* **Jumping into the deep end**
    * First, master the techniques and algorithms of "classical" machine learning, which serve as building blocks for advanced topics.
    * Know that classical machine learning still has incredible untapped potential. While the algorithms are already mature, we are still in the early stages of discovering fruitful ways to use them.
    * Learn a systematic approach to solving problems with any form of machine learning

* 在理论上花费太多时间
* 从头开始编写太多的算法
* 跳入高级主题，例如深入学习，太快了
* 简历中有太多的技术术语
* 高估学位的价值
* 过于狭隘地寻找工作
* 面试时没有准备讨论项目
* 低估领域知识的价值
* 忽视沟通技巧

## 学习资源

[65 Free Data Science Resources for Beginners](https://elitedatascience.com/data-science-resources)

## 各种算法的比较与优缺点

* 随机森林通常优于SVM
* 聚类是无监督的（即没有“正确的答案”），数据可视化通常用于评估结果
* K-Means是一种通用算法
* 如果数据中真实的底层聚类不是球状的，那么K-Means将产生不好的聚类
* 分层聚类的主要优点是不会假设球体是球状的。另外，它可以很好地扩展到更大的数据集
* DBSCAN是一种基于密度的算法，可以为密集的点区域生成群集，还有一个最近的新发展称为HDBSCAN，允许密度不同的群集
* 线性回归可以直观地理解和解释，并且可以正则化以避免过度拟合。另外，使用随机梯度下降的新数据可以很容易地更新线性模型
* 决策树可以学习非线性关系，并且对异常值相当健壮
* 随机森林 更有效
* 深度学习算法通常不适合作为通用算法，因为它们需要大量的数据
* 当存在多个或非线性决策边界时，Logistic回归往往表现不佳。它们不够灵活，无法自然地捕捉到更复杂的关系
* 由于选择正确的核函数的重要性，SVM是内存密集型的，难以调整，并且不能很好地扩展到较大的数据集，目前在行业中，随机森林通常优于SVM
* 特征选择用于从数据集中筛选不相关或冗余的特征。特征选择和提取的主要区别在于，特征选择保留了原始特征的一个子集，而特征提取创建了全新的特征
* 一些监督算法已经具有  **内置的特征选择**，例如正则化回归和随机森林
* 作为一个独立的任务，特征选择可以是无监督的（例如方差阈值）或监督的（例如遗传算法）
* Variance thresholds remove features whose values don't change much from observation to observation (i.e. their variance falls below a threshold). These features provide little value.
* Correlation thresholds remove features that are highly correlated with others (i.e. its values change very similarly to another's). These features provide redundant information.
* Which one should you remove? Well, you'd first calculate all pair-wise correlations. Then, if the correlation between a pair of features is above a given threshold, you'd remove the one that has larger mean absolute correlation with other features
* GA(遗传算法)有两个主要用途。首先是 优化，比如找到神经网络的最佳权重。第二个是监督功能选择
* 一些算法已经具有  内置的特征提取。最好的例子是深度学习，它通过每个隐藏的神经层提取越来越有用的原始输入数据表示
* 主成分分析（PCA）是一种无监督算法，可以创建原始特征的线性组合
* You should always normalize your dataset before performing PCA because the transformation is dependent on scale. If you don't, the features that are on the largest scale would dominate your new principal components
* 与PCA不同，LDA(线性判别分析)不能最大程度地解释方差。相反，它最大化了类之间的可分性
* 自动编码器是经过训练重构其原始输入的神经网络
* 自动编码器是神经网络，这意味着它们对于某些类型的数据（如图像和音频数据）表现良好。
但是他们需要更多的数据来训练。它们不被用作通用维度降低算法

1. [现代机器学习算法：优点和缺点](https://elitedatascience.com/machine-learning-algorithms)
2. [维度降低算法：优点和缺点](https://elitedatascience.com/dimensionality-reduction-algorithms)

## 数据集

1. [Datasets for Data Science and Machine Learning](https://elitedatascience.com/datasets)
2. [UCI](http://archive.ics.uci.edu/ml/)
3. [RotoWire.com - Fantasy Baseball, Football, Basketball, Hockey and More](https://www.rotowire.com/)
5. [数据集](https://www.quandl.com/open-data)

## 新手项目

1. [8个有趣的机器学习项目初学者](https://elitedatascience.com/machine-learning-projects-for-beginners)
2. **体育运动也是练习++数据可视化和探索性分析++的优秀领域**
3. [Python Machine learning with SKLearn Tutorial for Investing - Intro - YouTube](https://www.youtube.com/watch?v=URTZ2jKCgBc&list=PLQVvvaa0QuDd0flgGphKCej-9jp-QdzZ3&index=1)
4. [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/)
5. [Logistic Regression from Scratch in Python - nick becker](https://beckernick.github.io/logistic-regression-from-scratch/)
6. [Tutorial To Implement k-Nearest Neighbors in Python From Scratch - Machine Learning Mastery](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
7. [K-nearest neighbor algorithm implementation in Python from scratch](http://dataaspirant.com/2016/12/27/k-nearest-neighbor-algorithm-implementaion-python-scratch/)
8. [为疾病预测建立有意义的机器学习模型](https://shiring.github.io/machine_learning/2017/03/31/webinar_code)

## 参考资料
1. [Day 1: Bird's Eye View - Machine Learning Crash Course](https://elitedatascience.com/birds-eye-view)
2. [9 Mistakes to Avoid When Starting Your Career in Data Science](https://elitedatascience.com/beginner-mistakes)
3. [65 Free Data Science Resources for Beginners](https://elitedatascience.com/data-science-resources)
4. [Python Machine Learning Tutorial, Scikit-Learn: Wine Snob Edition](https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn)
5. [Modern Machine Learning Algorithms: Strengths and Weaknesses](https://elitedatascience.com/machine-learning-algorithms)
6. [Learn Python - Free Interactive Python Tutorial](https://www.learnpython.org/)

