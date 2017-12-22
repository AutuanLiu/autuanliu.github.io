# 处理类别不均衡
```
author: autuanliu
email: autuanliu@163.com
date: 2017/12/01
```
## 分割数据集

* 数据集的使用：
    * 训练模型
    * 评估模型
    * 不要重复使用
* 训练集(包括交叉验证)
* 测试集(最后才使用)
* 如果模型在训练集表现良好，但是在测试集表现非常差，那就说明出现了过拟合

## 调整模型

* 调整模型即意味着调整超参数
* 模型参数+超参数。模型参数可以从训练集中学习到，而超参数不可以
* 交叉验证可以 用于 调整模型，**也就是调整超参数**
* ++**交叉验证是只使用训练集来获得模型表现性能的可靠估计的一种方法**++
* 10折交叉验证的平均值是最终的 模型表现性能，also called your cross-validated score. Because you created 10 mini train/test splits, this score is usually pretty reliable

## Fit and tune models

* 大致过程
    
    ```
    For each algorithm (i.e. regularized regression, random forest, etc.):
        For each set of hyperparameter values to try:
            Perform cross-validation using the training set.
            Calculate cross-validated score.
    ```

    At the end of this process, you will have a cross-validated score for each set of hyperparameter values... for each algorithm.

    ```
    For each algorithm:
        Keep the set of hyperparameter values with best cross-validated score.
        Re-train the algorithm on the entire training set (without cross-validation).
    ```

        然后，将训练好的对应多种算法的多个模型在 **测试集**上进行测试，然后选出最好的那个模型

* 评估好坏
    * For regression tasks, we recommend Mean Squared Error (MSE) or Mean Absolute Error (MAE). (Lower values are better)
    * For classification tasks, we recommend Area Under ROC Curve (AUROC). (Higher values are better)
* 询问自己
    * Which model had the best performance on the test set? (performance)
    * Does it perform well across various performance metrics? (robustness)
    * Did it also have (one of) the best cross-validated scores from the training set? (consistency)
    * Does it solve the original business problem? (win condition) 

## Overfitting in Machine Learning: What It Is and How to Prevent
防止过度拟合

* Cross-validation
* Train with more data
* Remove features
* Early stopping
* Regularization
* Ensembling

## How to Handle Imbalanced Classes in Machine Learning

* 不平衡类出现在许多领域，包括：
    * 欺诈识别
    * 垃圾邮件过滤
    * 疾病筛查
    * SaaS订阅流失
    * 广告点击
* 处理方法
    * Up-sample Minority Class
        * 随机从最少的类别中 ++复制++ 信号，以加强少数类别的存在感（重采样）  
        1. 将每个类的观察分成不同的数据框
        2. 将使用replace方法对少数类重新抽样，并设置抽样数以匹配多数类
        3. 将把上采样的少数类DataFrame与原始的大多数类DataFrame结合起来
    * Down-sample Majority Class
        * 从多数类中去掉一部分，直接重新采样即可
        1. 将每个类的样本分成不同的数据框
        2. 将对大部分类重新取样，设置样本数量以匹配少数类别的样本数量
        3. 将下采样的多数类DataFrame与原始的少数类DataFrame组合起来
    * Change Your Performance Metric
        * 分类问题通常使用 AUROC 指标[classification - What does AUC stand for and what is it? - Cross Validated](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it)
        * 要计算AUROC，需要预测的类的概率，而不仅仅是预测类。可以使用predict_proba()函数获取
    * Penalize Algorithms (Cost-Sensitive Training)
        *  Penalized-SVM
        *  使用惩罚措施
    * Use Tree-Based Algorithms
        * 集成树（随机森林，梯度提升树）几乎总是胜过单一的 决策树
    * 将少数类别合在一起作为一个新的类
    * SMOTE算法


## 参考

1. [How to Handle Imbalanced Classes in Machine Learning](https://elitedatascience.com/imbalanced-classes)
2. [WTF is the Bias-Variance Tradeoff? (Infographic)](https://elitedatascience.com/bias-variance-tradeoff)
3. [Dimensionality Reduction Algorithms: Strengths and Weaknesses](https://elitedatascience.com/dimensionality-reduction-algorithms)
4. [Day 6: Model Training - Machine Learning Crash Course](https://elitedatascience.com/model-training)
5. [Day 7: Next Steps - Machine Learning Crash Course](https://elitedatascience.com/next-steps)
6. [Overfitting in Machine Learning: What It Is and How to Prevent It](https://elitedatascience.com/overfitting-in-machine-learning)
7. [The 5 Levels of Machine Learning Iteration](https://elitedatascience.com/machine-learning-iteration)