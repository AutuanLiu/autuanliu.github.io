
# 广义线性模型
Sklearn 的学习笔记

具体实现见 [GitHub](https://github.com/AutuanLiu/Machine-Learning-on-docker/tree/master/sklearn)

##  1.1.1 普通最小二乘法
* 使 **残差平方和** 最小
* LinearRegression 会调用 fit 方法来拟合数组 X, y, 并且将线性模型的系数 w 存储在其成员变量 ``coef_``中
* 最小二乘估计对于随机误差非常敏感, 产生很大的方差
* 形式
$$
  y = xW^T+b
$$


## 1.1.2 岭回归
* 最小化带有L2正则项的残差平方和
* RidgeCV 通过内置的 Alpha 参数的交叉验证来实现岭回归. 该对象与 GridSearchCV 的使用方法相同, 只是它默认为 Generalized Cross-Validation(广义交叉验证 GCV), 这是一种有效的留一验证方法(LOO-CV)



## 1.1.3 Lasso
* 估计稀疏系数的线性模型
* 它在一些情况下是有用的, 因为它倾向于使用具有较少参数值的情况, 有效地减少给定解决方案所依赖变量的数量
* Lasso及其变体是压缩感知领域的基础. 在一定条件下, 它可以恢复一组非零权重的精确集
* 损失函数 
$$\min \frac 1 {2 n_{samples}}||Xw-y||_2^2 +\alpha||w||_1 $$
* 使用了 coordinate descent (坐标下降算法)来拟合系数
* 由于 Lasso 回归产生稀疏模型, 因此可以用于执行特征选择
* 对于具有许多线性回归的高维数据集,  LassoCV 最常见
* LassoLarsCV 在寻找 alpha parameter 参数值上更具有优势, 而且如果样本数量与特征数量相比非常小时, 通常 LassoLarsCV 比 LassoCV 要快
* 当使用 k-fold 交叉验证时, 正则化路径只计算一次而不是k + 1次, 所以找到α的最优值是一种计算上更便宜的替代方法

## 1.1.4 MultiTaskLasso 
* MultiTaskLasso是一个估计多元回归稀疏系数的线性模型： y 是一个 (n_samples, n_tasks) 的二维数组, 其约束条件和其他回归问题(也称为任务)是一样的, 都是所选的特征值 

## 1.1.5 弹性网
* 弹性网络 是一种使用L1,L2范数作为先验正则项训练的线性回归模型. 这种组合允许学习到一个只有少量参数是非零稀疏的模型, 就像 Lasso 一样, 但是它仍然保持 一些像 Ridge 的正则性质. 我们可利用 l1_ratio 参数控制L1和L2的凸组合
* 弹性网络在很多特征互相联系的情况下是非常有用的
* Lasso很可能只随机考虑这些特征中的一个, 而弹性网络更倾向于选择两个
* loss 
$$\min \frac 1 {2 n_{samples}}||Xw-y||_2^2 +\alpha||w||_1 + \frac{\alpha(1-\rho)} 2 ||w||_2^2 $$

ElasticNetCV 类可以通过交叉验证来设置参数alpha ($\alpha$) 和 l1_ratio ($\rho$) 

* MultiTaskElasticNet 是一个对多回归问题估算稀疏参数的弹性网络: Y 是一个二维数组, 形状是 (n_samples,n_tasks). 其限制条件是和其他回归问题一样, 是选择的特征, 也称为 tasks
* 采用了坐标下降法求解参数

## 1.1.7 最小角回归
* 最小角回归 (LARS) 是对**高维数据**的回归算法
* LARS和逐步回归很像. 在每一步, 它寻找与响应最有关联的 预测. 当有很多预测由相同的关联时, 它没有继续利用相同的预测, 而是在这些预测中找出应该等角的方向
* **它对噪声非常敏感**
* LassoLars 是一个使用LARS算法的lasso模型,  不同于基于坐标下降法的实现, 它可以得到一个精确解, 也就是一个 关于自身参数标准化后的一个分段线性解

## 1.1.10 贝叶斯回归
* 贝叶斯回归可以用于在预估阶段的参数正则化: 正则化参数的选择不是通过人为的选择, 而是通过手动调节数据值来实现
* 它能根据已有的数据进行改变. 
* 它能在估计过程中引入正则项. 
* 它的推断过程是非常耗时的

## 1.1.10.1 贝叶斯岭回归
* 参数 w, $\alpha$ 和 $\lambda$ 是在模型拟合的时候一起被估算出来的. 剩下的超参数就是 gamma 分布的先验了. $\alpha$ 和 $\lambda$ . 它们通常被选择为 没有信息量 . 模型参数的估计一般利用 最大似然对数估计法 . 
*  $\alpha_1 = \alpha_2 =  \lambda_1 = \lambda_2 = 10^{-6}$
* 贝叶斯岭回归对病态问题(ill-posed)的鲁棒性要更好

## 1.1.11 logistic 回归
* logistic 回归又被称作 logit regression(logit 回归), maximum-entropy classification(MaxEnt, 最大熵分类), 或 log-linear classifier(线性对数分类器)
* 该模型利用函数 logistic function 将单次试验(single trial)的输出转化并描述为概率
* scikit-learn 中 logistic 回归在 LogisticRegression 类中实现了**二元(binary)、一对余(one-vs-rest)及多元 logistic 回归, 并带有可选的 L1 和 L2 正则化**
* liblinear 应用了坐标下降算法(Coordinate Descent, CD
* CD算法训练的模型不是真正意义上的多分类模型, 而是基于 one-vs-rest 思想分解了这个优化问题, 为每个类别都训练了一个二元分类器
* lbfgs, sag 和 newton-cg solvers (求解器)只支持 L2 罚项, 对某些高维数据收敛更快. 这些求解器的参数 `multi_class`设为 multinomial 即可训练一个真正的多元 logistic 回归 , 其预测的概率比默认的 one-vs-rest 设定更为准确
* sag 求解器基于平均随机梯度下降算法(Stochastic Average Gradient descent). 在大数据集上的表现更快, 大数据集指样本量大且特征数多
* saga solver 是 sag 的一类变体, 它支持非平滑(non-smooth)的 L1 正则选项 penalty=l1 . 因此对于稀疏多元 logistic 回归 , 往往选用该求解器

Case|Solver
---|---
L1正则|liblinear or saga
多元损失|lbfgs, sag, saga or newton-cg
大数据集|sag or saga


* saga 一般都是最佳的选择,但出于一些历史遗留原因默认的是 liblinear

* SGDClassifier 和 SGDRegressor 分别用于拟合分类问题和回归问题的线性模型, 可使用不同的(凸)损失函数, 支持不同的罚项

## 1.1.13 Perceptron感知机
* Perceptron 是适用于 large scale learning(大规模学习)的一种简单算法
* 不需要设置学习率(learning rate). 
* 不需要正则化处理. 
* 仅使用错误样本更新模型
* 合页损失(hinge loss)的感知机比SGD略快, 所得模型更稀疏

## 1.1.16 多项式回归：用基函数展开线性模型
*  polynomial regression 是线性模型中的同一类, 我们认为以上(即模型是线性 ), 可以用同样的方法解决
* poly = PolynomialFeatures(degree=2) 和 poly.fit_transform(X) 用于实现非线性到线性的转换
* 利用多项式特征训练的线性模型能够准确地恢复输入多项式系数


## 1.2 线性和二次判别分析
* Linear Discriminant Analysis(线性判别分析)(discriminant_analysis.LinearDiscriminantAnalysis) 和 Quadratic Discriminant Analysis (二次判别分析)(discriminant_analysis.QuadraticDiscriminantAnalysis) 是两个经典的分类器. 正如他们名字所描述的那样, 他们分别代表了**线性决策平面和二次决策平面**
* 其天生的多分类特性, 在实践中已经证明很有效, 并且 **不需要再次调参**
* 线性判别分析只能学习线性边界,  而二次判别分析则可以学习二次函数的边界, 因此它相对而言更加灵活
* 线性判别分析降维 是总体而言十分强大的降维方式, 同样也 **仅仅在多分类环境下才会起作用**
* `discriminant_analysis.LinearDiscriminantAnalysis.transform`
* LDA 分类器中存在一个利用线性投影到 K-1 个维度空间的降维工具
* 收缩是一个在训练样本数量相比特征而言很小的情况下可以提升预测(准确性)的协方差矩阵
* LDA 收缩可以通过设置 discriminant_analysis.LinearDiscriminantAnalysis 类的 shrinkage 参数为 ‘auto’ 以得到应用. 

## 1.3 核岭回归

* Kernel ridge regression (KRR) (内核岭回归由 使用内核方法的 (岭回归)(使用 l2 正则化的最小二乘法)所组成. 因此, 它所学习到的在空间中不同的线性函数是由不同的内核和数据所导致的. 对于非线性的内核, 它与原始空间中的非线性函数相对应
* 由 KernelRidge 学习的模型的形式与支持向量回归( SVR ) 是一样的. 但是他们使用不同的损失函数：内核岭回归(KRR)使用 squared error loss (平方误差损失函数)而 support vector regression (支持向量回归)(SVR)使用 $\epsilon$ -insensitive loss , 两者都使用 l2 regularization (l2 正则化). 与 SVR 相反, 拟合 KernelRidge 可以以 closed-form (封闭形式)完成, 对于中型数据集通常更快. 另一方面, 学习的模型是非稀疏的, 因此比 SVR 慢,  在预测时间, SVR 学习了 $\epsilon>0$ 的稀疏模型. 
* 它们的 learned functions (学习函数)非常相似;但是, 拟合 KernelRidge 大约比拟合 SVR 快七倍(都使用 grid-search ( 网格搜索 ) ). 然而, 由于 SVR 只学习了一个稀疏模型, 所以 SVR 预测 10 万个目标值比使用 KernelRidge 快三倍以上. SVR 只使用了百分之三十的数据点做为支撑向量
* 对于中型训练集(小于 1000 个样本), 拟合 KernelRidge 比 SVR 快; 然而, 对于更大的训练集 SVR 通常更好. 关于预测时间, 由于学习的稀疏解, SVR 对于所有不同大小的训练集都比 KernelRidge 快
* **稀疏解**是说只使用一部分特征作为关键训练特征

## 1.4 SVM

支持向量机可以用于 **分类, 回归, 异常检测**

优势:

1. 在高维空间中非常高效
2. 即使在**数据维度比样本数量大**的情况下仍然有效
3. 在决策函数(称为支持向量)中使用训练集的子集
4. 通用性: 不同的核函数与特定的决策函数一一对应

劣势:

1. 如果特征数量比样本数量大得多, 在选择核函数时要避免过拟合

当类别不均衡时, 使用带有 类别权重的 

```python
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
```

支持向量分类的方法可以被扩展用作解决回归问题. 这个方法被称作支持向量回归. 支持向量分类生成的模型只依赖于训练集的子集,因为构建模型的 cost function 不在乎边
缘之外的训练点. 类似的, 支持向量回归生成的模型只依赖于训练集的子集, 因为构建模型的 cost function 忽略任何接近于模型预测的训练数据. SVM的核心是一个二次规划
问题(Quadratic Programming, QP), 是将支持向量和训练数据的其余部分分离开来, 支持向量机是个强大的工具, 不过它的计算和存储空间要求也会随着要训练向量的数目增
加而快速增加

* 惩罚系数C的设置:在合理的情况下, C 的默认选择为 1. 如果您有很多混杂的观察数据, 您应该要去调小它. C 越小, 就能更好地去正规化估计
* 常用核函数
    * 线性 
$$ k(x_i, x_j) = x_i^T x_j $$

    * 多项式 
$$ k(x_i, x_j) = (x_i^T x_j)^d $$

    * 高斯核rbf 
$$ k(x_i, x_j) = exp(-\frac{||x_i - x_j||^2}{2 \sigma^2}) $$

    * 拉普拉斯 
$$ k(x_i, x_j) = exp(-\frac{||x_i - x_j||}{\sigma}) $$

    * sigmoid 
$$ k(x_i, x_j) = \tanh(\beta x_i^T x_j + \theta) $$

* 核函数通过创建实例时进行指定 
    svm.SVC(kernel='linear')

* 自定义核函数

```python
def my_kernel(x, y):
  return np.dot(x, y.T)
```

* 当用 径向基(RBF)内核去训练 SVM, 有两个参数必须要去考虑: C, $\gamma$. 较小的 C 会使决策表面更平滑, 同时较高的 C 旨在正确地分类所有训练样本. Gamma 定义了单一 训练样本能起到多大的影响. 较大的 gamma 会更让其他样本受到影响

## 1.5 随机梯度下降

* SGD 对特征缩放 敏感
* 必须将 相同 的缩放应用于对应的测试向量中, 以获得有意义的结果
* 在拟合模型前要确保训练数据经过了打乱 `shuffle=True`

```python
from sklearn.linear_model import SGDClassifier

X = [[0, 0], [1, 1]]
y = [0, 1]

clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=500, tol=1e-3)
clf.fit(X, y)
# prediction
res = clf.predict([[2, 2]])
print(res)
```
* SGDClassifier 通过在将多个 binary classifiers组合在 one versus all(OVA) 方案中来支持多类分类
* SGDRegressor 类实现了一个简单的随即梯度下降学习程序, 它支持不同的损失函数和惩罚来拟合线性回归模型

## 1.6 最近邻

* 无监督的最近邻是许多其它学习方法的基础, 尤其是 manifold learning 和 spectral clustering
* 受监督的 neighbors-based学习分为两种 classification 针对的是具有离散标签的数据, regression 针对的是具有连续标签的数据
* 非参数方法
* 应用于决策边界非常不规则的情景下

### 最近邻分类

* 分类是由每个点的最近邻的简单多数投票中计算得到的
* 一个查询点的数据类型是由它最近邻点中最具代表性的数据类型来决定的
* k 值的最佳选择是高度数据依赖的
* 通常较大的 k 是会抑制噪声的影响, 但是使得分类界限不明显
* 如果数据是不均匀采样的, 那么 RadiusNeighborsClassifier 中的基于半径的近邻分类可能是更好的选择
* 指定一个固定半径 r, 使得稀疏邻居中的点使用较少的最近邻来分类

### 最近邻回归

* 最近邻回归是用在数据标签为连续变量, 而不是离散变量的情况下
* 分配给查询点的标签是由它的 **最近邻标签的均值** 计算而来的
* 在某些环境下, 增加权重可能是有利的, 使得附近点对于回归所作出的贡献多于远处点. 这可以通过 weights 关键字来实现
* 高斯过程是一种常用的监督学习方法, 旨在解决 *回归问题* 和 *概率分类问题*

## 1.8 交叉分解

* 偏最小二乘法(PLS)和典型相关分析(CCA)
* **具有发现两个多元数据集之间的线性关系** 的用途: fit method(拟合方法)的参数 X 和 Y 都是 2 维数组 

## 1.9 朴素贝叶斯

* 朴素贝叶斯方法是基于贝叶斯定理的一组有监督学习算法
* 简单地假设 **每对特征之间相互独立**, 朴素的由来
* 高斯朴素贝叶斯 的参数使用 最大似然估计
* 多项分布朴素贝叶斯
* 伯努利朴素贝叶斯
* 堆外朴素贝叶斯模型拟合

## 1.10 决策树

* 无参监督学习方法
* 回归与分类
* 目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值
* 决策树模型容易产生一个过于复杂的模型, 这样的模型对数据的泛化性能会很差
* 决策树可能是不稳定的, 因为数据中的微小变化可能会导致完全不同的树生成

## 1.11 集成方法

集成方法 的目标是把使用给定学习算法构建的多个基估计器的预测结果结合起来, 从而获得比单个估计器更好的泛化能力/鲁棒性

* 平均方法, 该方法的原理是构建多个独立的估计器, 然后取它们的预测结果的平均. 一般来说组合之后的估计器是会比单个估计器要好的, 因为它的**方差**减小了. 
    * Bagging 方法, 随机森林, …

* 在 boosting 方法 中, 基估计器是依次构建的, 并且每一个基估计器都尝试去减少组合估计器的**偏差**. 这种方法主要目的是为了结合多个弱模型, 使集成的模型更加强大. 
    * AdaBoost, 梯度提升树, …

bagging 方法会在原始训练集的随机子集上构建一类黑盒估计器的多个实例,然后把这多个估计器的预测结果结合起来形成最终的预测结果. 该方法通过在构建模型的过程中引入随机性,来减少基估计器的方差(例如,决策树). 在多数情况下,bagging 方法提供了一种非常简单的方式来对单一模型进行改进,而无需修改背后的算法. 因为 bagging 方法可以减小过拟合,所以通常最适合在强分类器和复杂模型上使用,相比之下 boosting 方法则在弱模型上表现更好

bagging 方法有很多种,区别大多数在于抽取训练子集的方法：

* 抽取的数据集是对于样例抽取的子集,我们叫做Pasting
* 样例抽取是有放回的,我们称为 Bagging
* 抽取的数据集的随机子集是对于特征抽取的随机子集,我们叫做Random Subspaces
* 估计器构建在对于样本和特征抽取的子集之上时,我们叫做Random Patches

集成模型中的每棵树构建时的样本都是由训练集经过有放回抽样得来的. 另外,在构建树的过程中进行结点分割时,选择的分割点不再是所有特征中最佳分割点,而是特征的一个随机子集中的最佳分割点. 由于这种随机性,森林的偏差通常会有略微的增大(相对于单个非随机树的偏差),但是由于取了平均,其方差也会减小,通常能够补偿偏差的增加,从而产生一个总体上更好的模型

* scikit-learn 的实现是取每个分类器预测概率的平均,而不是让每个分类器对类别进行投票

在极限随机树中ExtraTreesClassifier 和 ExtraTreesRegressor, 计算分割点方法中的随机性进一步增强. 在随机森林中,使用的特征是**候选特征的随机子集**; 不同于寻找最具有区分度的阈值, 这里的阈值是针对每个候选特征随机生成的,并且选择这些随机生成的阈值中的最佳者作为分割规则. 这种做法通常能够减少一点模型的方差,代价则是略微地增大偏差

回归问题中使用 max_features = n_features, 分类问题使用 max_features = sqrt(n_features (其中 n_features 是特征的个数)是比较好的默认值. max_depth = None 和 min_samples_split = 2 结合通常会有不错的效果(即生成完全的树). 请记住,这些(默认)值通常不是最佳的,同时还可能消耗大量的内存,最佳参数值应由交叉验证获得

如果设置 n_jobs = k, 则计算被划分为 k 个作业, 并运行在机器的 k 个核上.  如果设置 n_jobs = -1, 则使用机器的所有核

特征对目标变量预测的相对重要性可以通过(树中的决策节点的)特征使用的相对顺序(即深度)来进行评估.  决策树顶部使用的特征对更大一部分输入样本的最终预测决策做出贡献；因此, 可以使用接受每个特征对最终预测的贡献的样本比例来评估该 特征的相对重要性

**首先应当评估各个特征的重要性**

http://sklearn.apachecn.org/cn/0.19.0/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py

### AdaBoost

AdaBoost 的核心思想是用反复修改的数据(主要是修正数据的权重)来学习一系列的弱学习器(一个弱学习器模型仅仅比随机猜测好一点, 比如一个简单的决策树),由这些弱学习器的预测结果通过加权投票(或加权求和)的方式组合, 得到我们最终的预测结果. 在每一次所谓的提升(boosting)迭代中, 数据的修改由应用于每一个训练样本的(新) 的权重 w_1, w_2, …, w_N 组成(即修改每一个训练样本应用于新一轮学习器的权重).  初始化时,将所有弱学习器的权重都设置为  w_i = 1/N ,因此第一次迭代仅仅是通过原始数据训练出一个弱学习器. 在接下来的 连续迭代中,样本的权重逐个地被修改,学习算法也因此要重新应用这些已经修改的权重. 在给定的一个迭代中, 那些在上一轮迭代中被预测为错误结果的样本的权重将会被增加, 而那些被预测为正确结果的样本的权 重将会被降低. 随着迭代次数的增加, 那些难以预测的样例的影响将会越来越大, 每一个随后的弱学习器都将 会被强迫更加关注那些在之前被错误预测的样例

### Gradient Tree Boosting

Gradient Tree Boosting 或梯度提升回归树(GBRT)是对于任意的可微损失函数的提升算法的泛化. GBRT 是一个准确高效的现有程序, 它既能用于分类问题也可以用于回归问题. 梯度树提升模型被应用到各种领域, 包括网页搜索排名和生态领域.


回归树基学习器的大小定义了可以被梯度提升模型捕捉到的变量(即特征)相互作用(即多个特征共同对预测产生影响)的程度. 通常一棵深度为 h 的树能捕获到秩为 h 的相互作用

* 目前支持的损失函数, 具体损失函数可以通过参数 loss 指定:

    * 回归 (Regression)
        * Least squares ('ls'): 由于其优越的计算性能,该损失函数成为回归算法中的自然选择.  初始模型通过目标值的均值给出.
        * Least absolute deviation ('lad'): 回归中具有鲁棒性的损失函数,初始模型通过目 标值的中值给出.
        * Huber ('huber'): 回归中另一个具有鲁棒性的损失函数,它是最小二乘和最小绝对偏差两者的结合. 其利用 alpha 来控制模型对于异常点的敏感度
        * Quantile ('quantile'): 分位数回归损失函数.用 0 < alpha < 1 来指定分位数这个损失函数可以用来产生预测间隔
    * 分类 (Classification)
        * Binomial deviance ('deviance'): 对于二分类问题(提供概率估计)即负的二项log似然 损失函数.模型以log的比值比来初始化.
        * Multinomial deviance ('deviance'): 对于多分类问题的负的多项log似然损失函数具有 n_classes 个互斥的类.提供概率估计. 初始模型由每个类的先验概率给出.在每一次迭代中 n_classes 回归树被构建,这使得 GBRT 在处理多类别数据集时相当低效.
        * Exponential loss ('exponential'): 与 AdaBoostClassifier 具有相同的损失 函数与 'deviance' 相比, 对具有错误标记的样本的鲁棒性较差,仅用于在二分类问题

在拟合一定数量的弱分类器时, 参数 learning_rate 和参数 n_estimators 之间有很强的制约关系. 较小的 learning_rate 需要大量的弱分类器才能保证训练误差的不变.经验表明数值较小的 learning_rate 将会得到更好的测试误差. 

## XGBoost

* XGBoost 是 Extreme Gradient Boosting 的缩写
* XGBoost 用于监督学习问题
* 正则化项控制模型的复杂性, 这有助于避免过拟合
* 树集成模型是一组 classification and regression trees(CART)
* CART 与 decision trees(决策树)有些许的不同, 就是叶子只包含决策值. 在 CART 中, 每个叶子都有一个 real score (真实的分数), 这给了我们更丰富的解释, 超越了分类

* 支持导入的数据格式
    * libsvm txt format file
    * numpy 的 2D 数组
    * xgboost binary buffer file

* 数据将会被存在一个名为 DMatrix 的对象中

* 加载 ligbsvm 文本格式或者 XGBoost 二进制文件到 DMatrix 对象中

```python
dtrain = xgb.DMatrix('train.svm.txt')
dtest = xgb.DMatrix('test.svm.buffer')
```

* 加载 numpy 的数组到 DMatrix 对象中

```python
data = np.random.rand(5,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix( data, label=label)
```

* 加载 scpiy.sparse 数组到 DMatrix 对象中

```python
csr = scipy.sparse.csr_matrix((dat, (row, col)))
dtrain = xgb.DMatrix(csr)
```

* 保存 DMatrix 到 XGBoost 二进制文件中后, 会在下次加载时更快

```python
dtrain = xgb.DMatrix('train.svm.txt')
dtrain.save_binary("train.buffer")
```

* 处理 DMatrix 中的缺失值, 可以通过指定缺失值的参数来初始化 DMatrix

```python
dtrain = xgb.DMatrix(data, label=label, missing = -999.0)
```

* xgboost 使用 字典 来保存参数
    * booster 参数
    ```python
    param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    ```
    * 可以指定多个评估指标
    ```python
    param['eval_metric'] = ['auc', 'ams@0'] 

    # alternativly:
    # plst = param.items()
    # plst += [('eval_metric', 'ams@0')]
    ```

    * 指定 模式

    ```python
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    ```

    * 训练模型

    ```python
    num_round = 10
    bst = xgb.train( plst, dtrain, num_round, evallist )
    ```

    * 保存

    ```python
    bst.save_model('0001.model')
    ```

    * 加载和转存

    ```python
    dump_model()
    load_model()
    ```

* 如果您有一个验证集, 你可以使用提前停止找到最佳数量的 boosting rounds（梯度次数）. 提前停止至少需要一个 evals 集合. 如果有多个, 它将使用最后一个.

```python
train(..., evals=evals, early_stopping_rounds=10)
```

* 如果提前停止，模型将有三个额外的字段: bst.best_score, bst.best_iteration 和 bst.best_ntree_limit. 请注意 train() 将从上一次迭代中返回一个模型, 而不是最好的一个.

* 这与两个度量标准一起使用以达到最小化（RMSE, 对数损失等）和最大化（MAP, NDCG, AUC）. 请注意, 如果您指定多个评估指标, 则 param ['eval_metric'] 中的最后一个用于提前停止

* 预测

```python
predict()
```

* 绘图

```python
# 重要性
xgb.plot_importance(bst)
# 树
xgb.plot_tree(bst, num_trees=2)
xgb.to_graphviz(bst, num_trees=2)
```

#### xgboost 库的主要 API

* class xgboost.DMatrix() # 类
    * get_label()
    * get_weight()
    * num_col()
    * save_binary(fname, silent=True)
    * set_label(label)
    * slice(rindex)
* class xgboost.Booster(params=None, cache=(), model_file=None) # 类
    * eval(data, name='eval', iteration=0)
    * get_score(fmap='', importance_type='weight')
    * get_split_value_histogram(feature, fmap='', bins=None, as_pandas=True)
    * predict(data, output_margin=False, ntree_limit=0, pred_leaf=False, pred_contribs=False, approx_contribs=False)
    * save_model(fname)
* xgboost.train() # 函数
    * params (dict) – Booster params
    * dtrain (DMatrix) – Data to be trained
    * num_boost_round (int) – Number of boosting iterations
    * early_stopping_rounds (int) – Activates early stopping
    * learning_rates
* xgboost.cv() # 函数
* class xgboost.XGBRegressor() # 类, sklearn wrapper
* class xgboost.XGBClassifier()
* xgboost.plot_importance() # 函数
* xgboost.plot_tree() # 函数
* xgboost.to_graphviz() # 函数

1. [xgboost/demo at master · dmlc/xgboost](https://github.com/dmlc/xgboost/tree/master/demo)
2. [xgboost/demo/guide-python at master · tqchen/xgboost](https://github.com/tqchen/xgboost/tree/master/demo/guide-python)
3. [Python API Reference — xgboost 0.6 documentation](https://xgboost.readthedocs.io/en/latest/python/python_api.html)

### 解释性

* 特征重要性 

单个决策树本质上是通过选择最佳切分点来进行特征选择.这个信息可以用来检测每个特征的重要性.基本思想是：在树 的分割点中使用的特征越频繁, 特征越重要.

* 部分依赖 (Partial dependence)

VotingClassifier (投票分类器)的原理是结合了多个不同的机器学习分类器, 并且采用多数表决(majority vote)或者平均预测概率(软投票)的方式来预测分类标签. 这样的分类器可以用于一组同样表现良好的模型, 以便平衡它们各自的弱点

## 1.12 多类和多标签算法

* 1 vs 1
    * sklearn.svm.NuSVC
    * sklearn.svm.SVC
    * sklearn.gaussian_process.GaussianProcessClassifier 
* 1 vs many
    * GradientBoostingClassifier
    * GaussianProcessClassifier
    * LogisticRegression
    * PassiveAggressiveClassifier
    * Perceptron

## 特征选择

* VarianceThreshold 是特征选择的一个简单基本方法，它会移除所有那些方差不满足一些阈值的特征, 默认情况下，它将会移除所有的零方差特征，即那些在所有的样本上的取值均不变的特征
* 单变量的特征选择是通过基于单变量的统计测试来选择最好的特征。它可以当做是评估器的预处理步骤
* 可以通过 统计测试 进行特征选择
    * 将得分函数作为输入，返回单变量的得分和 p 值 （或者仅仅是 SelectKBest 和 SelectPercentile 的分数）:
        * 对于回归: f_regression , mutual_info_regression
        * 对于分类: chi2 , f_classif , mutual_info_classif
    * 这些基于 F-test 的方法计算两个随机变量之间的线性相关程度
    * mutual information methods（互信息）能够计算任何种类的统计相关性，但是作为非参数的方法，互信息需要更多的样本来进行准确的估计
* 不要使用一个回归评分函数来处理分类问题
* 递归式特征消除
* recursive feature elimination ( RFE ) 通过考虑越来越小的特征集合来递归的选择特征
* Linear models 使用 L1 正则化的线性模型会得到稀疏解
* 基于 Tree（树）的特征选取 可以用来计算特征的重要性，然后可以消除不相关的特征
* 特征选择通常在实际的学习之前用来做预处理。在 scikit-learn 中推荐的方式是使用 sklearn.pipeline.Pipeline

## 概率校准

## 神经网络
