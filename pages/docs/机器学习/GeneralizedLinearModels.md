
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
* 必须将 相同 的缩放应用于对应的测试向量中，以获得有意义的结果
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
* 高斯过程是一种常用的监督学习方法，旨在解决 *回归问题* 和 *概率分类问题*

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
