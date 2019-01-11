# TensorFlow 基础

* TensorFlow
    * 使用图 (graph) 来表示计算任务.
    * 在被称之为 会话 (Session) 的上下文 (context) 中执行图.
    * 使用 tensor 表示数据.
    * 通过 变量 (Variable) 维护状态.
    * 使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据
    * 每个 Tensor 是一个类型化的多维数组
    * 在 Python 语言中, 返回的 tensor 是 numpy ndarray 对象

## 计算图
* TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段. 在构建阶段, op 的执行步骤 被描述成一个图. 在执行阶段, 使用会话执行执行图中的 op
* TensorFlow Python 库有一个默认图 (default graph), op 构造器可以为其增加节点. 这个默认图对 许多程序来说已经足够用
* Session 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作, 所以 with 的方式是更加合理的

```python
sess.close()
with tf.Session() as sess:
    pass
```

* with...Device 语句用来指派特定的 CPU 或 GPU 执行操作:

```python
with tf.Session() as sess:
    with tf.device("/gpu:1"):
```

* 目前支持的设备包括:
    * "/cpu:0" : 机器的 CPU.
    * "/gpu:0" : 机器的第一个 GPU, 如果有的话.
    * "/gpu:1" : 机器的第二个 GPU, 以此类推

* 为了便于使用诸如 IPython 之类的 **Python 交互环境, 可以使用 InteractiveSession 代替 Session 类**, 使用Tensor.eval() 和 Operation.run() 方法代替 Session.run() . 这样可以避免使用一个变量来持有会话

```python
sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
# 使用初始化器 initializer op 的 run() 方法初始化 'x'
x.initializer.run()
# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
sub = tf.sub(x, a)
print(sub.eval())
```

* 在调用 run() 执行表达式之前, 并不会真正执行操作

## fetch

* 为了取回操作的输出内容, 可以在使用 Session 对象的 run() 调用 执行图时, 传入一些 tensor, 这些 tensor 会帮助你取回结果

```python
with tf.Session() as sess:
    result = sess.run([mul, intermed])
```

## feed

* TensorFlow 还提供了 feed 机制, 该机制 可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor
* feed 只在调用它的方法内有效, 方法结束, feed 就会消失
* 最常见的用例是将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符

```python
input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1, input2)
with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
```

* softmax模型可以用来给不同的对象分配概率。即使在之后，我们训练更加精细的模型时，最后一步也需要用softmax来分配概率
* softmax可以看成是一个激励（activation）函数或者链接（link）函数，把我们定义的线性函数的输出转换成我们想要的格式
* softmax：归一化 $exp(x)$

$$\text{softmax}(x)=\text{normalize}(\exp(x))$$

$$\text{softmax}(x)_i=\frac{\exp(x_i)}{\sum_j\exp()x_i}$$

* 交叉熵(信息论, 可以简单理解为是测量两个概率之间距离的一个量)
(y' 是预测值, y 是真实值)

$$H_{y}(y')\sum_i y_{i}\log(y'_i)$$

* tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值
* 使用了占位符的 tensor 计算的时候必须要 feed
* 只在feed 的时候才会使用真实值，其他的时候都要使用 占位符代替
* 涉及到占位符的时候才需要 feed 变量 只需要 run 即可，也继自动微分部分


