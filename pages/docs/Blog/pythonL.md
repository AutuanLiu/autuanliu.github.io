## Python 随笔

### 迭代器 

* 迭代器对象要求支持迭代器协议的对象，在`Python`中，支持迭代器协议就是实现对象的 `__iter__()` 和 `next()` 方法。
* 其中`__iter__()`方法返回迭代器对象本身；`next()` 方法返回容器的下一个元素，在结尾时引发 `StopIteration` 异常
* 对于可迭代对象，可以使用内建函数`iter()`来获取它的迭代器对象
* 当我们使用`for`语句的时候，`for`语句就会自动的通过 `__iter__()` 方法来获得迭代器对象，并且通过`next()`方法来获取下一个元素

#### 自定义迭代器
```python
class MyRange(object):
    def __init__(self, n):
        self.idx = 0
        self.n = n
 
    def __iter__(self):
        return self

    def next(self):
        if self.idx < self.n:
            val = self.idx
            self.idx += 1
            return val
        else:
            raise StopIteration()

# test example
>>> myRange = MyRange(3)
>>> for i in myRange:
...     print i
```

* 对于一个可迭代对象，如果它本身又是一个迭代器对象，就没有办法支持多次迭代
* 可以分别定义`可迭代类型对象`和`迭代器类型对象`；然后可迭代类型对象的`__iter__()`方法可以获得一个迭代器类型的对象

```python
class Zrange:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return ZrangeIterator(self.n)

class ZrangeIterator:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def next(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i
        else:
            raise StopIteration()    

# test example
zrange = Zrange(3)     
print([i for i in zrange])
print([i for i in zrange])
```
* 所以 一般情况下 这两者都是分开定义的

### 生成器
* 在Python中，使用生成器可以很方便的支持迭代器协议。生成器通过生成器函数产生，生成器函数可以通过常规的def语句来定义，但是不用return返回，而是用yield一次返回一个结果，在每个结果之间挂起和继续它们的状态，来自动实现迭代协议

```python
def Zrange(n):
    i = 0
    while i < n:
        yield i
        i += 1

# test example
zrange = Zrange(3)
print([i for i in zrange])
```
* 生成器对象可以通过for语句进行迭代访问
* 当调用生成器函数的时候，函数只是返回了一个生成器对象，并没有 执行
* 当next()方法第一次被调用的时候，生成器函数才开始执行，执行到yield语句处停止
* next()方法的返回值就是yield语句处的参数（yielded value）
* 当继续调用next()方法的时候，函数将接着上一次停止的yield语句处继续执行，并到下一个yield处停止；如果后面没有yield就抛出StopIteration异常