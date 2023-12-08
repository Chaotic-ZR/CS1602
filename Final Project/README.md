# 函数

## 属性函数

```python
def __init__(self, data=None, dim=None, init_value=0)

def shape(self)

def copy(self)

def __getitem__(self, key)

def __setitem__(self, key, value)

def __len__(self)

def __str__(self)

```

## 计算函数

```python
def dot(self, other)

def Kronecker_product(self, other)

def __pow__(self, n)

def __sub__(self, other)

def __add__(self, other)

def __mul__(self, other)
```



## 变换与自计算函数

```python
def T(self)

def sum(self, axis=None)

def det(self)

def inverse(self)

def rank(self)

```



## 其他函数

```
def I(n)

```

