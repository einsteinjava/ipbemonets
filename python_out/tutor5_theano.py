
# coding: utf-8

# # Tutorial using Theano

# ## Basic operations

# In[1]:

import numpy as np
import theano.tensor as T
from theano import function, pp

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x,y], z)

print(pp(z))


# In[2]:

print(f(2, 3))


# In[3]:

print(f(16.3, 12.1))


# In[4]:

# assert
np.allclose(f(16.3, 12.1), 28.4)


# In[5]:

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x,y], z)

i = f([[1,2], [3,4]], [[10,20], [30,40]])
print(i)
type(i)


# In[6]:

j = f(np.array([[1,2], [3,4]]), np.array([[10,20], [30,40]]))
print(j)
type(j)


# ## Theano Function

# In[7]:

import theano
from theano import shared, function
from theano import tensor as T

x = T.scalar()
y = T.scalar()

f1 = function(inputs=[x, y], outputs=(x + y))
f2 = function([x, y], x + y)

print(f1(1., 2.))
print(f2(1., 2.))


# ## Logistic function

# In[8]:

import theano
import theano.tensor as T

x = T.dmatrix('x')

s1 = 1 / (1 + T.exp(-x))

f_logistic1 = theano.function([x], s1)
print('1:')
print(f_logistic1([[0,1],[-1,-2]]))

s2 = (1 + T.tanh(x /2)) / 2

f_logistic2 = theano.function([x], s2)
print('2:')
print(f_logistic2([[0,1], [-1,-2]]))


# ## multicalculation

# In[9]:

import numpy as np
from pprint import pprint

a, b = T.matrices('a','b')
diff = a - b
abs_diff = abs(diff)
sqr_diff = diff ** 2

f_diff = theano.function([a, b], [diff, abs_diff, sqr_diff])

rst = f_diff([[1, 1], [1, 1]], [[0, 1], [2, 3]])

pprint(rst)
print(np.shape(rst))


# ## Using Shared Variables

# ### Function using updates

# In[10]:

get_ipython().magic(u'reset -f')

import theano.tensor as T
from theano import shared, function
s = shared(0)
i = T.iscalar('i')
acc = function(inputs=[i], outputs=s, updates=[(s, s+i)])

print('0:')
print(s.get_value())

print('1:')
print(acc(1))
print(s.get_value())

print('2:')
print(acc(300))
print(s.get_value())

print('3:')
s.set_value(-1)
print(s.get_value())

print('4:')
print(acc(1))
print(s.get_value())


# ### Function using updates function

# In[11]:

get_ipython().magic(u'reset -f')

import theano
from theano import shared, function
from theano import tensor as T

s = shared(0)
i = T.iscalar('i')
upd = [(s, s+i)]

f = function([i], s, updates=upd)

print(s.get_value())
for n in range(5):
    f(10)
    print(s.get_value())


# ### Function using givens

# In[12]:

get_ipython().magic(u'reset -f')

import theano.tensor as T
from theano import shared, function

s0 = shared(0)
s1 = shared(0)
j = T.scalar('j', s1.dtype)  # s1 & j have same data type (givens)
i = T.scalar('i')
b = T.scalar('s0', s0.dtype) # s0 & b have save data eype (given) 

fn = s1 * i + s0

f = function(inputs=[i, j, b], outputs=fn, givens=[(s1, j), (s0, b)])
print(f(2, 2, 100))
print(f(2, 4, 200))

for n in range(1,6):
    print(f(2, n, n*100))


# ## Training simple

# In[13]:

get_ipython().magic(u'reset -f')

import theano as th
import numpy as np
from theano import tensor as T

x = T.fvector('x')   # x input
t = T.fscalar('t')   # y target
W = th.shared(np.asarray([0.2, 0.2, 0.2]), 'W')
y = (x * W).sum()    # y function

cost = T.sqr(t - y)
grad = T.grad(cost=cost, wrt=[W])

print('type_grad:', type(grad))
print('len_grad:' , len(grad))

W_upd = W - (0.1 * grad[0])
upd = [(W, W_upd)]

f = th.function(inputs=[x, t], outputs=y, updates=upd)
xin = [1.0, 1.0, 1.0]
tout = 100.0
step = 10

# Train for y = sum(x * W) =~ 100
# with initial x = [1.0 1.0 1.0]
# and initial W = [0.2 0.2 0.2]
# and target out = 100.0
print(xin)
print(W.get_value())
print(tout)
print(' ')

for i in range(step):
    print(f(xin, tout))
    print(W.get_value())
    print(' ')


# ## Useful Operations

# In[14]:

get_ipython().magic(u'reset -f')

import theano as th
import numpy as np
from theano import tensor as T

a = th.shared(np.asarray([[1.0,2.0], [3.0, 4.0]]), 'a')
print(a)
print(a.eval())


# In[15]:

# element-wise operation: + - * /
c = ((a + a) / 4.0)
print(c)
print(c.eval())


# In[16]:

# Dot product
d = T.dot(a, a)
print(d)
print(d.eval())


# In[17]:

# Activation functions
s1 = T.nnet.sigmoid(a)
print(s1)
print(s1.eval())

s2 = T.tanh(a)
print(s2)
print(s2.eval())


# In[18]:

# Softmax (row-wise)
s3 = T.nnet.softmax(a)
print(s3)
print(s3.eval())


# In[19]:

# Sum
print(a.eval())

sum1 = a.sum() 
print(sum1.eval())
sum2 = a.sum(axis=1)
print(sum2.eval())
sum3 = a.sum(axis=0)
print(sum3.eval())


# In[20]:

# Max
max1 = a.max()
print(max1.eval())
max2 = a.max(axis=1)
print(max2.eval())


# In[21]:

# Argmax
amax1 = T.argmax(a)
print(amax1.eval())
amax2 = T.argmax(a, axis=1)
print(amax2.eval())


# In[22]:

# Resahpe
print(a.eval())
shp1 = a.reshape((1,4))
print(shp1.eval())
shp2 = a.reshape((-1,))
print(shp2.eval())


# In[23]:

# Zeros like, ones like
zero = T.zeros_like(a)
print(zero.eval())


# In[24]:

# Reorder the tensor dimensions
print(a.eval())
c = a.dimshuffle((1,0))
print(c.eval()) 
c = a.dimshuffle(('x',0,1))
print(c.eval())


# In[25]:

# Indexing
print(a.eval())
b = [1,1,0]
c = a[b]
print(c.eval())

