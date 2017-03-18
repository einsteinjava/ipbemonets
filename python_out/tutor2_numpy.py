
# coding: utf-8

# # Tutorial using Numpy
# Library for scientific computing which provides high-performance multidinensional array object (Python version of MATLAB).

# ## Arrays

# In[1]:

import numpy as np

a = np.array([1, 2, 3,])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 5
print(a)


# In[2]:

b = np.array([[1,2,3],[4,5,6]])
print(type(b))
print(b.shape)
print(b[0,0], b[0,1], b[1,0])
print(b)


# ### Fucntion that create array

# In[3]:

import numpy as np

a = np.zeros((2,2))
print(a)

b = np.ones((2,3))
print(b)

c = np.full((2,2), 7)
print(c)

d = np.eye(3)
print(d)

e = np.random.random((3,3))
print(e)


# ### Array indexing

# In[4]:

import numpy as np

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)


# ### Slicing

# In[5]:

import numpy as np

b = a[:2, 1:3]
print(b)

print a[0, 1]
b[0, 0] = 77
print(a[0, 1])
print(a)

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)


# In[6]:

row_r1 = a[1,:]
row_r2 = a[1:2, :]
print(row_r1)
print(row_r2)
print(row_r1.shape)
print(row_r2.shape)


# In[7]:

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1)
print(col_r2)
print(col_r1.shape)
print(col_r2.shape)


# In[8]:

a = np.array([[1,2], [3,4], [5,6]])
print(a)

b = a[[0,1,2], [0,1,0]]
print(b)

c = np.array([a[0,0],a[1,1],a[2,0]])
print(c)

d = a[[0,0], [1,1]]
print(d)


# In[9]:

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
print(a)

b = np.array([0,2,0,1])
print(b)

print(a[np.arange(4), b])

a[np.arange(4), b] += 10
print(a)


# ### Boolean array indexing

# In[10]:

import numpy as np

a = np.array([[1,2], [3,4], [5,6]])
bool_idx = (a > 2)
print(bool_idx)
print(a[bool_idx])
print(a[a>2])


# ### Datatypes

# In[11]:

import numpy as np

x = np.array([1,2])
print(x.dtype)
print(type(x))
print(x.dtype)


# In[12]:

x = np.array([1.0, 2.0])
print(x.dtype)

x = np.array([1,2], dtype=np.int64)
print(x.dtype)


# ### Array math

# In[13]:

import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print('1:')
print(x + y)
print(np.add(x, y))

print('2:')
print(x - y)
print(np.subtract(x, y))

print('3:')
print(x * y)
print(np.multiply(x, y))

print('4:')
print(x / y)
print(np.divide(x, y))

print('5:')
print(np.sqrt(x))


# ### Vectors products

# In[14]:

import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11,12])

print('# inner product of vectors:')
print(v.dot(w))
print(np.dot(v, w))

print('# matrix/vector product:')
print(x.dot(v))
print(np.dot(x, v))

print('# matrix/matrix product:')
print(x.dot(y))
print(np.dot(x, y))


# ### Computation

# In[15]:

import numpy as np

x = np.array([[1,2],[3,4]])

print('# sum of all elements:')
print(np.sum(x))

print('# sum of each column:')
print(np.sum(x, axis=0))

print('# sum of each row:')
print(np.sum(x, axis=1))  

print('# Tranpose:')
print(x.T)


# ### Broadcasting

# In[16]:

import numpy as np

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)
print (y)

print('# add vector v to each row of matriix x')
for i in range(4):
    y[i, :] = x[i, :] + v
print(y)


# In[17]:

import numpy as np

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))
print(vv)

y = x + vv
print(y)


# In[18]:

import numpy as np

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(x)

v = np.array([1, 0, 1])
print(v)

y = x + v
print(y)


# In[19]:

# T : Transpose
import numpy as np

v = np.array([1,2,3])
w = np.array([4,5])
print(np.reshape(v, (3,1)) * w)

x = np.array([[1,2,3], [4,5,6]])
print(x + v)

print((x.T + w).T)

print(x + np.reshape(w, (2,1)))
print(x * 2)


# ### Image operations

# In[20]:

from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
print(img.dtype, img.shape)
plt.imshow(img)

img_tinted = img * [1, 0.95, 0.9]
plt.imshow(img_tinted)
img_tinted = imresize(img_tinted, (300, 300))
plt.imshow(img_tinted)
imsave('assets/cat_tinted.jpg', img_tinted)


# ### Distance between points

# In[21]:

import numpy as np
from scipy.spatial.distance import pdist, squareform
x = np.array([[0,1], [1,0], [2,0]])
print x


d = squareform(pdist(x, 'euclidean'))
print d


# ## Plotting

# In[22]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
print(x)
y = np.sin(x)
print(y)

plt.plot(x, y)


# In[ ]:




# In[23]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# ### Subplot

# In[24]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')
plt.show()


# ## Images

# In[26]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

img = imread('assets/cat.jpg')
img_tinted = imread('assets/cat_tinted.jpg')

img_tinted = img * [1., 0.95, 0.9]

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(img_tinted)
plt.show()

