
# coding: utf-8

# # Basic Data Types

# ### Integer

# In[1]:

x = 2
type(x)


# In[2]:

x


# In[3]:

print(x)


# In[4]:

print(x + 1)


# In[5]:

print(x - 1)



# In[6]:

print(x * 2)


# In[7]:

print(x ** 2)


# In[8]:

x += 1
print(x)


# In[9]:

x *=2
print(x)


# ### Float

# In[10]:

y = 2.5
type(y)


# In[11]:

print(y, y + 1, y * 2, y ** 2)


# ### Long
# #### (only on python 2, will not work on python 3)

# In[12]:

l = 1L
type(l)


# ### Complex Number

# In[13]:

j = 10 + 10j
type(j)


# In[14]:

print(j)


# ### Boolean

# In[15]:

t = True
f = False
type(t), type(f)


# ### Boolean operation

# In[16]:

print(t and f)
print(t or f)
print(not t)
print(t != f)


# ### String operation

# In[17]:

hello = 'hello'
world = "world"
print(hello)
print(len(hello))
hw = hello + ' ' + world
print(hw)
hw12 = '%s %s %d' % (hello, world, 12)
print(hw12)
print('The result is: %s %s %d! ' % (hello, world, 12))


# ### String methods

# In[18]:

s = "hello"
print(s.capitalize())
print(s.upper())
print(s.rjust(7))
print(s.center(7))
print(s.replace('l', '(ell)'))
print('  world'.strip())


# # Containers

# ## Lists

# In[19]:

xs = [3, 1, 2]
print(xs, xs[2])
print(xs[-1])
xs[2] = 'foo'
print(xs)
xs.append('bar')
print(xs)
x = xs.pop()
print(x, xs)


# ### Slicing

# In[46]:

nums = range(5)

print(type(nums))
print(nums)
print(nums[2:4])
print(nums[2:])
print(nums[:2])
print(nums[:])
print(nums[:-1])
nums[2:4] = [8, 9]
print(nums)


# ### Loops

# In[21]:

animals = ['cat', 'dog', 'monkey']
print(animals)
for animal in animals:
    print(animal)


# ### Index

# In[22]:

for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))


# ### List value modification

# In[23]:

nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)


# ### Simpled it with list comprehension

# In[24]:

nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)


# ### List comprehension contain conditions

# In[25]:

nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)


# ## Dictionaries/Maps

# In[47]:

d = {'cat': 'cute', 'dog': 'furry'}

print(type(d))
print(d['cat'])
print('cat' in d)
d['fish'] = 'wet'
print(d['fish'])
#print(d['monkey'])  # KeyError
print(d.get('monkey', 'N/A'))
print(d.get('fish', 'N/A'))
del d['fish']
print(d.get('fish', 'N/A'))


# ### Loop over dictionary

# In[28]:

d = {'person': 2, 'cat': 4, 'spyder': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))


# ### Simpled way

# In[29]:

d = {'person': 2, 'cat': 4, 'spyder': 8}
for animal, legs in d.iteritems():
    print('A %s has %d legs' % (animal, legs))


# ### Dictionary comprehension

# In[30]:

nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)


# ## Sets  (Unordered collection of distinct elements)

# In[48]:

animals = {'cat', 'dog'}
print(type(animals))
print('cat' in animals)    
print('fish' in animals)
animals.add('fish')
print(len(animals))
animals.add('cat')
print(len(animals))
animals.remove('cat')
print(len(animals))
print(animals)


# In[32]:

animals ={'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))


# In[33]:

from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)


# ### Tuples

# In[55]:

d = {(x, x + 1): x for x in range(10)}
print(type(d))
print(d)

t = (1, 2)
print(type(t))
print(d[t])
print(d[(2, 3)])


# ### Enumerate

# In[56]:

alist = ['a1', 'a2', 'a3']
for i, a in enumerate(alist):
    print(i, a)


# ### Zip- Iterate over two lists in parallel

# In[57]:

alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']
for a, b in zip(alist, blist):
    print(a, b)


# ### Enumerate with zip

# In[58]:

alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']
for i, (a, b) in enumerate(zip(alist, blist)):
    print(i, a, b)


# 
# # Function / def keyword

# In[59]:

def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print sign(x)


def hello(name, loud=False):
    if loud:
        print 'HELLO, %s!' % name.upper()
    else:
        print 'Hello, %s' % name
        
hello('Bob')
hello('Fred', loud=True)


# 
# # Class

# In[63]:

class Greeter(object):
    
    #consturtor
    def __init__(self, name='None'):
        self.name = name
        
    # method   
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name


# ### Test the class

# In[64]:

g = Greeter('Fred')
g.greet()
g.greet(loud=True)


# In[ ]:



