
# coding: utf-8

# # Tutorial using image from matplotlib

# In[3]:

get_ipython().magic(u'matplotlib inline')


# In[4]:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img=mpimg.imread('assets/stinkbug.png')

print(img)


# In[5]:

imgplot = plt.imshow(img)


# In[6]:

lum_img = img[:,:,0]

plt.imshow(lum_img)


# In[7]:

plt.imshow(lum_img, cmap="hot")


# In[8]:

imgplot = plt.imshow(lum_img, cmap='nipy_spectral')


# In[9]:

imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')


# In[10]:

imgplot = plt.imshow(lum_img)
plt.colorbar()


# In[11]:

plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')


# In[12]:

imgplot = plt.imshow(lum_img, clim=(0.0, 0.7))


# In[14]:

from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('assets/stinkbug.png')
img.thumbnail((64, 64), Image.ANTIALIAS) # resizes image in-place
imgplot = plt.imshow(img)


# In[15]:

imgplot = plt.imshow(img, interpolation="nearest")


# In[16]:

imgplot = plt.imshow(img, interpolation="bicubic")

