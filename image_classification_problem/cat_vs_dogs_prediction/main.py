import os
import random
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from matplotlib import pyplot as plt


import seaborn as sns
sns.set()

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, mean_squared_error

from keras.models import Sequential
from keras.layers import Dense

from utils import test_test_split

# _, _, cat_images = next(os.walk('data/PetImages/Cat'))

## Prepare a 3x3 plot (total of 9 images)
# fix, ax = plt.subplots(3, 3, figsize=(20, 10))

## Randomly select and plot an image
# for idx, img in enumerate(random.sample(cat_images, 9)):
#     img_read = plt.imread('data/PetImages/Cat/' + img)
#     ax[int(idx / 3), idx % 3].imshow(img_read)
#     ax[int(idx / 3), idx % 3].axis('off')
#     ax[int(idx / 3), idx % 3].set_title('Cat/' + img)
# plt.show()

## Get list of file names (Dog images)
# _, _, dog_images = next(os.walk('data/PetImages/Dog'))

# # Prepare a 3x3 plot (total of 9 images)
# fix, ax = plt.subplots(3, 3, figsize=(20, 10))

# # Randomly select and plot an image
# for idx, img in enumerate(random.sample(dog_images, 9)):
#     img_read = plt.imread('data/PetImages/Dog/' + img)
#     ax[int(idx / 3), idx % 3].imshow(img_read)
#     ax[int(idx / 3), idx % 3].axis('off')
#     ax[int(idx / 3), idx % 3].set_title('Dog/' + img)
# plt.show()

src_folder = 'data/PetImages/'
train_test_split(src_folder)
