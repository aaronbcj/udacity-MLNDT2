# Capstone project


## APTOS 2019 Blindness Detection
### Detect diabetic retinopathy to stop blindness before it's too late

###  Aaron Caroltin
Jupyter notebook from https://www.kaggle.com/aaronbcj/aptos-v4?scriptVersionId=18875527

Overview from https://www.kaggle.com/c/aptos2019-blindness-detection/overview

Download data from https://www.kaggle.com/c/aptos2019-blindness-detection/data

### Run notebook
You can join the kaggle APTOS 2019 competition and fork my public kernel above. Upload weight file to workspace and run the cells.
Alternatively if you want to work in local, you can download data folder from kaggle.

### Following DenseNet121 model weight file
DenseNet-BC-121-32-no-top.h5

### Output files in results folder
history.json
model.h5
submission.csv


### Following Imports were used
import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm
from keras.applications.densenet import DenseNet121
import seaborn as sns
sns.set()


from IPython.display import display