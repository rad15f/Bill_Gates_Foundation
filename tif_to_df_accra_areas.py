#%%
# Libraries Importation
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show
from rasterio.merge import merge
import matplotlib.pyplot as plt
import random
from toolbox import covariate_features as covf, contextual_features as ctx

seed= 42
np.random.seed(seed)
random.seed(seed)

#%%
#=======================ACCRA CITY=======================================

abspath_curr="E:/Mapping Deprived Areas/accra/100m/"
# Get Lagos's path then populate 53 covariate features and their labels (mask)
spfea= sorted(glob(abspath_curr+'ctx_100m'+os.sep+'*.tif'))
mask= sorted(glob(abspath_curr+'mask_100m'+os.sep+'*.tif'))
print(f'Length of shapes vs labels: {len(mask),len(spfea)}')

# covariate features names
contextual_features= ctx()

#%%
# Combining feature & label files together
Tiles = []
for item in zip(spfea, mask):
    # print(item)
    Tiles.append(item)
# Shuffle the samples
random.shuffle(Tiles)

# Seperate features and labels
RasterTiles = [] #features
MaskTiles = [] #labels
for raw_image, label in Tiles:
    RasterTiles.append(raw_image)
    MaskTiles.append(label)


#Create train validation & test lists
# X_test_list= [f'E:/Mapping Deprived Areas/accra/100m/ctx_100m\\acc_area{i}.tif' for i in [1,3,10,13]]
# RasterTiles =np.setdiff1d(RasterTiles,X_test_list).tolist() #Removing items existing in X_test_list from RasterTiles
# X_train_list = RasterTiles[:] #split train
X_test_list= [f'E:/Mapping Deprived Areas/accra/100m/ctx_100m\\acc_area{i}.tif' for i in range(0,14)]
RasterTiles =np.setdiff1d(RasterTiles,X_test_list).tolist() #Removing items existing in X_test_list from RasterTiles
X_train_list = RasterTiles[:] #split train

#%%

#Replicate the above for labels
# y_test_list= [f'E:/Mapping Deprived Areas/accra/100m/mask_100m\\acc_area{i}.tif' for i in [1,3,10,13]]
# MaskTiles =np.setdiff1d(MaskTiles,y_test_list).tolist()
# y_train_list = MaskTiles[:]
y_test_list= [f'E:/Mapping Deprived Areas/accra/100m/mask_100m\\acc_area{i}.tif' for i in range(0,14)]
MaskTiles =np.setdiff1d(MaskTiles,y_test_list).tolist()
y_train_list = MaskTiles[:]

#%%

# Preparing training
# raster_to_mosiac = []
# # Read the tif files with rasterio then parse into list
# for p in X_train_list:
#     raster = rasterio.open(p)
#     raster_to_mosiac.append(raster)
#
# # Get combined numpy array from the populated list
# Xtrain, out_transform = merge(raster_to_mosiac)
# print(Xtrain.shape)

# Replication for train, validation, test files on features and labels
# raster_to_mosiac = []
#
# for p in y_train_list:
#     raster = rasterio.open(p)
#     raster_to_mosiac.append(raster)
#
# ytrain, out_transform = merge(raster_to_mosiac)
# print(ytrain.shape)

# Preparing test data

raster_to_mosiac = []

for p in X_test_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)

Xtest, out_transform = merge(raster_to_mosiac)
print(Xtest.shape)

raster_to_mosiac = []

for p in y_test_list:
    raster = rasterio.open(p)
    raster_to_mosiac.append(raster)

ytest, out_transform = merge(raster_to_mosiac)
print(ytest.shape)

#%%

# Dataframe Setup

#Training Data
# X_train = Xtrain[:, Xtrain[0,...]!=-9999] #cleaning the numpy array from invalid inputs (-9999)
# y_train = ytrain[:, ytrain[0,...]!=-9999] #cleaning the numpy array from invalid inputs (-9999)
# y_train =y_train.astype(int) #change the label to integer
# X_train = np.transpose(X_train) #flip the array
# y_train = np.transpose(y_train) #flip the array

#Testing Data
X_test = Xtest[:, Xtest[0,...]!=-9999]
y_test = ytest[:, ytest[0,...]!=-9999]
y_test = y_test.astype(int)
X_test = np.transpose(X_test)
y_test = np.transpose(y_test)

# To DataFrames- Train, Validation & Test
# df_train = pd.DataFrame(X_train,columns=contextual_features)  # covariate features
# df_train['labels'] = y_train
# df_train['type']= 'train'

#%%
df_test = pd.DataFrame(X_test,columns=contextual_features)  # convariate features
df_test['labels'] = y_test
# df_test['type'] = 'test'

# lagos_df = pd.concat([df_train,df_test]) #Combine train, validation & test
lagos_df= df_test.copy(deep=True)
# Export to parquet
lagos_df.to_parquet('accra_contextual_areas_df.parquet.gzip',compression='gzip')