# Packages importing
#%%
import numpy as np
import pandas as pd
import pickle
import tkinter
import matplotlib
# matplotlib.use('TkAgg')  # !IMPORTANT
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.plot import show
import seaborn as sns
import scipy, os
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from toolbox import get_train_val_ps, important_features, contextual_features, tif_to_df, cm,covariate_features, important_55, plot_pred_area
random_seed=123
target_names = ['Not Deprived', 'Deprived']
rgbn=['r','g','b','n']
covariate_features=covariate_features()
contextual_features= contextual_features()

#%%
with open('model_lr_55_accra.pickle', 'rb') as f:
    lr= pickle.load(f)

best_55= important_55()
#%%
# FOR 1 MODEL AT A TIME (55 features)
mms= MinMaxScaler()
for i in [0,2,3,4,5,8, 9, 10, 11, 12, 13]:
    area0= [f"E:/Mapping Deprived Areas/accra/100m/rgb_100m/acc_area{i}.tif"]
    area0_mask= [f"E:/Mapping Deprived Areas/accra/100m/mask_100m/acc_area{i}.tif"]
    area0_df= tif_to_df(area0,area0_mask,rgbn)
    area0_df.drop(['labels'],axis=1,inplace=True)

    area00 = [f"E:/Mapping Deprived Areas/accra/100m/ctx_100m/acc_area{i}.tif"]
    area00_mask = [f"E:/Mapping Deprived Areas/accra/100m/mask_100m/acc_area{i}.tif"]
    area00_df = tif_to_df(area00, area00_mask, contextual_features)

    area000 = [f"E:/Mapping Deprived Areas/accra/100m/cov_100m/acc_area{i}.tif"]
    area000_mask = [f"E:/Mapping Deprived Areas/accra/100m/mask_100m/acc_area{i}.tif"]
    area000_df = tif_to_df(area000, area000_mask, covariate_features)
    labels_copy= area000_df['labels'].values.copy()
    area000_df.drop(['labels'],axis=1,inplace=True)

    area_df= pd.concat([area0_df,area000_df,area00_df],axis=1)
    area_df = area_df[~(area_df == -9999.000000)]
    values = {i: area_df[i].mean() for i in area_df.columns[:-2]}
    area_df.loc[:, area_df.columns[:-1]] = area_df.loc[:, area_df.columns[:-1]].fillna(value=values)
    # area0_pred = pd.Series(lr136.predict_proba(mms.transform(area0_df.drop('labels',
    #              axis=1).values))[:,1]).apply(lambda x:1 if x>0.41 else 0).values #41)
    # area0_pred = lr136.predict(mms.transform(area_df.drop('labels', axis=1).values))
    area0_pred= lr.predict(mms.fit_transform(area_df.loc[:,area_df.columns.isin(best_55)].values))
    # area0_pred = pd.Series(lr136.predict_proba(mms.transform(area0_df.loc[:,
    #                         area0_df.columns.isin(contextual_features)].drop('labels',
    #                         axis=1).values))[:,1]).apply(lambda x:0 if x>0.5 else 1).values #41)

    #plot original area
    img= rasterio.open(area0_mask[0]).read(1)
    show(img)
    #Plot predicted area
    plot_pred_area(img.shape[0],img.shape[1],area0_pred)
    #Plot confusion matrix
    cm(area_df.labels.values, area0_pred)
    print(classification_report(area_df.labels.values, area0_pred, target_names=target_names))

#%%
    # # Save to GeoTiff
    # import datetime
    #
    # T = datetime.datetime.now()
    # time = T.strftime("%y%m%d")
    #
    # filename = 'accra_lr55'
    # out_file = f"E:/Mapping Deprived Areas/accra/100m/{i}_{time}.tif"
    # fp =area0_mask[0]
    # with rasterio.open(fp, mode="r") as src:
    #     out_profile = src.profile.copy()
    #     out_profile.update(count=1,
    #                        nodata=-9999,
    #                        dtype='float32',
    #                        width=src.width,
    #                        height=src.height,
    #                        crs=src.crs)
    #
    # # open in 'write' mode, unpack profile info to dst
    # with rasterio.open(out_file,
    #                    'w', **out_profile) as dst:
    #     # dst.write_band(1, c.labels.values.reshape(544,805))
    #     dst.write_band(1, area0_pred.reshape(21, 20))