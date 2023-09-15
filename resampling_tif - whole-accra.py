#%%
from osgeo import gdal
import os
from glob import glob
from subprocess import Popen
import fiona
import rasterio
import numpy as np
import pandas as pd
from rasterio.merge import merge

#%%
fourier= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/fourier/"+'*.tif'))
gabor= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/gabor/"+'*.tif'))
hog= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/hog/"+'*.tif'))
lac= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/lac/"+'*.tif'))
lbpm= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/lbpm/"+'*.tif'))
lsr= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/lsr/"+'*.tif'))
mean= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/mean/"+'*.tif'))
ndvi= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/ndvi/"+'*.tif'))
orb= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/orb/"+'*.tif'))
pantex= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/pantex/"+'*.tif'))
sfs= sorted(glob(f"E:/Mapping Deprived Areas/accra/accra_contextual_10m/sfs/"+'*.tif'))

#%%
spfea= fourier+gabor+hog+lac+lbpm+lsr+mean+ndvi+orb+pantex+sfs
#%%
# spfea10= sorted(glob("C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos/lagos_covariate/covariate_100m/"+'*.tif'))
#%%
# Convert tifs to 1 vrt for resampling contextual
output1 = f"E:/Mapping Deprived Areas/accra/accra_whole/accra_contextual_100m"
vrt_options = gdal.BuildVRTOptions(separate=True) #outputSRS=None or separate=True
vrt =  gdal.BuildVRT(f'{output1}/acc_contextual.vrt', spfea, options=vrt_options)
vrt = None

#%%
# Convert tifs to 1 vrt for resampling rgbn
rgbn= f"E:/Mapping Deprived Areas/accra/accra_sentinel2_10m/acc_bgrn.tif"
output1 = f"E:/Mapping Deprived Areas/accra/accra_whole/accra_rgbn_100m"
vrt_options = gdal.BuildVRTOptions(outputSRS=None) #outputSRS=None or separate=True
vrt =  gdal.BuildVRT(f'{output1}/acc_rgbn.vrt', rgbn, options=vrt_options)
vrt = None

#%%
# Convert tif to vrt Covariate
covt= f"E:/Mapping Deprived Areas/accra/accra_covariate_feature_53/acc_covariate_compilation_53bands.tif"
output2 = f"E:/Mapping Deprived Areas/accra/accra_whole/accra_covariate_100m"
vrt_options = gdal.BuildVRTOptions(outputSRS=None) #outputSRS=None or separate=True
vrt =  gdal.BuildVRT(f'{output2}/acc_cov.vrt', covt, options=vrt_options)
vrt = None
#%%
# Convert tifs(mask) to 1 vrt for resampling
# vrt =  gdal.BuildVRT(f'{output2}/lag_covariate_mask.vrt', mask10, options=vrt_options)
# vrt = None
#%%
# gdal_warp
# resampling vrt contextual
# fp= f"E:/Mapping Deprived Areas/accra/accra_whole/accra_contextual_100m/acc_contextual.vrt"
fp= f"C:/Users/oseme/Desktop/accra/acc_contextual.vrt"
outfile= f"C:/Users/oseme/Desktop/accra/acc_contextual_resam.vrt"
# outfile= f"E:/Mapping Deprived Areas/accra/accra_whole/accra_contextual_100m/acc_contextual_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.0008333333299999819968 0.0008333333299999819968 -r bilinear  -of GTiff {fp} {outfile}'
Popen(command, shell=True)
# can use 0.00083 0.00083 or 0.0008333333299999819968 -0.0009722222183333369952

#%%
# gdal_warp
# resampling vrt covariate
fp= f"C:/Users/oseme/Desktop/accra/acc_cov.vrt"
outfile= f"C:/Users/oseme/Desktop/accra/acc_covariate_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.0008333333299999819968 0.0008333333299999819968 -r bilinear  -of GTiff {fp} {outfile}'
Popen(command, shell=True)
# can use 0.00083 0.00083 or 0.0008333333299999819968 -0.0009722222183333369952
#%%
# gdal_warp
# resampling vrt rgbn
# fp= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_rgbn_100m/lag_rgbn.vrt"
# outfile= f"C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_whole/lagos_rgbn_100m/lag_rgbn_resam.vrt"
fp= f"C:/Users/oseme/Desktop/accra/acc_rgbn.vrt"
outfile= f"C:/Users/oseme/Desktop/accra/acc_rgbn_resam.vrt"
command = f'gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -dstnodata -9999.0 -tr 0.0008333333299999819968 0.0008333333299999819968 -r bilinear  -of GTiff {fp} {outfile}'
Popen(command, shell=True)
# can use 0.00083 0.00083 or 0.0008333333299999819968 -0.0009722222183333369952
#%%
# Try vrt to Tif
ctx= f"C:/Users/oseme/Desktop/accra/acc_contextual_resam.vrt"
rgb= f"C:/Users/oseme/Desktop/accra/acc_rgbn_resam.vrt"
cov= f"C:/Users/oseme/Desktop/accra/acc_covariate_resam.vrt"
com= f"gdal_translate {ctx} acc_contextual_100m.tif"
Popen(com, shell=True)
com2= f"gdal_translate {rgb} acc_rgbn_100m.tif"
Popen(com2, shell=True)
com3= f"gdal_translate {cov} acc_covariate_100m.tif"
Popen(com3, shell=True)
#%%

#%%
# Clip using lagos polygon shape file
# Clip raster(rgbn) using polygon shape file
polygons = sorted(glob('C:/Users/oseme/Desktop/accra/region_of_interest\\*.shp'))
# polygons = f'C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_GSHL/lagos_GSHL/lagos_GHLS.shp'
VRT = f"C:/Users/oseme/Desktop/accra\\acc_rgbn_resam.vrt"
outfile = "C:/Users/oseme/Desktop/accra\\clipped"
# print(polygons)

for polygon in polygons:
    # print(polygon)
    feat = fiona.open(polygon, 'r')
    # add output file name
    head, tail = os.path.split(polygon)
    name=tail[:-4]
    # print(name)
    # command = f'gdalwarp -dstnodata -9999 -ts 6 6 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
    command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'

    Popen(command, shell=True)
#%%
# Clip using lagos polygon shape file
# Clip raster(covariate) using polygon shape file
polygons = sorted(glob('C:/Users/oseme/Desktop/accra/region_of_interest\\*.shp'))
# polygons = f'C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_GSHL/lagos_GSHL/lagos_GHLS.shp'
VRT = f"C:/Users/oseme/Desktop/accra\\acc_covariate_resam.vrt"
outfile = "C:/Users/oseme/Desktop/accra\\clipped"
# print(polygons)

for polygon in polygons:
    # print(polygon)
    feat = fiona.open(polygon, 'r')
    # add output file name
    head, tail = os.path.split(polygon)
    name=tail[:-4]
    # print(name)
    # command = f'gdalwarp -dstnodata -9999 -ts 6 6 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
    command = f'gdalwarp -dstnodata -9999 -ts 841 561 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
    # -ts 841 561 was added to fit the shapefile of Accra in context
    Popen(command, shell=True)

#%%
# Clip using lagos polygon shape file
# Clip raster(contextual) using polygon shape file
polygons = sorted(glob('C:/Users/oseme/Desktop/accra/region_of_interest\\*.shp'))
# polygons = f'C:/Users/oseme/Desktop/Capstone/data_Ryan/lagos_GSHL/lagos_GSHL/lagos_GHLS.shp'
VRT = f"C:/Users/oseme/Desktop/accra\\acc_contextual_resam.vrt"
outfile = "C:/Users/oseme/Desktop/accra\\clipped"
# print(polygons)

for polygon in polygons:
    # print(polygon)
    feat = fiona.open(polygon, 'r')
    # add output file name
    head, tail = os.path.split(polygon)
    name=tail[:-4]
    # print(name)
    # command = f'gdalwarp -dstnodata -9999 -ts 6 6 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'
    command = f'gdalwarp -dstnodata -9999 -cutline {polygon} -crop_to_cutline -of Gtiff {VRT} "{outfile}/{name}.tif"'

    Popen(command, shell=True)

#%%
# Check height and width of created tif files
files = sorted(glob("C:/Users/oseme/Desktop/accra/clipped/"+'*.tif'))
num = 0
for raster in files:
    # print(reference_raster)
    # img = utils_funcs.read_image(raster)
    img= rasterio.open(raster).read()
    width = img.shape[1]
    height = img.shape[2]
    num += 1
    print(num, width, height)