# -*- coding: utf-8 -*-
"""
The standalone code for calculating Artificial Surface Index (ASI)

Created on Fri Feb 11 22:05:10 2022
@author: Yongquan Zhao, University of Connecticut, Storrs.

Reference: Yongquan Zhao, Zhe Zhu. 2022. ASI: An artificial surface Index for Landsat 8 imagery. International Journal of Applied Earth Observation and Geoinformation 107: 102703.
https://doi.org/10.1016/j.jag.2022.102703

"""


import numpy as np
from osgeo import gdal


def read_img(filename):
    dataset=gdal.Open(filename) 

    img_width = dataset.RasterXSize
    img_height = dataset.RasterYSize

    img_geotrans = dataset.GetGeoTransform()
    img_proj = dataset.GetProjection()
    img_data = dataset.ReadAsArray(0,0,img_width,img_height)

    del dataset
    return img_data, img_proj, img_geotrans
    

def write_img(filename,img_proj,img_geotrans,img_data, fill_value):
    if 'int8' in img_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img_data.shape) == 3:
        img_bands, img_height, img_width = img_data.shape
    else:
        img_bands, (img_height, img_width) = 1,img_data.shape

    driver = gdal.GetDriverByName("GTiff")           
    dataset = driver.Create(filename, img_width, img_height, img_bands, datatype)
    dataset.SetGeoTransform(img_geotrans)  
    dataset.SetProjection(img_proj) 

    if img_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img_data)  # write data.
        dataset.GetRasterBand(1).SetNoDataValue(fill_value) #set transparent values
    else:
        for i in range(img_bands):
            dataset.GetRasterBand(i+1).WriteArray(img_data[i])  # write data.
            dataset.GetRasterBand(i+1).SetNoDataValue(fill_value) #set transparent values      
    
    dataset.FlushCache() #saves to disk.
    
    dataset = None
    del dataset


def hist_cut(band, mask, fill_value=-9999, k=3, minmax='std'):
    if minmax == 'std':
        mean = band[mask].mean()
        std = band[mask].std()
        low_val = (mean - k * std)
        high_val = (mean + k * std)
    else:
        low_val, high_val = minmax # use specified value range.
    is_low = band < low_val
    is_high = band > high_val
    mask_invalid_index = is_low | is_high
    band[mask_invalid_index] = fill_value
    return band, ~mask_invalid_index


def minmax_norm(band, mask, fill_value=-9999):
    max_val = band[mask].max()
    min_val = band[mask].min()
    extent = max_val - min_val
    if extent != 0:
        shifted = band - min_val
        scaled = shifted / extent
        band[mask] = scaled[mask]    
    band[~mask] = fill_value
    return band


# Rescale the data value range of Landsat Collection 2 surface reflectance to [0, 1*Scale]
# Reference: https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products?qt-news_science_products=0#qt-news_science_products
def Landsat_C2_Rescale(Img, Scale):
    # Conduct rescaling for the blue, green, red, NIR, SWIR1, SWIR2 bands.
    ScaleImg = (Img[0:6,:,:]*0.0000275 - 0.2) * Scale;
    return ScaleImg


# Artificial Surface Index (ASI) is designed based the surface reflectance imagery of Landsat 8.
def artificial_surface_index(Blue, Green, Red, NIR, SWIR1, SWIR2, Scale, MaskValid_Obs, fillV):
    ##### The calculation chain.

    # Artificial surface Factor (AF).
    AF = (NIR - Blue) / (NIR + Blue)    
    AF, MaskValid_AF = hist_cut(AF, MaskValid_Obs, fillV, 6, [-1, 1])
    MaskValid_AF_U = MaskValid_AF & MaskValid_Obs
    AF_Norm = minmax_norm(AF, MaskValid_AF_U, fillV)

    # Vegetation Suppressing Factor (VSF).
    MSAVI = ( (2*NIR+1*Scale) - np.sqrt((2*NIR+1*Scale)**2 - 8*(NIR-Red)) ) / 2 # Modified Soil Adjusted Vegetation Index (MSAVI).
    MSAVI, MaskValid_MSAVI = hist_cut( MSAVI, MaskValid_Obs, fillV, 6, [-1, 1])
    NDVI = (NIR - Red) / (NIR + Red)
    NDVI, MaskValid_NDVI  = hist_cut( NDVI, MaskValid_Obs, fillV, 6, [-1, 1])
    VSF = 1 - MSAVI*NDVI
    MaskValid_VSF = MaskValid_MSAVI & MaskValid_NDVI & MaskValid_Obs
    VSF_Norm = minmax_norm(VSF, MaskValid_VSF, fillV)

    # Soil Suppressing Factor (SSF).
    # Derive the Modified Bare soil Index (MBI).
    MBI = (SWIR1 - SWIR2 - NIR) / (SWIR1 + SWIR2 + NIR) + 0.5
    MBI, MaskValid_MBI = hist_cut(MBI, MaskValid_Obs, fillV, 6, [-0.5, 1.5])
    # Deriving Enhanced-MBI based on MBI and MNDWI.
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    MNDWI, MaskValid_MNDWI = hist_cut(MNDWI, MaskValid_Obs, fillV, 6, [-1, 1])
    EMBI = ((MBI+0.5) - (MNDWI+1)) / ((MBI+0.5) + (MNDWI+1))
    EMBI, MaskValid_EMBI = hist_cut(EMBI, MaskValid_Obs, fillV, 6, [-1, 1])
    # Derive SSF.
    SSF = (1 - EMBI)
    MaskValid_SSF = MaskValid_MBI & MaskValid_MNDWI & MaskValid_EMBI & MaskValid_Obs
    SSF_Norm = minmax_norm(SSF, MaskValid_SSF, fillV)

    # Modulation Factor (MF).
    MF = (Blue + Green - NIR - SWIR1) / (Blue + Green + NIR + SWIR1)
    MF, MaskValid_MF = hist_cut(MF, MaskValid_Obs, fillV, 6, [-1, 1])
    MaskValid_MF_U = MaskValid_MF & MaskValid_Obs
    MF_Norm = minmax_norm(MF, MaskValid_MF_U, fillV)

    # Derive Artificial Surface Index (ASI).
    ASI = AF_Norm * SSF_Norm * VSF_Norm * MF_Norm
    MaskValid_ASI = MaskValid_AF_U & MaskValid_VSF & MaskValid_SSF & MaskValid_MF_U & MaskValid_Obs    
    ASI[~MaskValid_ASI] = fillV
    
    return ASI



# Read Landsat C2 surface reflectance imagery.
Img_name = 'LC08_L2SP_044034_20201012_02_T1_SR_B234567_SF_Geo.tif' # Specify your input images.
Img, proj, geotrans  = read_img(Img_name) 

##### Transfer the value range of Landsat surface reflectance to [0, 1*Scale].
# Feed the Landsat Collection 2 surface reflectance image to this function, e.g., a numpy array with a image size of 6(band amount)*250(height)*327(width).
# Be careful. Landsat_C2_Rescale is based on the Landsat Collection 2 surface reflectance imagery, 
# which is not necessary for Landsat Collection 1 surface reflectancy imagery.
# For original digital value imagery, Landsat_C2_Rescale is not applicable.
Scale = 10000 
RescaleImg = Landsat_C2_Rescale(Img, Scale)

Blue = RescaleImg[0]
Green = RescaleImg[1]
Red = RescaleImg[2]
NIR = RescaleImg[3]
SWIR1 = RescaleImg[4]
SWIR2 = RescaleImg[5]

# Surface reflectance should be within [0, 1*Scale]
MaskValid_Obs = ((Blue>0) & (Blue<1*Scale) &
             (Green>0) & (Green<1*Scale) &
             (Red>0) & (Red<1*Scale) &
             (NIR>0) & (NIR<1*Scale) &
             (SWIR1>0) & (SWIR1<1*Scale) &
             (SWIR2>0) & (SWIR2<1*Scale)
             )

# The fill value for invalid satellite observations and invalid index values.
fill_value = -9999

# Calculating ASI.
ASI = artificial_surface_index(Blue.astype(np.float32), Green.astype(np.float32), Red.astype(np.float32), 
                               NIR.astype(np.float32), SWIR1.astype(np.float32), SWIR2.astype(np.float32), 
                               Scale, MaskValid_Obs, fill_value)

# Get land mask.
MNDWI = (Green - SWIR1) / (Green + SWIR1)
MNDWI, MaskValid_MNDWI = hist_cut(MNDWI, MaskValid_Obs, fill_value, 6, [-1, 1])
Water_Th = 0; # Water threshold for MNDWI (may need to be adjusted for different study areas).
MaskLand = (MNDWI<Water_Th)

# Exclude water pixels.
ASI[~MaskLand] = fill_value

# Write ASI.
write_img('ASI.tif', # Specifiy your output filename.
          proj, geotrans, ASI, fill_value)
