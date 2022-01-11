# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:42:59 2022

@author: tashl
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:56:43 2021

@author: tashl
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:26:00 2021
@author: tashl
"""

import os
import glob
import numpy as np
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io
from skimage.filters import sobel, threshold_isodata
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square, disk, binary_erosion
from skimage.restoration import  denoise_wavelet

from joblib import Parallel, delayed

# ===================================================================
# Objects
# ===================================================================

class FlocImg(object):
    """
    Image data object with metadata
    
    Parameters:
        - data: the M x N image data.
        - bgval: average pixel value of the background
        - fgval: average pixel value of flocs
        - threshold: value separating flocs from background
    
    """
    
    def __init__(self, data, bgval, fgval, threshold):
        self.data = data
        self.bgval = bgval
        self.fgval = fgval
        self.threshold = threshold


# ===================================================================
class ImgLoader(object):
    
    def __init__(self, flist):
        """Instantiate object for loading images
        
        Arguments:
          - flist: Sorted list of files. Order is important if using
            image differencing.
            
        """
            
        self.flist = flist
        self.nimg = len(self.flist)
        
    def difference(self, index):
        """ Create FlocImg object using differencing algorithm """
        
        # load denoised images
        target_img = denoise_wavelet(io.imread(flist[index]))
        previous = denoise_wavelet(io.imread(flist[index-1]))

        # compute image difference
        img = target_img - previous
        img[img > 0] = 0
        
        # calculate meta parameters
        threshold = threshold_isodata(target_img)
        bgval = np.nanmean(target_img[target_img > threshold])
        fgval = np.nanmean(target_img[target_img < threshold])

        # rescale image
        data = bgval+img
        data[data<0] = 0
        return FlocImg(data, bgval, fgval, threshold)


    def single(self, index):
        """ Create FlocImg object using a single denoised image """
        data = denoise_wavelet(io.imread(flist[imgix]))
        # TODO: implement rolling ball filter for removing background
        
        # calculate meta parameters
        threshold = threshold_isodata(data)
        bgval = np.nanmean(data[data > threshold])
        fgval = np.nanmean(data[data < threshold])
        
        return FlocImg(data, bgval, fgval, threshold)


# ===================================================================
class FlocID(object):
    
    def __init__(self, flocimg_object, resolution=0.95, extraparams=[]):
        """ Performs floc identification on target image 
        Arguments:
            
        img_object: an instance of 
        
        """
        
        self.resolution = resolution
        imgdata = flocimg_object.data
        bgval = flocimg_object.bgval
        fgval = flocimg_object.fgval
        threshold = flocimg_object.threshold
        
        
        # compute threshold using isodata method and segment image
        self.segmentation = closing(imgdata<threshold, disk(1))
        
        # remove artifacts connected to image border
        self.cleared = clear_border(self.segmentation, buffer_size=1)
        
        # label image regions
        self.label_image, self.nflocs = label(self.cleared, return_num=True)
        
                
        # make dictionary of floc info
        self.flocs = regionprops_table(self.label_image,
                                  properties=['area']+extraparams)
        
        # calculate edge gradient (proxy for focus)
        edgemask = (self.label_image > 0) & ~(binary_erosion(self.label_image>0, square(3)))
        rescaled_intensity = (imgdata - bgval)/(fgval - bgval)
        grad = sobel(rescaled_intensity.astype(float))
        grad[~edgemask] = np.nan
        
        def average_edge_gradient(regionmask, intensity_image):
            return np.nanmean(intensity_image)
        
        self.flocs.update(regionprops_table(self.label_image, grad, properties=[],
                                     extra_properties=(average_edge_gradient,)))
        
        # convert dimensions
        self.flocs['area'] = self.flocs['area'] * self.resolution**2
        self.flocs['edgewidth'] = 1/self.flocs['average_edge_gradient']
        del self.flocs['average_edge_gradient']
        
        
    def get_floc_table(self, min_area=0, max_edgewidth=np.inf):
        
        # build dataframe
        ix = (self.flocs['edgewidth'] < max_edgewidth) & \
        (self.flocs['area'] > min_area)
        
        flocdf = pd.DataFrame(self.flocs).iloc[ix].reset_index(drop=True)

        return flocdf

# ===================================================================
# Functions
# ===================================================================


def run(flist, resolution, min_area, max_edgewidth, 
        method='difference', extra_params=[], index=None, save=False,
        n_jobs=1):
    
    # Instantiate image loader object
    load_img = ImgLoader(flist)
    
    def process_one(i):
        # report progress
        if i % 10 == 0:
            print('Processing {}/{} \t at \t{}'.format(i, len(flist), time.asctime()))
        
        if method=='difference':
            img_object = load_img.difference(i)
        elif method=='single':
            img_object = load_img.single(i)
        
        # get filename for saving
        fname = os.path.splitext(flist[i])[0]
        
        # perform floc ID
        seg = FlocID(img_object)
        
        # get floc dataframe and save
        flocdf = seg.get_floc_table(min_area, max_edgewidth)
        
        if save==True:
            flocdf.to_csv(fname+'.csv', index_label='floc_ID')
            
        return img_object, seg, flocdf
    
    if method=='difference':
        startix = 1
    elif method=='single':
        startix=0
    
    # if index isn't specified, iterate over all files in flist
    if index==None:
        iterlist = range(startix, len(flist))
    else:
        iterlist = [index,]
    
    if index!=None:
        out = process_one(index)
    elif n_jobs==1:
        [process_one(imgix) for imgix in iterlist]
    else:
        Parallel(n_jobs=n_jobs)(delayed(process_one)(imgix) for imgix in iterlist)
    

def filesorter(path, extension, sortby=os.path.basename):
    unsorted_flist = glob.glob(path+os.sep+'*'+extension)
    return sorted(unsorted_flist)


#%%

path='C:\\Users\\tashl\\OneDrive\\Documents\\GitHub\\pyfloc\\pyfloc\\testdata'

def custom_datetime(fpath, dtprefix='img-'):
    i0 = len(dtprefix)
    detlen = 14
    return dt.datetime.strptime(os.path.basename(fpath[i0:i0+dtlen],
                                                 '%m%d%Y%H%M%S'))

flist = filesorter(path, '.png', custom_datetime)

#%%
out = run(flist, 
          resolution=0.95, 
          min_area=0, 
          max_edgewidth=np.inf, 
          extra_params=[], 
          index=None,
          save=True,
          n_jobs=4)