import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io
from skimage.filters import sobel, threshold_isodata
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square, disk, binary_erosion
from skimage.restoration import  denoise_wavelet
from skimage.color import label2rgb

from tqdm import tqdm
from joblib import Parallel, delayed

# ===================================================================
# Classes
# ===================================================================

class ImgSequence(object):
    
    def __init__(self, flist, resolution):
        """Instantiate object for loading images
        
        Arguments:
          - flist: Sorted list of files. Order is important if using
            image differencing.
            
        """
            
        self.flist = flist
        self.resolution = resolution
        
    def difference(self, index):
        """ Create FlocImg object using differencing algorithm """
        
        # load denoised images
        target_img = denoise_wavelet(io.imread(self.flist[index]))
        previous = denoise_wavelet(io.imread(self.flist[index-1]))
        
        # compute image difference
        img = target_img - previous
        img[img > 0] = 0
        
        if np.isnan(img).any():
            threshold = 0
            bgval = 0
            fgval = 0
            data = np.ones(target_img.shape)
            
        else:
            # calculate meta parameters
            threshold = threshold_isodata(target_img)
            bgval = np.nanmean(target_img[target_img > threshold])
            fgval = np.nanmean(target_img[target_img < threshold])
    
            # rescale image
            data = bgval+img
            data[data<0] = 0


        return FlocImg(data, bgval, fgval, threshold, self.resolution, target_img)
        

    def single(self, index):
        """ Create FlocImg object using a single denoised image """
        data = denoise_wavelet(io.imread(flist[imgix]))
        # TODO: implement rolling ball filter for removing background
            # calculate meta parameters
        if np.isnan(data).any():
            threshold = 0
            bgval = 0
            fgval = 0
            data = np.ones(target_img.shape)
        else:
            threshold = threshold_isodata(data)
            bgval = np.nanmean(data[data > threshold])
            fgval = np.nanmean(data[data < threshold])
        
        return FlocImg(data, bgval, fgval, threshold, self.resolution)

class FlocImg(object):
    
    def __init__(self, data, bgval, fgval, threshold, resolution, raw_img = None):
        """
        Image data object with metadata
        
        Parameters:
            - data: the M x N image data.
            - bgval: average pixel value of the background
            - fgval: average pixel value of flocs
            - threshold: value separating flocs from background
        
        """
        self.data = data
        self.bgval = bgval
        self.fgval = fgval
        self.threshold = threshold
        self.resolution = resolution
        self.raw_img = raw_img
        
    def identify_flocs(self, extra_params=[]):
        """ Performs floc identification on target image
        
        """
        
        # compute threshold using isodata method and segment image
        self.segmentation = closing(self.data<self.threshold, disk(1))
        
        # remove artifacts connected to image border
        self.cleared = clear_border(self.segmentation, buffer_size=1)
        
        # label image regions
        self.label_image, self.nflocs = label(self.cleared, return_num=True)
        
                
        # make dictionary of floc info
        self.flocs = regionprops_table(self.label_image,
                                  properties=['area']+extra_params)
        
        # calculate edge gradient (proxy for focus)
        edgemask = (self.label_image > 0) & ~(binary_erosion(self.label_image>0, square(3)))
        rescaled_intensity = (self.data - self.bgval)/(self.fgval - self.bgval)
        grad = sobel(rescaled_intensity.astype(float))
        grad[~edgemask] = np.nan
        self.grad=grad
        
        def average_edge_gradient(regionmask, intensity_image):
            return np.nanmean(intensity_image)

        self.flocs.update(regionprops_table(self.label_image, grad, properties=[],
                                     extra_properties=(average_edge_gradient,)))
        
        # convert dimensions
        self.flocs['area'] = self.flocs['area'] * self.resolution**2
        self.flocs['edgewidth'] = 1/self.flocs['average_edge_gradient']
        del self.flocs['average_edge_gradient']
        
        
    def get_floc_table(self, min_diameter=0, max_edgewidth=np.inf):
        min_area = np.pi * (min_diameter/2)**2

        # build dataframe
        ix = (self.flocs['edgewidth'] < max_edgewidth) & \
        (self.flocs['area'] > min_area)
        
        flocdf = pd.DataFrame(self.flocs).iloc[ix].reset_index(drop=True)

        return flocdf

    def get_regions(self, min_diameter=0, max_edgewidth=np.inf):
        min_area = np.pi * (min_diameter/2)**2
        regions = regionprops(self.label_image)
        ix = (self.flocs['edgewidth'] < max_edgewidth) & \
        (self.flocs['area'] > min_area)
        return np.array(regions)[ix]

    def plot_segmentation(self, ax, min_diameter=0, max_edgewidth=np.inf,
                      use_raw_img=True, show_overlay=True, show_boxes=True,  box_color='k', box_thickness=0.5):

        if use_raw_img:
            plotimg = self.raw_img
        else:
            plotimg = self.data


        if show_overlay:
            image_label_overlay = label2rgb(self.label_image, image=plotimg, bg_label=0)
            ax.imshow(image_label_overlay)
        else:
            ax.imshow(plotimg, cmap='Greys_r')

        if show_boxes:
            regions = self.get_regions(min_diameter, max_edgewidth)
            
            for region in regions:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                          fill=False, edgecolor=box_color, linewidth=box_thickness)
                ax.add_patch(rect)



# ===================================================================
# Functions
# ===================================================================


def identify_flocs(flist, resolution=1, min_diameter=0, max_edgewidth=np.inf, 
        method='difference', extra_params=[], index=None, save=False, return_data=False,
        n_jobs=1, report_progress=True):

    # Instantiate image loader object
    load_img = ImgSequence(flist, resolution)
    
    def process_one(i):
        # get filename for saving
        fname = os.path.splitext(flist[i])[0]
        
        # instantiate image object using chosen method
        if method=='difference':
            floc_img = load_img.difference(i)
        elif method=='single':
            floc_img = load_img.single(i)
            
        # perform floc ID
        floc_img.identify_flocs(extra_params)
        
        
        # get dataframe of parameters and save

        if save==True:
            flocdf = floc_img.get_floc_table(min_diameter, max_edgewidth)
            flocdf.to_csv(fname+'.csv', index_label='floc_ID')
        
        if return_data:
            return floc_img
    
    if method=='difference':
        startix = 1
    elif method=='single':
        startix=0
    
    # if index isn't specified, iterate over all files in flist
    if isinstance(index, np.integer):
        iterlist = [index,]
    elif isinstance(index, type(None)):
        iterlist = range(len(flist))
    else: 
        iterlist = index
    

    if isinstance(index, np.integer):
        out =  process_one(index)
    elif n_jobs==1:
        if report_progress==True:
            iterator = tqdm(iterlist)
        else:
            iterator = iterlist
        out = [process_one(imgix) for imgix in iterator]
    else:
        if report_progress==True:
            out = Parallel(n_jobs=n_jobs,verbose=10)(delayed(process_one)(imgix) for imgix in iterlist)
        else:
            out = Parallel(n_jobs=n_jobs)(delayed(process_one)(imgix) for imgix in iterlist)

    return out
