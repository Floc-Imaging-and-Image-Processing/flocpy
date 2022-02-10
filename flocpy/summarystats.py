import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import linregress
import glob
from scipy.interpolate import interp1d
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import tqdm

# ===================================================================
# Classes
# ===================================================================

class FlocSample(object):
    def __init__(self, Df_list, ix, nframes, params, percentile):
        self.Df_list = Df_list
        self.ix = ix
        self.nframes = nframes
        
        Lf = params['focal_range']
        Af = params['frame_area']
        Dp = params['primary_particle_size']
        nf = params['fractal_dimension']
        rho_s = params['sediment_density']
        
        vnorm = Lf * Af * self.nframes
        
        #####
        # Calculate mass concentration (Cs (M/L^3))
        #####
        
        sedfrac = floc_sediment_fraction(self.Df_list, Dp, nf) # by volume
        Vf_frame = floc_volume_in_focal_range(self.Df_list, Lf) # volume of sedient+water
        Vs_frame = Vf_frame * sedfrac # volume of sediment only
        
        self.Cs = np.sum(Vs_frame) * rho_s / vnorm 
        
        #####
        # Calculate floc size percentiles weighted by sediment mass
        #####
        self.D_percentiles = percentile
        sorted_D = np.sort(self.Df_list)                   # sort grain size
        sorted_Vs = Vs_frame[np.argsort(self.Df_list)]     # sort volume
        
        # Create interpolation function
        logd_ref = np.log10(sorted_D)
        pct = 100 * np.cumsum(sorted_Vs)/np.sum(sorted_Vs)   # calculate volume percentiles
        fn = interp1d(pct, logd_ref, assume_sorted=True, bounds_error=False, fill_value=np.nan)
        
        if isinstance(percentile, int):
            percentile = [percentile,]
        
        # interpolate to specified percentiles
        log10_Df = np.array([fn(pct_i) for pct_i in percentile]) # interpolate to percentiles
        
        self.Df = 10**log10_Df


class FlocData(object):
    def __init__(self, flist, indices, sediment_density, primary_particle_size, fractal_dimension, 
                 frame_area, focal_range, min_floc_diameter, max_edgewidth, report_progress=True):
        
        self.flist = flist
        self.indices = indices
        self.params = {'frame_area': frame_area,
                       'focal_range': focal_range,
                       'sediment_density':sediment_density,
                       'primary_particle_size': primary_particle_size,
                       'fractal_dimension': fractal_dimension}
        
        min_area = np.pi * (min_floc_diameter/2)**2
        
        self.frames = np.empty(len(self.flist), 'object')
        
        if report_progress==True:
            print('Loading data')
            iterator = tqdm.tqdm(range(len(self.frames)))
        else: 
            iterator = range(len(self.frames))

        for i in iterator:
            fdf = pd.read_csv(flist[i])
            fdf = fdf[(fdf['edgewidth'] < max_edgewidth) & (fdf['area'] > min_area)]
            fdf['D'] = 2 * (fdf['area']/np.pi)**0.5
            self.frames[i] = fdf['D'].to_numpy()          
        
    def get_flocstats_table(self, percentile=[16, 50, 84], method='all', **kwargs):
        
        # returns all flocs as a single sample
        if method=='all':
            Dflist = []
            for i in range(len(self.indices)):
                Dflist.extend(self.frames[i].tolist())
            
            Dflist = np.array(Dflist)
            
            floc_samples = [FlocSample(Dflist, self.indices[0], len(self.indices),
                                     self.params, percentile),]
        
        elif method=='frame':
            floc_samples = []
            for i in range(len(self.indices)):
                floc_samples.append(FlocSample(self.frames[i], self.indices[i], 1,
                                             self.params, percentile))
            
        # returns irregularly spaced samples that have minimum number of flocs
        elif method=='count':
            # minimum number of flocs per sample
            assert 'minflocs' in kwargs
            
            Dflist = []
            ixlist = []
            floc_samples = []
            for i in range(len(self.indices)):
                ixlist.append(self.indices[i])
                Dflist.extend(self.frames[i].tolist())
                if len(Dflist) >= kwargs['minflocs']:
                    floc_samples.append(FlocSample(Dflist, ixlist[0], len(ixlist), 
                                                   self.params, percentile))
                    
                    Dflist = []
                    ixlist = []
        
        else: 
            print('method not found')
            
        #TODO: implement time-based sampling
        
        Darr = np.array([samp.Df.tolist() for samp in floc_samples])
        fdict = {'index': [samp.ix for samp in floc_samples],
                     'Cs': [samp.Cs for samp in floc_samples]}
        fdict.update({'Df_'+str(percentile[i]):Darr[:,i] for i in range(len(percentile))})
        
        return fdict


# ===================================================================
# Functions
# ===================================================================

@np.vectorize
def floc_volume_in_focal_range(Df, Lf):
    """
    Calculates expected value of the volume of a spherical floc with
    diameter D contained in the a focal volume with thickness Lf. Assumes
    the center of the floc resides somewhere within the focal volume, and that 
    all depths within the focal volume are equally probable.
    
    Arguments:
        Df: floc diameter or array of floc diameters
        Lf: the thickness of the focal volume
        
    Returns:
        out: a volume or array of volumes
        
    """
    
    R = Df/2
    if Df/2 > Lf:
        Vf = np.pi * (R**2 * Lf - Lf**3 / 6)
    else:
        Vf = np.pi * R**3 * (4/3 - (R/(2*Lf)))
    return Vf

@np.vectorize
def floc_sediment_fraction(Df, Dp, nf):
    """
    Returns fraction of the total floc volume occupied by sediment assuming
    a specified fractal dimension.
    
    Arguments:
        Df: floc diameter or array of floc diameters
        Dp: primary particle diameter
        nf: fractal dimension
    """
    return (Dp/Df)**(3-nf)



def calculate_summarystats(flist, frame_area, focal_range, min_floc_diameter, max_edgewidth,
                           index=None, sediment_density=2650000, primary_particle_size=5, fractal_dimension=2.4):
    
    if isinstance(index, type(None)):
        index = np.arange(len(flist))

    flocdata = FlocData(flist, index, sediment_density, primary_particle_size, fractal_dimension, 
                 frame_area, focal_range, min_floc_diameter, max_edgewidth)

    return flocdata