import pyfloc
import os
import numpy as np


__all__ = ['dotest']

def dotest():
    # File name formatting string things
    datadir = pyfloc.__path__[0]+os.sep+'testdata'

    flist = file_sorter(path, extension='.bmp')
    
    out = run(flist, 
              resolution=0.95,          # units per pixel
              min_area=100,             # minimum floc size to save (units)
              max_edgewidth=5,          # maximum value of edge width to save (proxy for focus)
              extra_params=[],          # specify other parameters to save
              index=1,                  # process a single image
              save=False,               # save results as a csv
              n_jobs=1)                 # parallel processing. Set n_jobs=1 to skip parallel processing.

    print('Single Core Test Successful.', end='\t\t\t\t\n')